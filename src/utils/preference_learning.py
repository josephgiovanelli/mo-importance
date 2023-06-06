import os

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


from ConfigSpace import Configuration
from ConfigSpace.conditions import EqualsCondition, InCondition, NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from ranker.my_rank_svc import MyRankSVM

from utils.common import make_dir
from utils.input import ConfDict
from utils.output import (
    load_encoded,
    load_preferences,
)


def create_preference_dataset(preference_path):
    flatten_encoded = {
        key: np.array(value).flatten()
        for key, value in load_encoded(preference_path).items()
    }

    # paretos = np.array(list(flatten_encoded.values()))
    # # for n_clusters in range(3, len(paretos)):
    # n_clusters = 6
    # kmedoids = KMedoids(n_clusters=n_clusters, random_state=ConfDict()["seed"]).fit(
    #     paretos
    # )
    # centers = kmedoids.cluster_centers_

    # # print(n_clusters)
    # # print(silhouette_score(paretos, kmedoids.labels_))
    # print(np.where(flatten_encoded == [elem for elem in centers]))
    # # print()

    preferences = load_preferences(path=preference_path)
    X = np.array(
        [
            np.array([flatten_encoded[str(pair[0])], flatten_encoded[str(pair[1])]])
            for pair in preferences[["pair_0", "pair_1"]].to_numpy()
        ]
    )
    y = np.array(
        [
            # np.array([0, 1] if preference == 0 else [1, 0])
            preference
            for preference in preferences["preference"].to_numpy()
        ]
    )
    return X, y, preferences


def configspace() -> ConfigurationSpace:
    # C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    C = UniformFloatHyperparameter("C", 0.8, 1.5, log=False, default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    dual = CategoricalHyperparameter("dual", ["True", "False"], default_value="False")
    loss = CategoricalHyperparameter(
        "loss", ["squared_hinge", "hinge"], default_value="squared_hinge"
    )
    penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l1")

    # n_features = UniformIntegerHyperparameter(
    #     "n_features", 1, 19, default_value=19, log=False
    # )
    # svm_implementation = CategoricalHyperparameter(
    #     "svm_implementation",
    #     ["logistic", "linear"],
    #     default_value="linear",
    # )
    # features_implementation = CategoricalHyperparameter(
    #     "features_implementation",
    #     ["selection", "pca", "none"],
    #     default_value="none",
    # )
    normalize = CategoricalHyperparameter(
        "normalize", ["True", "False"], default_value="False"
    )

    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            C,
            tol,
            dual,
            loss,
            penalty,
            # n_features,
            # svm_implementation,
            # features_implementation,
            normalize,
        ]
    )

    # loss_condition = EqualsCondition(loss, svm_implementation, "linear")
    # dual_condition = EqualsCondition(dual, svm_implementation, "linear")
    # svm_implementation_condition = EqualsCondition(svm_implementation, penalty, "l2")
    # n_features_condition = InCondition(
    #     n_features, features_implementation, ["selection", "pca"]
    # )
    # cs.add_condition(dual_condition)
    # cs.add_condition(loss_condition)
    # cs.add_condition(svm_implementation_condition)
    # cs.add_condition(n_features_condition)
    return cs


def compute_raw_results(config_dict, result_dict, feat_dict, mode, seed):
    if mode == "cross_validation":
        splits = KFold(n_splits=10, random_state=seed).split(ConfDict()["X"])
    else:
        splits = [
            train_test_split(
                range(ConfDict()["X"].shape[0]),
                test_size=0.33,
                random_state=seed,
            )
        ]
    raw_results = []
    for train, test in splits:
        fate = MyRankSVM(**config_dict, random_state=seed)
        fate.fit(ConfDict()["X"][train].copy(), ConfDict()["Y"][train].copy())

        raw_results.append(
            pd.DataFrame(
                {
                    "pair_0": [pref[0] for pref in ConfDict()["preferences"][test]],
                    "pair_0_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array([[elem[0]] for elem in ConfDict()["X"][test]])
                        )
                        for elem in list
                    ],
                    "pair_1": [pref[1] for pref in ConfDict()["preferences"][test]],
                    "pair_1_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array([[elem[1]] for elem in ConfDict()["X"][test]])
                        )
                        for elem in list
                    ],
                    "y_true": ConfDict()["Y"][test],
                    "y_pred": fate.predict(ConfDict()["X"][test].copy()),
                }
            )
        )

    feat_dict[mode] = fate.features_expl_
    result_dict[mode] = np.mean(
        [accuracy_score(result["y_true"], result["y_pred"]) for result in raw_results]
    )
    pd.concat(raw_results, ignore_index=True).to_csv(
        os.path.join(
            make_dir(os.path.join(ConfDict()["output_folder"], f"{mode}")),
            f"""predictions_{ConfDict()["iteration"]}.csv""",
        ),
        index=False,
    )


def objective(config: Configuration, seed: int = 0) -> float:
    config_dict = config.get_dictionary()
    result_dict = {
        "iteration": ConfDict()["iteration"],
        "cross_validation": np.nan,
        "train_test": np.nan,
    }
    feat_dict = {
        "cross_validation": "",
        "train_test": "",
    }

    try:
        for mode in result_dict.keys():
            if mode != "iteration":
                compute_raw_results(config_dict, result_dict, feat_dict, mode, seed)

        log = "success"
    except Exception as e:
        log = e

    ConfDict(
        {
            "summary": pd.concat(
                [
                    ConfDict()["summary"],
                    pd.DataFrame(
                        {
                            **{key: [value] for key, value in result_dict.items()},
                            **{key: [value] for key, value in config_dict.items()},
                            **{
                                f"feat_{key}": [value]
                                for key, value in feat_dict.items()
                            },
                            **{"log": [log]},
                        }
                    ),
                ],
                ignore_index=True,
            )
        }
    )

    ConfDict({"iteration": ConfDict()["iteration"] + 1})

    return 1 - result_dict["cross_validation"]