import os
import json
from utils.common import make_dir

from utils.dataset import load_dataset_from_openml


class ConfDict(dict):
    def __new__(cls, conf=None):
        if not hasattr(cls, "instance"):
            cls.instance = super(ConfDict, cls).__new__(cls)
        return cls.instance


def create_configuration(file_name: str, origin: str = "preliminar_sampling"):
    with open(os.path.join("/", "home", "input", file_name)) as file:
        conf = json.load(file)

    conf["output_folder"] = os.path.join(
        "/",
        "home",
        "output",
        conf["output_folder"],
        origin,
    )
    make_dir(conf["output_folder"])

    conf["objectives"] = [conf["performance_objective"], conf["use_case_objective"]]
    conf["obj_metrics"] = [c["metric"] for c in conf["objectives"]]
    conf["obj_modes"] = [c["mode"] for c in conf["objectives"]]
    if conf["model"] != "lcbench":
        (
            conf["X"],
            conf["y"],
            conf["categorical_indicator"],
            conf["feature_names"],
        ) = load_dataset_from_openml(conf["dataset"])
        if "sensitive_feature" in conf["use_case_objective"]:
            conf["use_case_objective"]["sensitive_feature_idx"] = conf[
                "feature_names"
            ].index(conf["use_case_objective"]["sensitive_feature"])

    ConfDict(conf)
