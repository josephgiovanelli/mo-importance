# %%
import random

from csrank.dataset_reader import ChoiceDatasetGenerator, ObjectRankingDatasetGenerator
import csrank as cs
import numpy as np

random.seed(42)
np.random.seed(42)


def modify(original):
    return np.array(
        [
            np.array([np.array([object] * 10) for object in list(instance)])
            for instance in list(original)
        ]
    )


gen = ObjectRankingDatasetGenerator(n_instances=20, n_objects=2, n_features=2)
# gen = ChoiceDatasetGenerator(
#     dataset_type="pareto", n_instances=50, n_objects=10, n_features=2
# )
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()

# X_train, X_test = modify(X_train), modify(X_test)
print(X_train.shape, X_test.shape)
print(type(X_train))
print(Y_test.shape)
print(np.all(np.all(np.array([0, 1]) == Y_train, axis=1)))
# print(
#     [
#         y
#         for x in X_train
#         for y in x
#         if np.allclose(y, np.array([[-1.55302329, -1.31248661]]))
#         # if np.allclose(y, np.array([0.45470105, 0.83263203]))
#         # if np.allclose(y, np.array([0.45470105]))
#     ]
# )

# %%
fate = cs.ExpectedRankRegression()
fate.fit(X_train[:20], Y_train[:20])

print(fate.predict(X_test))

# %%
