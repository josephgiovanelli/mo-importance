# %%
import random

from csrank.dataset_reader import ChoiceDatasetGenerator, ObjectRankingDatasetGenerator
import csrank as cs
import numpy as np

random.seed(42)
np.random.seed(42)

gen = ObjectRankingDatasetGenerator(n_instances=50, n_objects=3, n_features=2)
# gen = ChoiceDatasetGenerator(
#     dataset_type="pareto", n_instances=50, n_objects=10, n_features=2
# )
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()

print(Y_train)
print(X_train)
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
fate = cs.RankSVM()
fate.fit(X_train, Y_train)

print(fate.predict(X_test))

# %%
