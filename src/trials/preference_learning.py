from csrank.dataset_reader import ChoiceDatasetGenerator
import csrank as cs

gen = ChoiceDatasetGenerator(dataset_type="pareto", n_objects=30, n_features=2)
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()

print(X_train.shape)
print(Y_train.shape)

fate = cs.FATEChoiceFunction()
fate.fit(X_train, Y_train)

print(fate.predict(X_test))
