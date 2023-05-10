from hpobench.benchmarks.nas.nasbench_201 import (
    Cifar10ValidNasBench201BenchmarkOriginal,
)

b = Cifar10ValidNasBench201BenchmarkOriginal()

space = b.get_configuration_space(seed=1)
print(space)
config = space.sample_configuration()
print(config)

result_dict = b.objective_function_test(
    configuration=config, fidelity={"epoch": 12}, rng=1
)
print(result_dict)
