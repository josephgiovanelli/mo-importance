import json
import os
import copy
import subprocess
from tqdm import tqdm

from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from utils.common import get_tuning_datasets, make_dir

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "input"))
    subprocess.call("python src/scenario_generator.py", shell=True)

    confs = [p for p in os.listdir(input_path) if ".json" in p]
    print("--- PRELIMINAR SAMPLING ---")
    with tqdm(total=len(confs)) as pbar:
        for conf in confs:
            subprocess.call(
                f"python src/preliminar_sampling.py --conf_file {conf}", shell=True
            )
            subprocess.call(
                f"python src/automatic_ordering.py --conf_file {conf}", shell=True
            )
            pbar.update()
    print("--- PREFERENCE LEARNING ---")
    subprocess.call(f"python src/preference_learning.py", shell=True)

    evaluation_confs = [elem for elem in confs if elem not in get_tuning_datasets()]
    print("--- OPTIMIZATION LOOP ---")
    with tqdm(total=len(evaluation_confs)) as pbar:
        for conf in evaluation_confs:
            subprocess.call(
                f"python src/optimization.py --conf_file {conf}", shell=True
            )
            subprocess.call(f"python src/comparison.py --conf_file {conf}", shell=True)
