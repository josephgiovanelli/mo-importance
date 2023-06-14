import json
import os
import copy
import subprocess
from tqdm import tqdm

from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from utils.common import make_dir

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "input"))

    subprocess.call("python src/scenario_generator.py", shell=True)
    confs = [p for p in os.listdir(input_path) if ".json" in p]
    with tqdm(total=len(confs)) as pbar:
        for conf in confs:
            subprocess.call(
                f"python src/preliminar_sampling.py --conf_file {conf}", shell=True
            )
            subprocess.call(
                f"python src/automatic_ordering.py --conf_file {conf}", shell=True
            )
            pbar.update()
