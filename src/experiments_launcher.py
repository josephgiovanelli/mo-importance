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
    input_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "input"))
    common_log_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "logs"))
    log_paths = {
        "preliminar_sampling": make_dir(
            os.path.join(common_log_path, "preliminar_sampling")
        ),
        "automatic_ordering": make_dir(
            os.path.join(common_log_path, "automatic_ordering")
        ),
        "preference_learning": make_dir(
            os.path.join(common_log_path, "preference_learning_eval")
        ),
        "optimization": make_dir(os.path.join(common_log_path, "optimization")),
        "comparison": make_dir(os.path.join(common_log_path, "comparison")),
        "summarizer": make_dir(os.path.join(common_log_path, "summarizer")),
        "plotter": make_dir(os.path.join(common_log_path, "plotter")),
    }

    print("--- SCENARIO GENERATION ---")
    subprocess.call("python src/scenario_generator.py", shell=True)

    confs = [p for p in os.listdir(input_path) if ".json" in p]
    print("--- PRELIMINAR SAMPLING ---")
    with tqdm(total=len(confs)) as pbar:
        for conf in confs:
            log_file_name = conf.split(".")[0]
            subprocess.call(
                f"""python src/preliminar_sampling.py --conf_file {conf} > {log_paths["preliminar_sampling"]}/{log_file_name}_out.txt""",
                shell=True,
            )
            subprocess.call(
                f"""python src/automatic_ordering.py --conf_file {conf} > {log_paths["automatic_ordering"]}/{log_file_name}_out.txt""",
                shell=True,
            )
            pbar.update()
    print("--- PREFERENCE LEARNING ---")
    subprocess.call(
        f"""python src/preference_learning_eval.py > {log_paths["preference_learning"]}/out.txt""",
        shell=True,
    )

    evaluation_confs = [elem for elem in confs if elem not in get_tuning_datasets()]
    print("--- OPTIMIZATION LOOP ---")
    with tqdm(total=len(evaluation_confs)) as pbar:
        for conf in evaluation_confs:
            log_file_name = conf.split(".")[0]
            subprocess.call(
                f"""python src/optimization.py --conf_file {conf} > {log_paths["optimization"]}/{log_file_name}_out.txt""",
                shell=True,
            )
            subprocess.call(
                f"""python src/comparison.py --conf_file {conf} > {log_paths["comparison"]}/{log_file_name}_out.txt""",
                shell=True,
            )
            pbar.update()

    subprocess.call(
        f"""python src/summarizer.py > {log_paths["summarizer"]}/{log_file_name}_out.txt""",
        shell=True,
    )
    subprocess.call(
        f"""python src/plotter.py > {log_paths["plotter"]}/{log_file_name}_out.txt""",
        shell=True,
    )
