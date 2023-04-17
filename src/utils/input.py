import os
import json

from utils.dataset import load_dataset_from_openml


class ConfDict(dict):
    def __new__(cls, conf=None):
        if not hasattr(cls, "instance"):
            cls.instance = super(ConfDict, cls).__new__(cls)
        return cls.instance


def create_configuration(file_name: str):
    with open(os.path.join("input", file_name)) as file:
        conf = json.load(file)

    conf["output_folder"] = os.path.join("output", conf["output_folder"])
    conf["obj_metrics"] = [c["metric"] for c in conf["objectives"]]
    conf["obj_modes"] = [c["mode"] for c in conf["objectives"]]
    conf["obj_upper_bounds"] = [c["upper_bound"] for c in conf["objectives"]]
    conf["obj_lower_bound"] = [c["lower_bound"] for c in conf["objectives"]]
    conf["X"], conf["y"], _ = load_dataset_from_openml(conf["dataset"])

    ConfDict(conf)
