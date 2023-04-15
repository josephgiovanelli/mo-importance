import os
import json


def read_configuration(file_name: str):
    with open(os.path.join("input", file_name)) as file:
        conf = json.load(file)
    conf["output_folder"] = os.path.join("output", conf["output_folder"])
    conf["obj_metrics"] = [c["metric"] for c in conf["objectives"]]
    conf["obj_modes"] = [c["mode"] for c in conf["objectives"]]
    conf["obj_upper_bounds"] = [c["upper_bound"] for c in conf["objectives"]]
    conf["obj_lower_bound"] = [c["lower_bound"] for c in conf["objectives"]]
    return conf
