import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-conf_file",
        "--conf_file",
        nargs="?",
        type=str,
        default="trial.json",
        help="configuration file name",
    )

    return parser.parse_known_args()
