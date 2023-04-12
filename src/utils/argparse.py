import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-dataset",
        "--dataset",
        nargs="?",
        type=str,
        default="iris",
        help="dataset to analise",
    )
    parser.add_argument(
        "-metrics",
        "--metrics",
        nargs="+",
        type=str,
        default=["accuracy", "balanced_accuracy"],
        help="metrics to optimize",
    )
    parser.add_argument(
        "-modes",
        "--modes",
        nargs="+",
        type=str,
        default=["max", "min"],
        help="either minimize or maximize the metric (min or max)",
    )
    parser.add_argument(
        "-time_budget",
        "--time_budget",
        nargs="?",
        type=int,
        default=30,
        help="time budget in seconds",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        nargs="?",
        type=int,
        default=200,
        help="time budget in seconds",
    )
    parser.add_argument(
        "-output_path",
        "--output_path",
        nargs="?",
        type=str,
        default=os.path.join("output", "trial"),
        help="path to the automl ouput",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        nargs="?",
        type=int,
        default=0,
        help="seed for reproducibility",
    )

    args = parser.parse_args()
    return args
