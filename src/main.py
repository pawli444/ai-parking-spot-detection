import argparse

from check_env import add_args as add_check_args, run as run_check
from dataset_stats import add_args as add_stats_args, run as run_stats
from train import add_args as add_train_args, run as run_train
from evaluate import add_args as add_eval_args, run as run_eval
from infer_video import add_args as add_infer_args, run as run_infer
from export_onnx import add_args as add_export_args, run as run_export


def build_parser():
    parser = argparse.ArgumentParser(description="Parking spot detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-env", help="Show GPU and torch info")
    add_check_args(check_parser)

    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    add_stats_args(stats_parser)

    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    add_train_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate on test split")
    add_eval_args(eval_parser)

    infer_parser = subparsers.add_parser("infer-video", help="Run inference on a video")
    add_infer_args(infer_parser)

    export_parser = subparsers.add_parser("export-onnx", help="Export model to ONNX")
    add_export_args(export_parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check-env":
        run_check(args)
    elif args.command == "stats":
        run_stats(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "infer-video":
        run_infer(args)
    elif args.command == "export-onnx":
        run_export(args)


if __name__ == "__main__":
    main()
