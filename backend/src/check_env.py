import argparse
import subprocess

import torch


def add_args(parser):
    return parser


def run(_args):
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        output = result.stdout.strip()
        if output:
            print(output)
        else:
            err = result.stderr.strip()
            print(err if err else "nvidia-smi not available")
    except FileNotFoundError:
        print("nvidia-smi not found")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Check GPU and torch setup")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
