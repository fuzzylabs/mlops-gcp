import argparse
from init import init
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google AI Platform helper utility")
    parser.add_argument('command', choices=["init", "train"])

    args = parser.parse_args()

    print(args)
    if args.command == "init":
        init()
    elif args.command == "train":
        train()
