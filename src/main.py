# src/main.py
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # CPU only here; determinism flags not needed
    except Exception:
        pass

def main():
    import argparse
    p = argparse.ArgumentParser(description="RL-Dewey-Tutor entry")
    p.add_argument("cmd", choices=["train", "eval"], help="what to run")
    p.add_argument("--seed", type=int, default=42, help="global seed")
    args = p.parse_args()

    set_seed(args.seed)

    if args.cmd == "train":
        from src.train import main as train_main
        train_main()
    elif args.cmd == "eval":
        from src.evaluate import main as eval_main
        eval_main()

if __name__ == "__main__":
    main()
