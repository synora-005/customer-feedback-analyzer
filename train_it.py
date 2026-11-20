"""Fine-tune GPT-2 on the labelled feedback: `python train.py` (roughly 5 to 15 minutes on a laptop CPU)."""
import argparse
import json
import time
from pathlib import Path

from app.core import (MODEL_DIR, add_category_tokens, load_feedback, load_gpt2, save_classifier,
                      split_dataset, train_classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, val, test = split_dataset(load_feedback("dataset/feedback.csv"))
    print(f"train {len(train)} / validation {len(val)} / test {len(test)} (the test set is not used here)")

    tokenizer, model = load_gpt2()
    add_category_tokens(tokenizer, model)
    started = time.time()
    history = train_classifier(model, tokenizer, train, val, epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size, seed=args.seed)
    minutes = round((time.time() - started) / 60, 1)

    save_classifier(model, tokenizer, MODEL_DIR)
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(json.dumps(
        {"epochs": args.epochs, "learning_rate": args.lr, "batch_size": args.batch_size, "seed": args.seed,
         "train_examples": len(train), "validation_examples": len(val), "minutes": minutes, **history},
        indent=2))
    print("validation accuracy per epoch:", [round(a, 3) for a in history["eval_accuracy"]])
    print(f"saved the model to {MODEL_DIR} and the numbers to results/training.json ({minutes} min)")