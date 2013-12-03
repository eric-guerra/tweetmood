"""python -m tweetmood.cli {evaluate,train,predict,serve}"""
from __future__ import annotations

import argparse
import json

from .model import DATA, SentimentModel, evaluate, load_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tweetmood")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--data", default=str(DATA))
    e.add_argument("--folds", type=int, default=5)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(DATA))
    t.add_argument("-o", "--out", default="model.joblib")
    pr = sub.add_parser("predict")
    pr.add_argument("text", nargs="+")
    pr.add_argument("--model", default="model.joblib")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=5000)
    args = p.parse_args(argv)

    if args.cmd == "evaluate":
        texts, labels = load_dataset(args.data)
        ev = evaluate(texts, labels, folds=args.folds)
        print(f"{len(texts)} tweets, {args.folds}-fold CV")
        print(f"accuracy {ev.accuracy:.3f} (majority baseline {ev.baseline_accuracy:.3f}), macro-F1 {ev.macro_f1:.3f}\n")
        print(ev.report)
        print("confusion (rows=true neg/neu/pos):", ev.confusion)
    elif args.cmd == "train":
        model = SentimentModel.train(*load_dataset(args.data))
        model.save(args.out)
        print(f"saved {args.out}")
    elif args.cmd == "predict":
        for pred in SentimentModel.load(args.model).predict(args.text):
            print(json.dumps(pred))
    elif args.cmd == "serve":
        from .app import create_app
        create_app().run(port=args.port)


if __name__ == "__main__":
    main()
