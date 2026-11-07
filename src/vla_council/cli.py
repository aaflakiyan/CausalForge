"""CLI: collect, train, refine, evaluate. All commands run toy simulation only."""
import argparse
import json
from pathlib import Path
from .pipeline import collect, load_dataset, evaluate, refine
from .student import LinearStudent

def main():
    parser = argparse.ArgumentParser(description="VLA Council: toy simulation-only proof of concept")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("collect"); a.add_argument("--episodes", type=int, default=100); a.add_argument("--output", type=Path, default=Path("runs/demos.jsonl"))
    a = sub.add_parser("train"); a.add_argument("--dataset", type=Path, default=Path("runs/demos.jsonl")); a.add_argument("--output", type=Path, default=Path("runs/student.json"))
    a = sub.add_parser("refine"); a.add_argument("--dataset", type=Path, default=Path("runs/demos.jsonl")); a.add_argument("--student", type=Path, default=Path("runs/student.json")); a.add_argument("--output", type=Path, default=Path("runs/student_refined.json")); a.add_argument("--episodes", type=int, default=30)
    a = sub.add_parser("evaluate"); a.add_argument("--student", type=Path, default=Path("runs/student.json")); a.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.episodes, args.output)
    elif args.command == "train":
        obs, acts = load_dataset(args.dataset)
        LinearStudent.fit(obs, acts).save(args.output)
        result = {"examples": len(acts), "model": str(args.output)}
    elif args.command == "refine":
        result = refine(LinearStudent.load(args.student), args.dataset, args.output, args.episodes)
    else:
        result = evaluate(LinearStudent.load(args.student), args.episodes)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
