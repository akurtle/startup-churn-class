from __future__ import annotations

import argparse
import json

import uvicorn

from startup_churn_classifier.training import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Startup churn classifier CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run model training and artifact generation.")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI inference service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "train":
        print(json.dumps(run_training_pipeline(), indent=2))
        return 0

    if args.command == "serve":
        uvicorn.run(
            "startup_churn_classifier.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
