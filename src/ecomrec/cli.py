"""CLI: ingest, train, serve, agent, mcp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecomrec.config import settings


def _cmd_ingest(args: argparse.Namespace) -> None:
    from ecomrec.data.pipeline import run_pipeline

    csv_path = Path(args.csv) if args.csv else None
    if args.rows:
        settings.sample_rows = int(args.rows)
    stats = run_pipeline(csv_path=csv_path, use_postgres=None if not args.pandas_only else False)
    print(json.dumps(stats, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    from ecomrec.models.train import run_training

    if args.epochs:
        settings.epochs = int(args.epochs)
    if args.dim:
        settings.embedding_dim = int(args.dim)
    metrics = run_training(settings)
    print(json.dumps({k: v for k, v in metrics.items() if k != "train_loss"}, indent=2))
    print(f"train_loss={metrics.get('train_loss')}")


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("ecomrec.serving.api:app", host=args.host, port=args.port, reload=False)


def _cmd_agent(args: argparse.Namespace) -> None:
    from ecomrec.agent.graph import run_turn
    from ecomrec.models.infer import RecommendationService

    service = RecommendationService.load()
    user_id = args.user_id
    print("E-commerce recommendation agent. Empty line to exit.")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if not message:
            break
        state = run_turn(service, message, user_id=user_id)
        print(f"[{state.get('route')}] {state.get('reply')}\n")


def _cmd_mcp(_args: argparse.Namespace) -> None:
    from ecomrec.serving.mcp_server import run_mcp

    run_mcp()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ecomrec")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load events and write train/val/test splits")
    ingest.add_argument("--csv", type=str, default=None, help="Kaggle-style events CSV")
    ingest.add_argument("--rows", type=int, default=None, help="Synthetic row count if no CSV")
    ingest.add_argument("--pandas-only", action="store_true", help="Skip PostgreSQL even if running")
    ingest.set_defaults(func=_cmd_ingest)

    train = sub.add_parser("train", help="Train BPR-MF and export embeddings")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--dim", type=int, default=None)
    train.set_defaults(func=_cmd_train)

    serve = sub.add_parser("serve", help="Run the FastAPI recommendation server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    agent = sub.add_parser("agent", help="Conversational LangGraph agent")
    agent.add_argument("--user-id", type=int, default=None)
    agent.set_defaults(func=_cmd_agent)

    mcp = sub.add_parser("mcp", help="Run the MCP tool server on stdio")
    mcp.set_defaults(func=_cmd_mcp)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
