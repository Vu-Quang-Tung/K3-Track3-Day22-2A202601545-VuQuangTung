from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")

def _heuristic_score(prompt: str, response: str) -> float:
    """Simple heuristic scorer: keyword overlap ratio minus penalty for very short responses."""
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    if not prompt_words:
        return 0.0
    overlap = len(prompt_words & response_words) / len(prompt_words)
    
    length_penalty = -0.5 if len(response_words) < 5 else 0.0
    return float(overlap + length_penalty)


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")

@app.command()
def evaluate(config: Path = typer.Option(..., "--config", help="Path to config file")) -> None:  # noqa: B008
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [_heuristic_score(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [_heuristic_score(ex.prompt, ex.rejected) for ex in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")

@app.command()
def train(config: Path = typer.Option(..., "--config", help="Path to config file")) -> None:  # noqa: B008
    cfg = load_config(config)
    train_cfg = TrainingConfig(
        method=cfg.get("training", {}).get("method", "DPO"),
        output_dir=cfg.get("paths", {}).get("output_dir", "outputs"),
        beta=cfg.get("training", {}).get("beta", 0.1),
        lambda_orpo=cfg.get("training", {}).get("lambda_orpo", 0.1),
        max_length=cfg.get("training", {}).get("max_length", 512),
        batch_size=cfg.get("training", {}).get("batch_size", 2),
    )
    trainer = PreferenceTrainer(train_cfg)
    trainer.train()

if __name__ == "__main__":
    app()
