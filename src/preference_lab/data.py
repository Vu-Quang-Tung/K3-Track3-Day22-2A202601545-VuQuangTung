from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    TODO(student): add line-numbered errors, duplicate prompt checks, and optional PII guardrails.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON không hợp lệ - {exc}") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{path}:{line_no}: schema không hợp lệ - {exc}") from exc
            
            if example.prompt in seen_prompts:
                raise ValueError(f"{path}:{line_no}: prompt bị trùng lặp")
            seen_prompts.add(example.prompt)
            examples.append(example)
    return examples

def split_by_prompt(examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage."""
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex.prompt].append(ex)
        
    prompts = list(grouped.keys())
    random.Random(seed).shuffle(prompts)
    
    val_count = max(1, int(len(prompts) * validation_ratio))
    val_prompts = set(prompts[:val_count])
    
    train, val = [], []
    for ex in examples:
        if ex.prompt in val_prompts:
            val.append(ex)
        else:
            train.append(ex)
            
    return train, val
