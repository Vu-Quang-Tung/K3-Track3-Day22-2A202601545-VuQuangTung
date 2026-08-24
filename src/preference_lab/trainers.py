from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .losses import dpo_loss, orpo_loss

@dataclass(frozen=True)
class TrainingConfig:
    method: str
    output_dir: str = "outputs"
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    epochs: int = 3
    num_examples: int = 10

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Mock trainer for CPU that writes loss history to output_dir."""
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        history = []
        num_batches = max(1, self.config.num_examples // self.config.batch_size)
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            for batch in range(num_batches):
                # Generate mock logprobs
                if self.config.method.upper() == "DPO":
                    p_chosen = np.random.uniform(-2.0, -0.1, size=self.config.batch_size)
                    p_rejected = np.random.uniform(-3.0, -1.0, size=self.config.batch_size)
                    r_chosen = np.random.uniform(-1.5, -0.5, size=self.config.batch_size)
                    r_rejected = np.random.uniform(-1.5, -0.5, size=self.config.batch_size)
                    loss = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, self.config.beta)
                else:
                    sft_nll = np.random.uniform(0.1, 1.0, size=self.config.batch_size)
                    c_logps = np.random.uniform(-1.5, -0.1, size=self.config.batch_size)
                    r_logps = np.random.uniform(-3.0, -1.0, size=self.config.batch_size)
                    loss = orpo_loss(sft_nll, c_logps, r_logps, self.config.lambda_orpo)
                
                epoch_loss += loss
                history.append({"epoch": epoch, "batch": batch, "loss": loss})
            
            print(f"Epoch {epoch} Loss: {epoch_loss / num_batches:.4f}")
            
        metrics_file = path / f"{self.config.method.lower()}_train_history.json"
        metrics_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Mock training complete. History written to {metrics_file}")
