from __future__ import annotations

import typing
import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(sigmoid(x)) ổn định số cho mọi giá trị x."""
    return typing.cast(np.ndarray, -np.logaddexp(0.0, -x))

def _log_odds(logp: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)) tính từ log(p)."""
    logp_safe = np.clip(logp, -30.0, -1e-7)
    return typing.cast(np.ndarray, logp_safe - np.log1p(-np.exp(logp_safe)))

def dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray, ref_chosen_logps: np.ndarray, ref_rejected_logps: np.ndarray, beta: float) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    TODO(student): implement numerically stable DPO loss.
    Hint: compare policy log-ratio against reference log-ratio, then use log-sigmoid.
    """
    policy_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps
    margin = beta * (policy_diff - ref_diff)
    losses = -_log_sigmoid(margin)
    return float(np.mean(losses))

def orpo_loss(sft_nll: np.ndarray, chosen_logps: np.ndarray, rejected_logps: np.ndarray, lambda_orpo: float) -> float:
    """Compute a simplified ORPO-style objective.

    TODO(student): implement SFT loss + odds-ratio preference penalty.
    """
    chosen_log_odds = _log_odds(chosen_logps)
    rejected_log_odds = _log_odds(rejected_logps)
    log_odds_diff = chosen_log_odds - rejected_log_odds
    
    loss = np.mean(sft_nll) - lambda_orpo * np.mean(_log_sigmoid(log_odds_diff))
    return float(loss)
