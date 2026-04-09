"""
RecBole-SAE: Sparse Autoencoder interpretation layer for RecBole recommendation models.

Follows the same extension pattern as RecBole-GNN:
  - RecBole handles all base models, datasets, dataloaders, configs
  - This package adds: SAE model, probing, LLM interpretation, evaluation

Usage:
    from recbole_sae import probe_model, SAETrainer, LLMInterpreter, full_report
"""

from recbole_sae.probe      import probe_model
from recbole_sae.trainer    import SAETrainer
from recbole_sae.interpret  import LLMInterpreter, compute_activation_levels, predict_top1, full_report

__version__ = "0.1.0"
__all__ = [
    "probe_model",
    "SAETrainer",
    "LLMInterpreter",
    "compute_activation_levels",
    "predict_top1",
    "full_report",
]
