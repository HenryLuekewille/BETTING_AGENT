"""
Betting Agent Core Module
"""

import sys
from pathlib import Path

# ✅ Stelle sicher, dass core im Python Path ist
core_dir = Path(__file__).parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

__version__ = "1.0.0"

# Imports
from core.betting_env import BettingEnvOptimized
from core.evaluation_tracker import EvaluationTracker
from core.incremental_learner import IncrementalLearner
from core.adaptive_training import AdaptiveTrainingSystem

__all__ = [
    "BettingEnvOptimized",
    "EvaluationTracker",
    "IncrementalLearner",
    "AdaptiveTrainingSystem"
]