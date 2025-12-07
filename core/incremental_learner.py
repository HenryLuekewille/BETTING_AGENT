"""
Inkrementelles Lernen mit variabler Bet Size.
✅ FIX: Kelly Criterion + Confidence-Based Scaling implementiert
"""

import sys
from pathlib import Path
import logging

BASE_DIR = Path(__file__).parent.parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from core.betting_env import BettingEnvOptimized


"""
Inkrementelles Lernen mit variabler Bet Size.
✅ NEU: Model Ensemble Support
"""

# ... (Imports wie vorher) ...

class IncrementalLearner:
    """Verwaltet inkrementelle Updates + Model Ensemble."""
    
    def __init__(self, model, global_model, ensemble_config, config, tracker, file_logger=None):
        """
        ✅ NEU: global_model + ensemble_config Parameter
        """
        self.model = model  # Fine-tuned model
        self.global_model = global_model  # ✅ NEU: Global model
        self.ensemble_config = ensemble_config  # ✅ NEU
        self.config = config
        self.tracker = tracker
        self.file_logger = file_logger or logging.getLogger('training')
        
        self.incremental_timesteps = config["training"]["incremental_timesteps"]
        self.incremental_lr = config["model"]["dqn"]["learning_rate_incremental"]
        
        self.model.learning_rate = self.incremental_lr
        
        # ✅ NEU: Ensemble Weights
        self.use_ensemble = ensemble_config.get("enabled", False) and global_model is not None
        self.global_weight = ensemble_config.get("global_weight", 0.5)
        self.finetune_weight = ensemble_config.get("finetune_weight", 0.5)
        
        self.file_logger.info(f"🔧 Incremental Learner initialisiert")
        self.file_logger.info(f"   Timesteps pro Update: {self.incremental_timesteps:,}")
        self.file_logger.info(f"   Learning Rate: {self.incremental_lr}")
        
        if self.use_ensemble:
            self.file_logger.info(f"   ⚖️  Ensemble Mode: Global={self.global_weight}, Finetune={self.finetune_weight}")
        
        self.file_logger.info("")
    
    def predict_gameday(self, gameday_data):
        """
        Vorhersagen für gesamten Spieltag.
        ✅ NEU: Mit Ensemble Support
        """
        predictions = []
        
        temp_env = BettingEnvOptimized(
            gameday_data,
            use_betting_odds=self.config["environment"]["use_betting_odds"],
            confidence_threshold=self.config["environment"]["confidence_threshold"],
            bet_amount=self.config["environment"]["bet_amount"],
            max_bet_amount=self.config["environment"]["max_bet_amount"],
            min_gameday=1,
            apply_gameday_filter=False,
            reward_shaping=self.config["environment"]["reward_shaping"],
            no_bet_reward_multiplier=self.config["environment"].get("no_bet_reward_multiplier", 0.5),
            draw_penalty_multiplier=self.config["environment"].get("draw_penalty_multiplier", 1.5),
            min_edge_required=self.config["environment"].get("min_edge_required", 0.05),
            max_bet_rate=self.config["environment"].get("max_bet_rate", 0.30),
            use_kelly_criterion=self.config["environment"].get("use_kelly_criterion", True),
            kelly_fraction=self.config["environment"].get("kelly_fraction", 0.25),
        )
        
        for idx in range(len(gameday_data)):
            temp_env.current_step = idx
            obs = temp_env._get_obs()
            
            # ✅ NEU: Ensemble Prediction
            if self.use_ensemble:
                action = self._ensemble_predict(obs)
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            
            row = gameday_data.iloc[idx]
            
            predictions.append({
                'match_idx': idx,
                'home_team': row['HomeTeam'],
                'away_team': row['AwayTeam'],
                'action': int(action),
                'actual_ftr': row['FTR'],
                'actual_fthg': int(row['FTHG']),
                'actual_ftag': int(row['FTAG']),
            })
        
        return predictions
    
    def _ensemble_predict(self, obs):
        """
        ✅ NEU: Kombiniert Q-Values von Global + Finetune Model
        
        Args:
            obs: Observation array
            
        Returns:
            int: Gewählte Action
        """
        import torch
        
        # Get Q-Values from both models
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        
        q_finetune = self.model.policy.q_net(obs_tensor)[0].detach().cpu().numpy()
        q_global = self.global_model.policy.q_net(obs_tensor)[0].detach().cpu().numpy()
        
        # ✅ Weighted Ensemble
        q_combined = (
            self.global_weight * q_global + 
            self.finetune_weight * q_finetune
        )
        
        # Select best action
        action = int(np.argmax(q_combined))
        
        return action
    
    # ... (Rest der Methoden wie vorher) ...
    
    def evaluate_predictions(self, predictions, gameday_data):
        """✅ Evaluiere Vorhersagen mit VARIABLER BET SIZE."""
        action_names = ["No Bet", "Home", "Away", "Over", "Under"]
        
        total_bets = 0
        total_wins = 0
        total_profit = 0.0
        total_invested = 0.0
        
        # ✅ Config-Parameter laden
        base_bet = self.config["environment"]["bet_amount"]
        max_bet = self.config["environment"]["max_bet_amount"]
        use_kelly = self.config["environment"].get("use_kelly_criterion", True)
        kelly_fraction = self.config["environment"].get("kelly_fraction", 0.25)
        
        for pred in predictions:
            action = pred['action']
            
            if action == 0:
                continue
            
            row = gameday_data.iloc[pred['match_idx']]
            
            # ✅ HOLE QUOTE UND PROBABILITIES
            if action == 1:
                quote = self._safe_float(row.get('MaxQuote_H', np.nan))
                prob = self._safe_float(row.get('ImpProb_H', 0.33))
            elif action == 2:
                quote = self._safe_float(row.get('MaxQuote_A', np.nan))
                prob = self._safe_float(row.get('ImpProb_A', 0.33))
            elif action == 3:
                quote = self._safe_float(row.get('OU_Over', np.nan))
                expected_goals = self._safe_float(row.get('Expected_Goals', 2.5))
                prob = min(0.8, expected_goals / 5.0)
            elif action == 4:
                quote = self._safe_float(row.get('OU_Under', np.nan))
                expected_goals = self._safe_float(row.get('Expected_Goals', 2.5))
                prob = max(0.2, 1 - (expected_goals / 5.0))
            else:
                continue
            
            if np.isnan(quote) or quote < 1.01:
                continue
            
            # ✅ BERECHNE VARIABLE BET SIZE
            bet_size = self._calculate_bet_size(
                action, quote, prob, row,
                base_bet, max_bet, use_kelly, kelly_fraction
            )
            
            total_bets += 1
            total_invested += bet_size
            
            # Evaluiere Wette
            ftr = pred['actual_ftr']
            total_goals = pred['actual_fthg'] + pred['actual_ftag']
            
            if action == 1:
                won = (ftr == 'H')
            elif action == 2:
                won = (ftr == 'A')
            elif action == 3:
                won = (total_goals > 2.5)
            elif action == 4:
                won = (total_goals <= 2.5)
            
            if won:
                profit = (quote - 1.0) * bet_size
                total_wins += 1
            else:
                profit = -bet_size
            
            total_profit += profit
            
            # ✅ ENHANCED LOGGING
            self.file_logger.info(
                f"   {pred['home_team']:20} - {pred['away_team']:20} | "
                f"Action: {action_names[action]:7} | "
                f"Bet: {bet_size:>6.2f}€ | "
                f"Quote: {quote:>5.2f} | "
                f"Prob: {prob*100:>5.1f}% | "
                f"Result: {ftr:1} | "
                f"{'✅ WON' if won else '❌ LOST':8} | "
                f"Profit: {profit:>8.2f}€"
            )
        
        roi = total_profit / max(1, total_invested)
        winrate = total_wins / max(1, total_bets)
        
        return {
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_profit': total_profit,
            'total_invested': total_invested,
            'roi': roi,
            'winrate': winrate,
        }
    
    def _calculate_bet_size(self, action, quote, prob, row, base_bet, max_bet, use_kelly, kelly_fraction):
        """✅ Berechnet variable Bet Size basierend auf Kelly + Confidence."""
        
        # KELLY CRITERION (wenn aktiviert und profitable)
        if use_kelly and prob > 0.5 and quote > 1.0:
            b = quote - 1  # Net odds
            q = 1 - prob
            
            # Kelly Formula: f* = (bp - q) / b
            kelly_value = (b * prob - q) / b
            kelly_value = max(0, kelly_value)  # No negative bets
            
            # Fractional Kelly (safer)
            kelly_value *= kelly_fraction
            
            # Bet Size = Base + Kelly Bonus (bis zu 3x Base)
            bet_size = base_bet * (1 + kelly_value * 3)
            
            # Cap bei Maximum
            bet_size = min(bet_size, max_bet)
        
        else:
            # CONFIDENCE-BASED SCALING (Fallback)
            if prob > 0.75:
                multiplier = 2.5
            elif prob > 0.70:
                multiplier = 2.0
            elif prob > 0.65:
                multiplier = 1.5
            elif prob > 0.60:
                multiplier = 1.2
            else:
                multiplier = 1.0
            
            bet_size = base_bet * multiplier
            bet_size = min(bet_size, max_bet)
        
        # ✅ VALUE BONUS (basierend auf Edge)
        if action in [1, 2]:
            value = self._safe_float(row.get(f'Value_{"Home" if action == 1 else "Away"}', 0))
            if value > 0.08:
                bet_size = min(bet_size * 1.2, max_bet)
            elif value > 0.05:
                bet_size = min(bet_size * 1.1, max_bet)
        
        return round(bet_size, 2)
    
    def evaluate_and_track(self, predictions, gameday_data, gameday, tracker):
        """Evaluiere UND logge."""
        results = self.evaluate_predictions(predictions, gameday_data)
        
        tracker.log_gameday(
            gameday=gameday,
            results=results,
            predictions=predictions,
            gameday_data=gameday_data
        )
        
        return results
    
    def incremental_update(self, cumulative_data):
        """Update Modell."""
        self.file_logger.info(f"\n🔄 Inkrementelles Update...")
        self.file_logger.info(f"   Training auf {len(cumulative_data)} Spiele")
        
        update_env = BettingEnvOptimized(
            cumulative_data,
            use_betting_odds=self.config["environment"]["use_betting_odds"],
            confidence_threshold=self.config["environment"]["confidence_threshold"],
            bet_amount=self.config["environment"]["bet_amount"],
            max_bet_amount=self.config["environment"]["max_bet_amount"],
            min_gameday=1,
            apply_gameday_filter=False,
            reward_shaping=self.config["environment"]["reward_shaping"],
            no_bet_reward_multiplier=self.config["environment"].get("no_bet_reward_multiplier", 0.5),
            draw_penalty_multiplier=self.config["environment"].get("draw_penalty_multiplier", 1.5),
            min_edge_required=self.config["environment"].get("min_edge_required", 0.05),
            max_bet_rate=self.config["environment"].get("max_bet_rate", 0.30),
            use_kelly_criterion=self.config["environment"].get("use_kelly_criterion", True),
            kelly_fraction=self.config["environment"].get("kelly_fraction", 0.25),
        )
        
        self.model.set_env(update_env)
        
        self.model.learn(
            total_timesteps=self.incremental_timesteps,
            reset_num_timesteps=False,
            progress_bar=False
        )
        
        self.file_logger.info(f"   ✅ Update abgeschlossen\n")
    
    @staticmethod
    def _safe_float(x):
        try:
            val = float(x)
            return val if not (np.isnan(val) or np.isinf(val)) else np.nan
        except (TypeError, ValueError):
            return np.nan