"""
Multi-Agent Reinforcement Learning System
✅ Mehrere spezialisierte Agenten mit Ensemble-Entscheidung
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

BASE_DIR = Path(__file__).parent.parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.betting_env import BettingEnvOptimized
from core.evaluation_tracker import EvaluationTracker


class AgentProfile:
    """Definiert ein Agent-Profil mit spezifischen Parametern."""
    
    def __init__(
        self,
        name: str,
        confidence_threshold: float,
        min_edge: float,
        max_bet_rate: float,
        bet_amount: float,
        max_bet_amount: float,
        kelly_fraction: float = 0.25,
        reward_shaping: str = "balanced"
    ):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.min_edge = min_edge
        self.max_bet_rate = max_bet_rate
        self.bet_amount = bet_amount
        self.max_bet_amount = max_bet_amount
        self.kelly_fraction = kelly_fraction
        self.reward_shaping = reward_shaping


# Vordefinierte Profile
AGENT_PROFILES = {
    'conservative': AgentProfile(
        name='Conservative',
        confidence_threshold=0.70,
        min_edge=0.06,
        max_bet_rate=0.20,
        bet_amount=10.0,
        max_bet_amount=20.0,
        kelly_fraction=0.15,
        reward_shaping='conservative'
    ),
    'balanced': AgentProfile(
        name='Balanced',
        confidence_threshold=0.60,
        min_edge=0.04,
        max_bet_rate=0.30,
        bet_amount=10.0,
        max_bet_amount=30.0,
        kelly_fraction=0.25,
        reward_shaping='balanced'
    ),
    'aggressive': AgentProfile(
        name='Aggressive',
        confidence_threshold=0.55,
        min_edge=0.02,
        max_bet_rate=0.45,
        bet_amount=10.0,
        max_bet_amount=50.0,
        kelly_fraction=0.35,
        reward_shaping='aggressive'
    )
}


class MultiAgentSystem:
    """
    Multi-Agent System für Sportwetten.
    
    Trainiert mehrere spezialisierte Agenten und kombiniert ihre Vorhersagen.
    """
    
    def __init__(self, config: dict, file_logger=None):
        self.config = config
        self.file_logger = file_logger or logging.getLogger('multi_agent')
        
        # Agent Profiles
        self.profiles = self._load_profiles()
        
        # Storage
        self.agents = {}
        self.trackers = {}
        
        self.file_logger.info(f"\n{'='*80}")
        self.file_logger.info(f"🤖 MULTI-AGENT SYSTEM INITIALISIERT")
        self.file_logger.info(f"{'='*80}")
        self.file_logger.info(f"Anzahl Agenten: {len(self.profiles)}\n")
        
        for profile in self.profiles:
            self.file_logger.info(f"  📊 Agent: {profile.name}")
            self.file_logger.info(f"     Confidence: {profile.confidence_threshold}")
            self.file_logger.info(f"     Min Edge: {profile.min_edge}")
            self.file_logger.info(f"     Max Bet Rate: {profile.max_bet_rate*100:.0f}%\n")
    
    def _load_profiles(self):
        """Lädt Agent-Profile aus Config."""
        profiles = []
        
        # Check ob Multi-Agent aktiv
        if not self.config.get('multi_agent', {}).get('enabled', False):
            # Fallback: Single Agent
            profiles.append(AGENT_PROFILES['balanced'])
        else:
            # Multi-Agent: Lade Profile
            agent_types = self.config['multi_agent'].get('agent_types', ['conservative', 'balanced', 'aggressive'])
            
            for agent_type in agent_types:
                if agent_type in AGENT_PROFILES:
                    profiles.append(AGENT_PROFILES[agent_type])
        
        return profiles
    
    def train_all_agents(self, data: pd.DataFrame, phase: str = "global"):
        """Trainiert alle Agenten parallel."""
        self.file_logger.info(f"\n{'='*80}")
        self.file_logger.info(f"🎓 TRAINING PHASE: {phase.upper()}")
        self.file_logger.info(f"{'='*80}\n")
        
        for profile in self.profiles:
            self.file_logger.info(f"\n🤖 Trainiere Agent: {profile.name}")
            self.file_logger.info(f"   {'-'*60}\n")
            
            # Create Environment
            env = self._create_env(data, profile)
            vec_env = DummyVecEnv([lambda: env])
            
            # Create Model
            model = self._create_model(vec_env, profile, phase)
            
            # Train
            timesteps = self._get_timesteps(phase)
            model.learn(total_timesteps=timesteps, progress_bar=False)
            
            # Store
            self.agents[profile.name] = model
            
            # Save
            model_path = Path(self.config['paths']['models_dir']) / f"{profile.name.lower()}_{phase}.zip"
            model.save(str(model_path))
            
            self.file_logger.info(f"\n   ✅ Agent {profile.name} gespeichert: {model_path}\n")
    
    def _create_env(self, data: pd.DataFrame, profile: AgentProfile):
        """Erstellt Environment für Agent-Profil."""
        return BettingEnvOptimized(
            data,
            use_betting_odds=self.config['environment']['use_betting_odds'],
            confidence_threshold=profile.confidence_threshold,
            bet_amount=profile.bet_amount,
            max_bet_amount=profile.max_bet_amount,
            min_gameday=self.config['environment'].get('min_gameday', 4),
            apply_gameday_filter=False,
            reward_shaping=profile.reward_shaping,
            min_edge_required=profile.min_edge,
            max_bet_rate=profile.max_bet_rate,
            use_kelly_criterion=True,
            kelly_fraction=profile.kelly_fraction,
        )
    
    def _create_model(self, vec_env, profile: AgentProfile, phase: str):
        """Erstellt DQN Model für Agent."""
        dqn_cfg = self.config['model']['dqn']
        
        # Adjust LR based on phase
        if phase == "global":
            lr = dqn_cfg['learning_rate']
        elif phase == "finetune":
            lr = dqn_cfg['learning_rate_finetune']
        else:
            lr = dqn_cfg['learning_rate_incremental']
        
        return DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            buffer_size=dqn_cfg['buffer_size'],
            learning_starts=dqn_cfg['learning_starts'],
            batch_size=dqn_cfg['batch_size'],
            gamma=dqn_cfg['gamma'],
            tau=dqn_cfg['tau'],
            exploration_fraction=dqn_cfg['exploration_fraction'],
            exploration_initial_eps=dqn_cfg['exploration_initial_eps'],
            exploration_final_eps=dqn_cfg['exploration_final_eps'],
            target_update_interval=dqn_cfg['target_update_interval'],
            policy_kwargs=self.config['model']['policy_kwargs'],
            verbose=0,
            device="auto",
        )
    
    def _get_timesteps(self, phase: str):
        """Gibt Timesteps für Phase zurück."""
        if phase == "global":
            return self.config['training']['global_timesteps']
        elif phase == "finetune":
            return self.config['training']['finetune_timesteps']
        else:
            return self.config['training']['incremental_timesteps']
    
    def predict_ensemble(self, gameday_data: pd.DataFrame, strategy: str = "voting"):
        """
        Ensemble-Vorhersage über alle Agenten.
        
        Args:
            gameday_data: DataFrame mit Spielen
            strategy: 'voting', 'weighted', 'max_confidence'
        """
        predictions = []
        
        for idx in range(len(gameday_data)):
            row = gameday_data.iloc[idx]
            
            # Sammle Vorhersagen aller Agenten
            agent_predictions = []
            
            for profile in self.profiles:
                agent = self.agents[profile.name]
                env = self._create_env(gameday_data, profile)
                env.current_step = idx
                obs = env._get_obs()
                
                action, _ = agent.predict(obs, deterministic=True)
                
                # Hole Confidence (Q-Value)
                q_values = agent.policy.q_net(agent.policy.obs_to_tensor(obs)[0])
                confidence = q_values[0][action].item()
                
                agent_predictions.append({
                    'agent': profile.name,
                    'action': int(action),
                    'confidence': confidence
                })
            
            # Kombiniere Vorhersagen
            final_action = self._combine_predictions(agent_predictions, strategy)
            
            predictions.append({
                'match_idx': idx,
                'home_team': row['HomeTeam'],
                'away_team': row['AwayTeam'],
                'action': final_action,
                'actual_ftr': row['FTR'],
                'actual_fthg': int(row['FTHG']),
                'actual_ftag': int(row['FTAG']),
                'agent_votes': agent_predictions
            })
        
        return predictions
    
    def _combine_predictions(self, agent_predictions: list, strategy: str):
        """Kombiniert Agent-Vorhersagen zu finaler Entscheidung."""
        
        if strategy == "voting":
            # Mehrheitsentscheidung
            actions = [p['action'] for p in agent_predictions]
            return max(set(actions), key=actions.count)
        
        elif strategy == "weighted":
            # Gewichtet nach Confidence
            weighted_actions = {}
            for p in agent_predictions:
                action = p['action']
                conf = p['confidence']
                weighted_actions[action] = weighted_actions.get(action, 0) + conf
            
            return max(weighted_actions, key=weighted_actions.get)
        
        elif strategy == "max_confidence":
            # Agent mit höchster Confidence
            best = max(agent_predictions, key=lambda p: p['confidence'])
            return best['action']
        
        else:
            # Fallback: Voting
            actions = [p['action'] for p in agent_predictions]
            return max(set(actions), key=actions.count)
    
    def evaluate_gameday(self, predictions: list, gameday_data: pd.DataFrame, gameday: int):
        """Evaluiert Vorhersagen für einen Spieltag."""
        action_names = ["No Bet", "Home", "Away", "Over", "Under"]
        
        total_bets = 0
        total_wins = 0
        total_profit = 0.0
        total_invested = 0.0
        
        for pred in predictions:
            action = pred['action']
            
            if action == 0:
                continue
            
            row = gameday_data.iloc[pred['match_idx']]
            
            # Quote und Bet Size
            if action == 1:
                quote = self._safe_float(row.get('MaxQuote_H', np.nan))
            elif action == 2:
                quote = self._safe_float(row.get('MaxQuote_A', np.nan))
            elif action == 3:
                quote = self._safe_float(row.get('OU_Over', np.nan))
            elif action == 4:
                quote = self._safe_float(row.get('OU_Under', np.nan))
            else:
                continue
            
            if np.isnan(quote) or quote < 1.01:
                continue
            
            # Bet Size (simplified)
            bet_size = 10.0  # Could be calculated from profiles
            
            total_bets += 1
            total_invested += bet_size
            
            # Evaluate
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
            
            # Log
            self.file_logger.info(
                f"   {pred['home_team']:20} - {pred['away_team']:20} | "
                f"Action: {action_names[action]:7} | "
                f"Bet: {bet_size:>6.2f}€ | "
                f"Quote: {quote:>5.2f} | "
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
    
    @staticmethod
    def _safe_float(x):
        try:
            val = float(x)
            return val if not (np.isnan(val) or np.isinf(val)) else np.nan
        except (TypeError, ValueError):
            return np.nan