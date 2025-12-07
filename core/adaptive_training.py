"""
Adaptive RL Training System
✅ Mit korrektem Logging in Datei
✅ Model Ensemble Support
✅ FIX: Robuste Handhabung von None-Werten
✅ FIX: WalkForwardCallback korrekt implementiert
"""

import os
import sys
from pathlib import Path
import logging

# ✅ FIX: Projekt-Root zum Python Path hinzufügen
SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_DIR = SCRIPT_DIR.parent.absolute()

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from core.betting_env import BettingEnvOptimized
from core.incremental_learner import IncrementalLearner
from core.evaluation_tracker import EvaluationTracker


# ✅ Logger Setup
def setup_logger(log_file):
    """
    Richtet Logging für Konsole + Datei ein.
    ✅ FIX: Verhindert doppelte Ausgaben
    """
    
    # Get or create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # ✅ Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # ✅ Prevent propagation to root logger
    logger.propagate = False
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File Handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class WalkForwardCallback(BaseCallback):
    """✅ FIX: Einfacher Callback ohne Config-Loading."""
    
    def __init__(self, check_freq=10000, file_logger=None):
        super().__init__()
        self.check_freq = check_freq
        self.file_logger = file_logger or logging.getLogger('training')
        self.step_counter = 0
        self.episode_rewards = []
        self.current_rewards = []
        self.last_check = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        self.current_rewards.append(reward)
        self.step_counter += 1
        
        if done:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []

        if self.n_calls % self.check_freq == 0 and self.n_calls != self.last_check:
            self.last_check = self.n_calls
            env = self.training_env.envs[0]
            while hasattr(env, "env"):
                env = env.env
            
            roi = env.total_profit / max(1, env.total_invested)
            winrate = env.success_bets / max(1, env.total_bets)
            bet_rate = env.total_bets / max(1, env.current_step)
            no_bet_acc = env.no_bet_correct / max(1, env.no_bet_correct + env.no_bet_wrong)
            avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
            
            # ✅ WARNUNG wenn zu wenig Wetten
            if bet_rate < 0.05:
                self.file_logger.warning(
                    f"⚠️  BET RATE ZU NIEDRIG: {bet_rate*100:.1f}% "
                    f"(Target: 20-40%)"
                )
                self.file_logger.info(
                    "   → Erwäge: Threshold senken, Edge-Requirement lockern\n"
                )
            
            # Logging...
            msg = f"\n{'='*80}\n"
            msg += f"  📊 TRAINING PROGRESS - Step {self.n_calls:,}\n"
            msg += f"{'='*80}\n"
            msg += f"  Episodes:           {len(self.episode_rewards):>10,}\n"
            msg += f"  ROI:                {roi*100:>10.2f}%\n"
            msg += f"  Winrate:            {winrate*100:>10.2f}%\n"
            msg += f"  Bet Rate:           {bet_rate*100:>10.2f}%"
            
            if bet_rate < 0.10:
                msg += " ⚠️  TOO LOW!\n"
            else:
                msg += "\n"
            
            msg += f"  No-Bet Accuracy:    {no_bet_acc*100:>10.2f}%\n"
            msg += f"  Balance:            {env.balance:>10.2f}€\n"
            msg += f"  Total Bets:         {env.total_bets:>10,}\n"
            msg += f"  Wins:               {env.success_bets:>10,}\n"
            msg += f"  Invested:           {env.total_invested:>10.2f}€\n"
            msg += f"  Draw Avoided:       {env.draw_avoided:>10,}\n"
            msg += f"  Avg Reward (50ep):  {avg_reward:>10.3f}\n"
            msg += f"{'='*80}\n"
            
            self.file_logger.info(msg)
        
        return True


class AdaptiveTrainingSystem:
    """Verwaltet gesamtes Training mit Logging."""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialisiere System mit Config."""
        config_path = Path(config_path).absolute()
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Config nicht gefunden: {config_path}")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self._setup_paths()
        
        # ✅ Setup Logger
        log_file = Path(self.config["paths"]["logs_dir"]) / "training.log"
        self.file_logger = setup_logger(log_file)
        
        self.file_logger.info(f"🔧 Base Dir: {BASE_DIR}")
        self.file_logger.info(f"🔧 Config: {config_path}\n")
        
        self._load_data()
        self.tracker = EvaluationTracker(self.config["paths"]["results_dir"])
        
        # Store global model for ensemble
        self.global_model = None
    
    def _setup_paths(self):
        """Erstelle alle benötigten Verzeichnisse (silent)."""
        for key, path in self.config["paths"].items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def _load_data(self):
        """Lade Daten und erstelle Splits."""
        if "data" in self.config:
            data_dir = Path(self.config["data"]["data_dir"])
            pattern = self.config["data"].get("pattern", "Bundesliga_*.csv")
        else:
            data_dir = Path(self.config["paths"]["data_dir"])
            pattern = "Bundesliga_*.csv"
        
        data_dir = data_dir.absolute()
        csv_files = sorted(data_dir.glob(pattern))
        
        if not csv_files:
            raise FileNotFoundError(
                f"❌ Keine CSV in {data_dir} mit Pattern '{pattern}'"
            )
        
        latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
        self.file_logger.info(f"📂 Lade Daten: {latest_csv.name}")
        
        df = pd.read_csv(latest_csv, sep=";")
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
        
        cfg = self.config["training"]
        
        global_start = cfg["global_seasons"]["start"]
        global_end = cfg["global_seasons"]["end"]
        self.global_data = df[
            (df["Season"] >= global_start) & (df["Season"] <= global_end)
        ].reset_index(drop=True)
        
        self.target_season = cfg["target_season"]
        self.target_data = df[df["Season"] == self.target_season].reset_index(drop=True)
        
        # ✅ FIX: Robuste Handhabung von fine_tune_gamedays
        ft_gamedays = cfg.get("fine_tune_gamedays")
        
        # Handle None, 0, or missing values
        if ft_gamedays is None or ft_gamedays == 0:
            ft_gamedays = 8  # Default value
            self.file_logger.warning(
                f"⚠️  fine_tune_gamedays nicht gesetzt oder 0, nutze default: {ft_gamedays}"
            )
        
        # Validate range
        if not (1 <= ft_gamedays <= 33):
            self.file_logger.warning(
                f"⚠️  fine_tune_gamedays={ft_gamedays} außerhalb Bereich 1-33, nutze 8"
            )
            ft_gamedays = 8
        
        self.finetune_data = self.target_data[
            self.target_data["Gameday"] <= ft_gamedays
        ].reset_index(drop=True)
        
        self.deploy_data = self.target_data[
            self.target_data["Gameday"] > ft_gamedays
        ].reset_index(drop=True)
        
        # Validate splits
        if len(self.finetune_data) == 0:
            raise ValueError(
                f"❌ Keine Daten für Fine-Tuning (Gameday <= {ft_gamedays})!"
            )
        
        if len(self.deploy_data) == 0:
            raise ValueError(
                f"❌ Keine Daten für Deployment (Gameday > {ft_gamedays})!"
            )
        
        msg = f"\n📊 Daten-Split:\n"
        msg += f"   Global Training:  {len(self.global_data):>6,} Spiele ({global_start}-{global_end})\n"
        msg += f"   Fine-Tuning:      {len(self.finetune_data):>6,} Spiele (Spieltag 1-{ft_gamedays})\n"
        msg += f"   Deployment:       {len(self.deploy_data):>6,} Spiele (Spieltag {ft_gamedays+1}-34)\n"
        
        self.file_logger.info(msg)
    
    def _create_env(self, data, name="env"):
        """Erstelle Betting Environment."""
        env_cfg = self.config["environment"]
        
        return BettingEnvOptimized(
            data,
            use_betting_odds=env_cfg.get("use_betting_odds", True),
            confidence_threshold=env_cfg.get("confidence_threshold", 0.60),
            bet_amount=env_cfg.get("bet_amount", 10.0),
            max_bet_amount=env_cfg.get("max_bet_amount", 30.0),
            min_gameday=env_cfg.get("min_gameday", 4),
            apply_gameday_filter=False,
            reward_shaping=env_cfg.get("reward_shaping", "conservative"),
            no_bet_reward_multiplier=env_cfg.get("no_bet_reward_multiplier", 0.5),
            draw_penalty_multiplier=env_cfg.get("draw_penalty_multiplier", 1.5),
            min_edge_required=env_cfg.get("min_edge_required", 0.05),
            max_bet_rate=env_cfg.get("max_bet_rate", 0.30),
            use_kelly_criterion=env_cfg.get("use_kelly_criterion", True),
            kelly_fraction=env_cfg.get("kelly_fraction", 0.25),
            confidence_scaling_mode=env_cfg.get("confidence_scaling_mode", "none"),
        )
    
    def phase1_global_training(self):
        """Phase 1: Global Pretraining."""
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info("🌍 PHASE 1: GLOBAL PRETRAINING")
        self.file_logger.info("="*80 + "\n")
        
        env = self._create_env(self.global_data, "global_train")
        vec_env = DummyVecEnv([lambda: env])
        
        dqn_cfg = self.config["model"]["dqn"]
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=dqn_cfg.get("learning_rate", 0.0001),
            buffer_size=dqn_cfg.get("buffer_size", 100000),
            learning_starts=dqn_cfg.get("learning_starts", 5000),
            batch_size=dqn_cfg.get("batch_size", 64),
            gamma=dqn_cfg.get("gamma", 0.95),
            tau=dqn_cfg.get("tau", 1.0),
            exploration_fraction=dqn_cfg.get("exploration_fraction", 0.3),
            exploration_initial_eps=dqn_cfg.get("exploration_initial_eps", 1.0),
            exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.02),
            target_update_interval=dqn_cfg.get("target_update_interval", 500),
            policy_kwargs=self.config["model"].get("policy_kwargs", {"net_arch": [256, 256, 128]}),
            verbose=0,
            device="auto",
        )
        
        checkpoint_cb = CheckpointCallback(
            save_freq=25000,
            save_path=self.config["paths"]["checkpoints_dir"],
            name_prefix="global"
        )
        
        walkforward_cb = WalkForwardCallback(
            check_freq=10000, 
            file_logger=self.file_logger
        )
        
        timesteps = self.config["training"].get("global_timesteps", 1000000)
        self.file_logger.info(f"🚀 Starte Global Training ({timesteps:,} steps)...\n")
        
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, walkforward_cb],
            progress_bar=False
        )
        
        model_path = Path(self.config["paths"]["models_dir"]) / f"global_model_{self.target_season}.zip"
        model.save(str(model_path))
        self.file_logger.info(f"\n💾 Global Model gespeichert: {model_path}\n")
        
        # ✅ Store global model for ensemble
        self.global_model = model
        
        return model
    
    def phase2_finetuning(self, base_model):
        """Phase 2: Fine-Tuning."""
        ft_gamedays = self.config["training"].get("fine_tune_gamedays", 8)
        
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info(f"🎯 PHASE 2: FINE-TUNING (Spieltag 1-{ft_gamedays})")
        self.file_logger.info("="*80 + "\n")
        
        env = self._create_env(self.finetune_data, "finetune")
        vec_env = DummyVecEnv([lambda: env])
        
        base_model.set_env(vec_env)
        base_model.learning_rate = self.config["model"]["dqn"].get("learning_rate_finetune", 0.00001)
        
        timesteps = self.config["training"].get("finetune_timesteps", 100000)
        
        self.file_logger.info(f"🔧 Fine-Tuning ({timesteps:,} steps)...")
        self.file_logger.info(f"   Learning Rate: {base_model.learning_rate}\n")
        
        base_model.learn(
            total_timesteps=timesteps,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        model_path = Path(self.config["paths"]["models_dir"]) / f"finetuned_model_{self.target_season}.zip"
        base_model.save(str(model_path))
        self.file_logger.info(f"\n💾 Fine-Tuned Model gespeichert: {model_path}\n")
        
        return base_model
    
    def phase3_incremental_deployment(self, model):
        """Phase 3: Deployment mit inkrementellem Lernen."""
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info("🚀 PHASE 3: INCREMENTAL DEPLOYMENT")
        self.file_logger.info("="*80 + "\n")
        
        if not self.config["training"].get("incremental_learning", True):
            self.file_logger.info("⚠️  Inkrementelles Lernen deaktiviert\n")
            return self._deployment_without_learning(model)
        
        # ✅ Check if model ensemble is configured
        ensemble_config = self.config["training"].get("model_ensemble", {})
        
        # ✅ If ensemble enabled and we have global_model
        global_model = None
        if ensemble_config.get("enabled", False):
            if self.global_model is not None:
                global_model = self.global_model
                self.file_logger.info(f"✅ Global Model für Ensemble geladen (aus Phase 1)\n")
            else:
                # Try to load from disk
                model_path = Path(self.config["paths"]["models_dir"]) / f"global_model_{self.target_season}.zip"
                if model_path.exists():
                    global_model = DQN.load(str(model_path))
                    self.file_logger.info(f"✅ Global Model für Ensemble geladen: {model_path}\n")
                else:
                    self.file_logger.warning(f"⚠️  Global Model nicht gefunden: {model_path}")
                    self.file_logger.warning(f"   Ensemble wird deaktiviert.\n")
                    ensemble_config["enabled"] = False
        
        # ✅ Create IncrementalLearner with correct parameters
        learner = IncrementalLearner(
            model=model,
            global_model=global_model,
            ensemble_config=ensemble_config,
            config=self.config,
            tracker=self.tracker,
            file_logger=self.file_logger
        )
        
        deploy_sorted = self.deploy_data.sort_values("Gameday").reset_index(drop=True)
        gamedays = sorted(deploy_sorted["Gameday"].unique())
        
        self.file_logger.info(f"📅 Spieltage für Deployment: {len(gamedays)}")
        self.file_logger.info(f"   {gamedays}\n")
        
        cumulative_data = self.finetune_data.copy()
        
        for gd in gamedays:
            self.file_logger.info(f"\n{'='*80}")
            self.file_logger.info(f"📍 SPIELTAG {gd}")
            self.file_logger.info(f"{'='*80}\n")
            
            gameday_data = deploy_sorted[deploy_sorted["Gameday"] == gd].reset_index(drop=True)
            
            predictions = learner.predict_gameday(gameday_data)
            results = learner.evaluate_and_track(predictions, gameday_data, gd, self.tracker)
            
            cumulative_data = pd.concat([cumulative_data, gameday_data], ignore_index=True)
            learner.incremental_update(cumulative_data)
            
            self.file_logger.info(f"\n✅ Spieltag {gd} abgeschlossen")
            self.file_logger.info(f"   ROI: {results['roi']*100:.2f}%")
            self.file_logger.info(f"   Winrate: {results['winrate']*100:.2f}%")
            self.file_logger.info(f"   Kumulative ROI: {self.tracker.cumulative_profit / max(1, self.tracker.cumulative_invested) * 100:.2f}%\n")
        
        excel_path = self.tracker.save_final_report(self.target_season)
        
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info("✅ DEPLOYMENT ABGESCHLOSSEN")
        self.file_logger.info("="*80)
        self.file_logger.info(f"\n📊 Final Stats:")
        self.file_logger.info(f"   Total Bets:      {self.tracker.cumulative_bets:>10,}")
        self.file_logger.info(f"   Total Wins:      {self.tracker.cumulative_wins:>10,}")
        self.file_logger.info(f"   Winrate:         {(self.tracker.cumulative_wins/max(1,self.tracker.cumulative_bets)*100):>10.2f}%")
        self.file_logger.info(f"   Total Invested:  {self.tracker.cumulative_invested:>10.2f}€")
        self.file_logger.info(f"   Total Profit:    {self.tracker.cumulative_profit:>10.2f}€")
        self.file_logger.info(f"   ROI:             {(self.tracker.cumulative_profit/max(1,self.tracker.cumulative_invested)*100):>10.2f}%")
        self.file_logger.info(f"\n📁 Excel Report: {excel_path}")
        self.file_logger.info("="*80 + "\n")
    
    def _deployment_without_learning(self, model):
        """Deployment ohne Updates."""
        self.file_logger.info("🎯 Standard Deployment (ohne Lernen)...\n")
        
        env = self._create_env(self.deploy_data, "deploy")
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            
            if step % 50 == 0:
                self.file_logger.info(
                    f"  Step {step:>3} | ROI: {info['roi']*100:>7.2f}% | Bets: {info['bets']:>3}"
                )
            
            step += 1
        
        self.file_logger.info(f"\n✅ Deployment abgeschlossen")
        self.file_logger.info(f"   Final ROI: {info['roi']*100:.2f}%\n")
    
    def run_full_pipeline(self):
        """Führe komplettes Training durch."""
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info("🤖 ADAPTIVE RL BETTING SYSTEM")
        self.file_logger.info("="*80 + "\n")
        
        global_model = self.phase1_global_training()
        finetuned_model = self.phase2_finetuning(global_model)
        self.phase3_incremental_deployment(finetuned_model)
        
        self.file_logger.info("\n" + "="*80)
        self.file_logger.info("✅ PIPELINE ABGESCHLOSSEN")
        self.file_logger.info("="*80 + "\n")


# ✅ CLI Entry Point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive RL Training")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()
    
    system = AdaptiveTrainingSystem(args.config)
    system.run_full_pipeline()