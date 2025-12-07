"""
Meta-Combined Test System with YAML Integration
✅ Lädt Konfiguration aus meta_test_config.yaml
✅ Nutzt base_training_template.yaml als Basis
✅ Speichert alle Configs in separaten YAMLs
✅ FIX: Kein doppeltes Logging mehr
✅ FIX: Fine-Tune Gamedays werden korrekt getestet
✅ FIX: Ensemble Weights werden korrekt getestet
"""

import sys
import itertools
import random
import os
import yaml
import json
import subprocess
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# Setup paths
BASE_DIR = Path(__file__).parent.absolute()
CORE_DIR = BASE_DIR / "core"
CONFIG_DIR = BASE_DIR / "config"
RESULT_ROOT = BASE_DIR / "meta_combined_results"
RESULT_ROOT.mkdir(exist_ok=True)

# Add to path
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


class MetaTestConfig:
    """Lädt und verwaltet Meta-Test-Konfiguration."""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = CONFIG_DIR / "meta_test_config.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Load base template
        template_path = CONFIG_DIR / "base_training_template.yaml"
        with open(template_path, "r", encoding="utf-8") as f:
            self.base_template = yaml.safe_load(f)
    
    def get_dataset_config(self, mode: str):
        """Gibt Dataset-Konfiguration für Mode zurück."""
        datasets = self.config["datasets"]
        
        if mode == "quick":
            return datasets["quick_test"]
        elif mode == "balanced":
            return datasets["balanced_test"]
        else:
            return datasets["comprehensive_test"]
    
    def get_search_space(self, mode: str):
        """Gibt Suchraum für Mode zurück."""
        if mode in self.config["modes"]:
            # Mode-spezifische Einschränkungen
            return self.config["modes"][mode]
        else:
            # Full search space
            return self.config["search_space"]
    
    def create_training_config(self, params: dict, season_cfg: dict, run_dir: Path):
        """
        ✅ FIX: Erstellt vollständige Training-Config mit ALLEN Parametern.
        
        Args:
            params: Parameter-Dict
            season_cfg: Season-Konfiguration
            run_dir: Output-Verzeichnis
            
        Returns:
            Path: Pfad zur gespeicherten Config
        """
        # ✅ Deep copy to avoid reference issues
        import copy
        config = copy.deepcopy(self.base_template)
        
        # ✅ Training Settings (ALLE Parameter!)
        config["training"]["target_season"] = season_cfg["target"]
        config["training"]["global_seasons"]["start"] = season_cfg["train_start"]
        config["training"]["global_seasons"]["end"] = season_cfg["train_end"]
        config["training"]["global_timesteps"] = params["global_timesteps"]
        config["training"]["finetune_timesteps"] = params["finetune_timesteps"]
        config["training"]["fine_tune_gamedays"] = params["fine_tune_gamedays"]  # ✅ FIX!
        
        # ✅ Environment Settings
        env = config["environment"]
        env["confidence_threshold"] = params["confidence"]
        env["min_edge_required"] = params["min_edge"]
        env["max_bet_rate"] = params["max_bet_rate"]
        env["bet_amount"] = params["base_bet"]
        env["max_bet_amount"] = params["max_bet"]
        env["use_kelly_criterion"] = params["use_kelly"]
        env["kelly_fraction"] = params["kelly_fraction"]
        env["reward_shaping"] = params["reward_mode"]
        env["no_bet_reward_multiplier"] = params["no_bet_mult"]
        env["draw_penalty_multiplier"] = params["draw_penalty"]
        env["confidence_scaling_mode"] = params["confidence_scaling"]
        
        # ✅ Model Settings
        config["model"]["dqn"]["learning_rate"] = params["learning_rate"]
        
        # ✅ Model Ensemble (falls aktiviert)
        if "ensemble_weights" in params:
            ensemble = params["ensemble_weights"]
            config["training"]["model_ensemble"] = {
                "enabled": True,
                "global_weight": ensemble["global"],
                "finetune_weight": ensemble["finetune"],
                "ensemble_name": ensemble["name"]
            }
        else:
            # Deaktiviere Ensemble
            config["training"]["model_ensemble"] = {
                "enabled": False
            }
        
        # ✅ Paths (relative zu run_dir)
        for key in ["models_dir", "results_dir", "logs_dir", "checkpoints_dir"]:
            sub = run_dir / key
            sub.mkdir(parents=True, exist_ok=True)
            config["paths"][key] = str(sub)
        
        # ✅ Save Config
        cfg_path = run_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return cfg_path
    
    def generate_experiment_label(self, params: dict, season_cfg: dict):
        """Generiert eindeutiges Label für Experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ✅ Include Fine-Tune Gamedays in label
        label = (
            f"S{season_cfg['target']}_"
            f"Seed{params['seed']}_"
            f"FT{params['fine_tune_gamedays']}_"  # ✅ NEU!
            f"C{params['confidence']}_"
            f"Edge{params['min_edge']}_"
            f"Rate{params['max_bet_rate']}_"
            f"Scale-{params['confidence_scaling']}_"
            f"{'Kelly' if params['use_kelly'] else 'Flat'}_"
            f"{params['reward_mode']}"
        )
        
        # ✅ Add ensemble info if present
        if "ensemble_weights" in params:
            label += f"_Ens-{params['ensemble_weights']['name']}"
        
        return label, timestamp


class MetaTestRunner:
    """Führt Meta-Tests aus mit Live-Log-Output."""
    
    def __init__(self, config: MetaTestConfig, mode: str = "balanced"):
        self.config = config
        self.mode = mode
        
        # Load dataset and search space
        self.dataset = config.get_dataset_config(mode)
        self.search_space = config.get_search_space(mode)
        
        # ✅ Setup Result Directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = RESULT_ROOT / f"meta_{mode}_{timestamp}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ Setup Log File
        self.log_file = self.result_dir / "meta_test.log"
        
        # ✅ Setup Logger (NUR EINMAL!)
        self.logger = self._setup_logger()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 META-TEST SYSTEM")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Dataset: {self.dataset['name']}")
        self.logger.info(f"Seasons: {len(self.dataset['season_pairs'])}")
        self.logger.info(f"Seeds: {len(self.dataset['seeds'])}")
        self.logger.info(f"Result Dir: {self.result_dir}")
        self.logger.info(f"Log File: {self.log_file}\n")
    
    def _setup_logger(self):
        """✅ FIX: Setup Logger NUR EINMAL (verhindert doppelte Ausgaben)."""
        logger_name = f'meta_test_{self.mode}_{id(self)}'  # ✅ Unique name
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # ✅ Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # ✅ Prevent propagation
        logger.propagate = False
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File Handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def create_parameter_combinations(self, max_experiments: int = None):
        """✅ FIX: Erstellt ALLE Parameter-Kombinationen inkl. Fine-Tune Gamedays & Ensemble Weights."""
        
        # Build parameter space
        param_space = {
            "season_pairs": self.dataset["season_pairs"],
            "seed": self.dataset["seeds"],
            "confidence": self.search_space.get("confidence_thresholds", [0.60]),
            "min_edge": self.search_space.get("min_edge_values", [0.05]),
            "max_bet_rate": self.search_space.get("max_bet_rates", [0.30]),
            "base_bet": self.search_space.get("base_bet_amounts", [10.0]),
            "max_bet": self.search_space.get("max_bet_amounts", [30]),
            "use_kelly": self.search_space.get("use_kelly_options", [True]),
            "kelly_fraction": self.search_space.get("kelly_fractions", [0.25]),
            "confidence_scaling": self.search_space.get("confidence_scaling_modes", ["none"]),
            "reward_mode": self.search_space.get("reward_modes", ["balanced"]),
            "no_bet_mult": self.search_space.get("no_bet_multipliers", [0.5]),
            "draw_penalty": self.search_space.get("draw_penalties", [1.5]),
            
            # ✅ FIX: ALLE Fine-Tune Gamedays Optionen
            "fine_tune_gamedays": self.search_space.get("fine_tune_gamedays_options", [4, 8, 12, 16]),
            
            "global_timesteps": self.search_space.get("global_timesteps_options", [1000000]),
            "finetune_timesteps": self.search_space.get("finetune_timesteps_options", [100000]),
            "learning_rate": self.search_space.get("learning_rates", [0.0001]),
        }
        
        # ✅ FIX: Ensemble Weights als EIGENE Dimension behandeln
        ensemble_weights_list = self.search_space.get("model_ensemble_weights", [])
        
        if ensemble_weights_list:
            # ✅ Füge Ensemble Weights als Parameter hinzu
            param_space["ensemble_weights"] = ensemble_weights_list
            self.logger.info(f"\n✅ Ensemble Weights aktiviert: {len(ensemble_weights_list)} Konfigurationen")
        else:
            # ✅ Fallback: Nur "pure_finetune" (kein Ensemble)
            param_space["ensemble_weights"] = [
                {"global": 0.0, "finetune": 1.0, "name": "pure_finetune"}
            ]
            self.logger.info(f"\n⚠️  Kein Ensemble konfiguriert, nutze pure_finetune")
        
        # Generate all combinations
        keys = list(param_space.keys())
        values = list(param_space.values())
        all_combinations = list(itertools.product(*values))
        
        # ✅ Extended Debug Output
        self.logger.info(f"\n📊 Suchraum-Statistik:")
        for key, vals in param_space.items():
            if key == "ensemble_weights":
                # Special handling for ensemble weights
                names = [w['name'] for w in vals]
                self.logger.info(f"   {key:30} {len(vals):>3} Optionen: {names}")
            elif key == "season_pairs":
                seasons = [f"{s['train_start']}-{s['train_end']}→{s['target']}" for s in vals]
                self.logger.info(f"   {key:30} {len(vals):>3} Optionen: {seasons}")
            else:
                val_preview = vals if len(vals) <= 5 else f"{vals[:3]}...+{len(vals)-3} more"
                self.logger.info(f"   {key:30} {len(vals):>3} Optionen: {val_preview}")
        
        total = len(all_combinations)
        self.logger.info(f"\n   Total Combinations: {total:,}")
        
        # ✅ Show example combinations
        if total > 0:
            first = dict(zip(keys, all_combinations[0]))
            self.logger.info(f"\n   📝 Beispiel-Kombination #1:")
            self.logger.info(f"      Fine-Tune GDs:  {first.get('fine_tune_gamedays')}")
            self.logger.info(f"      Ensemble:       {first.get('ensemble_weights', {}).get('name', 'N/A')}")
            self.logger.info(f"      Confidence:     {first.get('confidence')}")
            self.logger.info(f"      Min Edge:       {first.get('min_edge')}")
            self.logger.info(f"      Seed:           {first.get('seed')}")
            
            if total > 1:
                last = dict(zip(keys, all_combinations[-1]))
                self.logger.info(f"\n   📝 Beispiel-Kombination #{total}:")
                self.logger.info(f"      Fine-Tune GDs:  {last.get('fine_tune_gamedays')}")
                self.logger.info(f"      Ensemble:       {last.get('ensemble_weights', {}).get('name', 'N/A')}")
                self.logger.info(f"      Confidence:     {last.get('confidence')}")
                self.logger.info(f"      Seed:           {last.get('seed')}")
        
        # Apply limit
        if max_experiments and total > max_experiments:
            self.logger.info(f"\n⚡ Sampling {max_experiments} aus {total:,} Kombinationen")
            all_combinations = random.sample(all_combinations, max_experiments)
        
        # Convert to dicts
        experiments = []
        for combo in all_combinations:
            param_dict = dict(zip(keys, combo))
            experiments.append(param_dict)
        
        # ✅ Verify: Count unique values
        unique_ft_gds = sorted(set(exp['fine_tune_gamedays'] for exp in experiments))
        unique_ensembles = sorted(set(exp['ensemble_weights']['name'] for exp in experiments))
        unique_confs = sorted(set(exp['confidence'] for exp in experiments))
        unique_seeds = sorted(set(exp['seed'] for exp in experiments))
        
        self.logger.info(f"\n   ✅ Verification - Unique Values in Experiments:")
        self.logger.info(f"      Fine-Tune GDs tested:  {unique_ft_gds}")
        self.logger.info(f"      Ensembles tested:      {unique_ensembles}")
        self.logger.info(f"      Confidences tested:    {unique_confs}")
        self.logger.info(f"      Seeds tested:          {unique_seeds}")
        self.logger.info(f"\n   📊 Total Experiments:      {len(experiments):,}")
        
        return experiments
    
    def run_experiment(self, experiment_id: int, params: dict):
        """
        ✅ FIX: Führt ein einzelnes Experiment aus mit korrekten Parametern.
        
        Args:
            experiment_id: Experiment-Nummer
            params: Parameter-Dict
            
        Returns:
            dict: Ergebnis-Dict
        """
        # Extract season config
        season_cfg = params.pop("season_pairs")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🧪 EXPERIMENT {experiment_id}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Season: {season_cfg['target']}")
        self.logger.info(f"Seed: {params['seed']}")
        self.logger.info(f"Fine-Tune Gamedays: {params['fine_tune_gamedays']}")  # ✅ NEU!
        self.logger.info(f"Confidence: {params['confidence']}")
        self.logger.info(f"Scaling: {params['confidence_scaling']}")
        self.logger.info(f"Min Edge: {params['min_edge']}")
        self.logger.info(f"Max Bet Rate: {params['max_bet_rate']}")
        
        # ✅ Log ensemble if present
        if "ensemble_weights" in params:
            ens = params["ensemble_weights"]
            self.logger.info(f"Ensemble: {ens['name']} (G:{ens['global']}, F:{ens['finetune']})")
        
        self.logger.info("")
        
        # Generate label
        label, timestamp = self.config.generate_experiment_label(params, season_cfg)
        
        # Create run directory
        run_dir = self.result_dir / f"run_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ Create config (mit allen Parametern!)
        cfg_path = self.config.create_training_config(params, season_cfg, run_dir)
        
        # Set seed
        os.environ["PYTHONHASHSEED"] = str(params["seed"])
        random.seed(params["seed"])
        np.random.seed(params["seed"])
        
        # Run training
        success = self._run_training(cfg_path, label)
        
        # Extract results
        results = self._extract_results(run_dir) if success else None
        
        # Store metadata
        experiment_data = {
            "experiment_id": experiment_id,
            "label": label,
            "target_season": season_cfg["target"],
            "train_start": season_cfg["train_start"],
            "train_end": season_cfg["train_end"],
            "success": success,
            "run_path": str(run_dir),
            "config_path": str(cfg_path),
            **params
        }
        
        if results:
            experiment_data.update(results)
        
        return experiment_data
    
    def _run_training(self, cfg_path: Path, label: str, timeout: int = 7200):
        """
        ✅ FIX: Führt Training aus mit LIVE-LOG-OUTPUT (keine doppelten Logs).
        
        Args:
            cfg_path: Path zur Config-Datei
            label: Experiment-Label
            timeout: Timeout in Sekunden
            
        Returns:
            bool: True wenn erfolgreich
        """
        out_log = self.result_dir / f"{label}.log"
        
        cmd = [
            sys.executable,
            str(CORE_DIR / 'adaptive_training.py'),
            str(cfg_path)
        ]
        
        try:
            self.logger.info(f"🚀 Starte Training: {label}")
            self.logger.info(f"   Config: {cfg_path}")
            self.logger.info(f"   Log: {out_log}")
            self.logger.info("")
            
            # ✅ LIVE-LOG-STREAMING (ohne doppelte Ausgaben)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            with open(out_log, 'w', encoding='utf-8') as f_out:
                for line in process.stdout:
                    # Write to file
                    f_out.write(line)
                    f_out.flush()
                    
                    # ✅ Print to console (no logging - prevents duplicates)
                    print(line, end='', flush=True)
            
            process.wait(timeout=timeout)
            
            if process.returncode == 0:
                self.logger.info(f"\n✅ Run abgeschlossen: {label}\n")
                return True
            else:
                self.logger.error(f"\n❌ Fehler in Run {label} (Code: {process.returncode})\n")
                return False
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"\n⏱️  Timeout nach {timeout}s: {label}\n")
            if process:
                process.kill()
            return False
            
        except Exception as e:
            self.logger.error(f"\n❌ Unerwarteter Fehler: {e}\n")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_results(self, run_dir: Path):
        """Extrahiert Ergebnisse aus Run."""
        results_dir = run_dir / "results_dir"
        
        if not results_dir.exists():
            self.logger.warning(f"   ⚠️  Kein results-Verzeichnis gefunden: {results_dir}")
            return None
        
        excel_files = list(results_dir.glob("evaluation_*.xlsx"))
        
        if not excel_files:
            self.logger.warning(f"   ⚠️  Keine Excel-Dateien gefunden in {results_dir}")
            return None
        
        try:
            excel_file = excel_files[0]
            df = pd.read_excel(excel_file, sheet_name='Gameday_Summaries')
            
            final_roi = df['CumulativeROI'].iloc[-1] if len(df) > 0 else 0
            final_winrate = df['CumulativeWinrate'].iloc[-1] if len(df) > 0 else 0
            total_bets = df['CumulativeBets'].iloc[-1] if len(df) > 0 else 0
            total_profit = df['CumulativeProfit'].iloc[-1] if len(df) > 0 else 0
            
            returns = df['TotalProfit'].values
            sharpe = (returns.mean() / returns.std()) if len(returns) > 1 and returns.std() > 0 else 0
            
            cumulative = df['CumulativeProfit'].values
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
            
            self.logger.info(f"   ✅ Ergebnisse extrahiert:")
            self.logger.info(f"      ROI: {final_roi*100:.2f}%")
            self.logger.info(f"      Winrate: {final_winrate*100:.2f}%")
            self.logger.info(f"      Total Bets: {int(total_bets)}")
            self.logger.info(f"      Sharpe: {sharpe:.2f}")
            
            return {
                'roi': final_roi,
                'winrate': final_winrate,
                'total_bets': int(total_bets),
                'total_profit': float(total_profit),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_dd),
                'excel_file': excel_file.name,
            }
            
        except Exception as e:
            self.logger.error(f"   ⚠️  Fehler beim Extrahieren: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_experiments(self, max_experiments: int = None):
        """Führt alle Experimente aus."""
        experiments = self.create_parameter_combinations(max_experiments)
        
        self.logger.info(f"\n🚀 Starte {len(experiments)} Experimente...\n")
        
        results = []
        successful = 0
        failed = 0
        
        for i, params in enumerate(experiments, 1):
            result = self.run_experiment(i, params)
            results.append(result)
            
            if result.get("success"):
                successful += 1
            else:
                failed += 1
            
            # Save intermediate
            self._save_results(results)
            
            self.logger.info(f"\n✅ Progress: {successful} successful, {failed} failed")
            self.logger.info(f"   Total: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)\n")
        
        # Final analysis
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 FINAL ANALYSIS")
        self.logger.info(f"{'='*80}\n")
        
        self._create_final_report(results)
    
    def _save_results(self, results: list):
        """Speichert Zwischenergebnisse."""
        summary_file = self.result_dir / "meta_experiment_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    
    def _create_final_report(self, experiments: list):
        """Erstellt finalen Report mit Ensemble & Fine-Tune Analyse."""
        successful = [e for e in experiments if e.get('success') and e.get('roi') is not None]
        
        if not successful:
            self.logger.error("❌ Keine erfolgreichen Experimente!")
            return
        
        df = pd.DataFrame(successful)
        
        self.logger.info(f"\n✅ Erfolgreiche Runs: {len(successful)}/{len(experiments)}")
        self.logger.info(f"\n📈 OVERALL STATISTICS:")
        self.logger.info(f"   ROI:          {df['roi'].mean()*100:>8.2f}% (±{df['roi'].std()*100:.2f}%)")
        self.logger.info(f"   Winrate:      {df['winrate'].mean()*100:>8.2f}%")
        self.logger.info(f"   Sharpe:       {df['sharpe_ratio'].mean():>8.2f}")
        self.logger.info(f"   Max Drawdown: {df['max_drawdown'].mean():>8.2f}€")
        
        # ✅ Best configurations
        self.logger.info(f"\n🏆 BEST CONFIGURATIONS:")
        
        for metric in ['roi', 'sharpe_ratio', 'winrate']:
            best = df.loc[df[metric].idxmax()]
            self.logger.info(f"\n   Best {metric.upper()}:")
            self.logger.info(f"      Value:        {best[metric]:.4f}")
            self.logger.info(f"      Confidence:   {best['confidence']}")
            self.logger.info(f"      Fine-Tune GDs: {best['fine_tune_gamedays']}")
            
            if 'ensemble_weights' in best:
                ens = best['ensemble_weights']
                if isinstance(ens, dict):
                    self.logger.info(f"      Ensemble:     {ens.get('name', 'N/A')}")
            
            self.logger.info(f"      Scaling:      {best['confidence_scaling']}")
            self.logger.info(f"      Min Edge:     {best['min_edge']}")
            self.logger.info(f"      Kelly:        {best['kelly_fraction']}")
        
        # ✅ Fine-Tune Gamedays Analysis
        self.logger.info(f"\n📊 FINE-TUNE GAMEDAYS ANALYSIS:")
        
        if 'fine_tune_gamedays' in df.columns:
            ft_analysis = df.groupby('fine_tune_gamedays').agg({
                'roi': ['mean', 'std', 'count'],
                'winrate': 'mean',
                'sharpe_ratio': 'mean'
            }).round(4)
            
            self.logger.info(f"\n{ft_analysis.to_string()}")
        
        # ✅ Ensemble Weights Analysis
        self.logger.info(f"\n🎭 ENSEMBLE WEIGHTS ANALYSIS:")
        
        if 'ensemble_weights' in df.columns:
            # Extract ensemble names
            df['ensemble_name'] = df['ensemble_weights'].apply(
                lambda x: x.get('name', 'unknown') if isinstance(x, dict) else 'unknown'
            )
            
            ensemble_analysis = df.groupby('ensemble_name').agg({
                'roi': ['mean', 'std', 'count'],
                'winrate': 'mean',
                'sharpe_ratio': 'mean'
            }).round(4)
            
            self.logger.info(f"\n{ensemble_analysis.to_string()}")
        
        # Save DataFrame
        csv_path = self.result_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False, sep=';')
        self.logger.info(f"\n💾 CSV gespeichert: {csv_path}")
        
        # Excel with multiple sheets
        excel_path = self.result_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Top 10 for each metric
            for metric in ['roi', 'sharpe_ratio', 'winrate']:
                top10 = df.nlargest(10, metric)
                top10.to_excel(writer, sheet_name=f'Top10_{metric}', index=False)
            
            # ✅ Fine-Tune Gamedays Sheet
            if 'fine_tune_gamedays' in df.columns:
                ft_analysis.to_excel(writer, sheet_name='FineTune_Analysis')
            
            # ✅ Ensemble Analysis Sheet
            if 'ensemble_name' in df.columns:
                ensemble_analysis.to_excel(writer, sheet_name='Ensemble_Analysis')
        
        self.logger.info(f"💾 Excel gespeichert: {excel_path}")
        self.logger.info(f"\n{'='*80}\n")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Meta-Combined Test System für Betting Agent"
    )
    parser.add_argument(
        "--mode",
        choices=['quick', 'balanced', 'comprehensive'],
        default='balanced',
        help="Search mode"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="Max number of experiments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to meta_test_config.yaml"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = MetaTestConfig(args.config)
    
    # Run tests
    runner = MetaTestRunner(config, mode=args.mode)
    runner.run_all_experiments(max_experiments=args.max_experiments)