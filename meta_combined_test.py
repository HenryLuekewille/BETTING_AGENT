"""
Meta-Combined Test System with YAML Integration
✅ Lädt Konfiguration aus meta_test_config.yaml
✅ Nutzt base_training_template.yaml als Basis
✅ Speichert alle Configs in separaten YAMLs
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
        Erstellt vollständige Training-Config aus Template + Parametern.
        
        Args:
            params: Parameter-Dict
            season_cfg: Season-Konfiguration
            run_dir: Output-Verzeichnis
            
        Returns:
            Path: Pfad zur gespeicherten Config
        """
        # Kopiere Template
        config = self.base_template.copy()
        
        # Training Settings
        config["training"]["target_season"] = season_cfg["target"]
        config["training"]["global_seasons"]["start"] = season_cfg["train_start"]
        config["training"]["global_seasons"]["end"] = season_cfg["train_end"]
        config["training"]["global_timesteps"] = params["global_timesteps"]
        config["training"]["finetune_timesteps"] = params["finetune_timesteps"]
        
        # Environment Settings
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
        env["confidence_scaling_mode"] = params["confidence_scaling"]  # NEU!
        
        # Model Settings
        config["model"]["dqn"]["learning_rate"] = params["learning_rate"]
        
        # Paths
        for key in ["models_dir", "results_dir", "logs_dir", "checkpoints_dir"]:
            sub = run_dir / key
            sub.mkdir(parents=True, exist_ok=True)
            config["paths"][key] = str(sub)
        
        # Save Config
        cfg_path = run_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return cfg_path
    
    def generate_experiment_label(self, params: dict, season_cfg: dict):
        """Generiert eindeutiges Label für Experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        label = (
            f"S{season_cfg['target']}_"
            f"Seed{params['seed']}_"
            f"C{params['confidence']}_"
            f"Edge{params['min_edge']}_"
            f"Rate{params['max_bet_rate']}_"
            f"Scale-{params['confidence_scaling']}_"
            f"{'Kelly' if params['use_kelly'] else 'Flat'}_"
            f"{params['reward_mode']}"
        )
        
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
        
        # ✅ Setup Logger
        self.logger = logging.getLogger(f'meta_test_{mode}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing
        
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
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 META-TEST SYSTEM")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Dataset: {self.dataset['name']}")
        self.logger.info(f"Seasons: {len(self.dataset['season_pairs'])}")
        self.logger.info(f"Seeds: {len(self.dataset['seeds'])}")
        self.logger.info(f"Result Dir: {self.result_dir}")
        self.logger.info(f"Log File: {self.log_file}\n")
    
    def create_parameter_combinations(self, max_experiments: int = None):
        """Erstellt alle Parameter-Kombinationen."""
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
            "global_timesteps": self.search_space.get("global_timesteps_options", [1000000]),
            "finetune_timesteps": self.search_space.get("finetune_timesteps_options", [100000]),
            "learning_rate": self.search_space.get("learning_rates", [0.0001]),
        }
        
        # Generate all combinations
        keys = param_space.keys()
        values = param_space.values()
        all_combinations = list(itertools.product(*values))
        
        self.logger.info(f"📊 Suchraum-Statistik:")
        for key, vals in param_space.items():
            self.logger.info(f"   {key:25} {len(vals):>3} Optionen")
        
        total = len(all_combinations)
        self.logger.info(f"\n   Total Combinations: {total:,}")
        
        # Apply limit if specified
        if max_experiments and total > max_experiments:
            self.logger.info(f"\n⚡ Sampling {max_experiments} aus {total:,} Kombinationen")
            all_combinations = random.sample(all_combinations, max_experiments)
        
        # Convert to list of dicts
        experiments = []
        for combo in all_combinations:
            param_dict = dict(zip(keys, combo))
            experiments.append(param_dict)
        
        return experiments
    
    def run_experiment(self, experiment_id: int, params: dict):
        """
        Führt ein einzelnes Experiment aus.
        ✅ Mit Live-Log-Output
        
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
        self.logger.info(f"Confidence: {params['confidence']}")
        self.logger.info(f"Scaling: {params['confidence_scaling']}")
        self.logger.info(f"Min Edge: {params['min_edge']}")
        self.logger.info(f"Max Bet Rate: {params['max_bet_rate']}")
        self.logger.info("")
        
        # Generate label
        label, timestamp = self.config.generate_experiment_label(params, season_cfg)
        
        # Create run directory
        run_dir = self.result_dir / f"run_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config
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
        Führt Training aus mit LIVE-LOG-OUTPUT.
        ✅ FIX: Logs werden direkt gestreamt
        
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
            
            # ✅ FIX: LIVE-LOG-STREAMING
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            with open(out_log, 'w', encoding='utf-8') as f_out:
                for line in process.stdout:
                    # Write to file
                    f_out.write(line)
                    f_out.flush()
                    
                    # ✅ Print to console (wird von Flask erfasst)
                    print(line, end='', flush=True)
                    
                    # ✅ Log to meta log
                    self.logger.info(line.rstrip())
            
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
        """
        Extrahiert Ergebnisse aus Run.
        
        Args:
            run_dir: Run-Verzeichnis
            
        Returns:
            dict: Ergebnis-Dict oder None
        """
        results_dir = run_dir / "results"
        
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
        """
        Führt alle Experimente aus.
        
        Args:
            max_experiments: Max. Anzahl Experimente (optional)
        """
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
        """
        Speichert Zwischenergebnisse.
        
        Args:
            results: Liste der Experiment-Ergebnisse
        """
        summary_file = self.result_dir / "meta_experiment_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        self.logger.debug(f"   💾 Zwischenergebnisse gespeichert: {summary_file}")
    
    def _create_final_report(self, experiments: list):
        """
        Erstellt finalen Report.
        
        Args:
            experiments: Liste aller Experimente
        """
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
        
        # Best configurations
        self.logger.info(f"\n🏆 BEST CONFIGURATIONS:")
        
        for metric in ['roi', 'sharpe_ratio', 'winrate']:
            best = df.loc[df[metric].idxmax()]
            self.logger.info(f"\n   Best {metric.upper()}:")
            self.logger.info(f"      Value: {best[metric]:.4f}")
            self.logger.info(f"      Confidence: {best['confidence']}")
            self.logger.info(f"      Scaling: {best['confidence_scaling']}")
            self.logger.info(f"      Min Edge: {best['min_edge']}")
            self.logger.info(f"      Kelly: {best['kelly_fraction']}")
        
        # Save DataFrame
        csv_path = self.result_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False, sep=';')
        self.logger.info(f"\n💾 CSV gespeichert: {csv_path}")
        
        # Excel with multiple sheets
        excel_path = self.result_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            for metric in ['roi', 'sharpe_ratio', 'winrate']:
                top10 = df.nlargest(10, metric)
                top10.to_excel(writer, sheet_name=f'Top10_{metric}', index=False)
        
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