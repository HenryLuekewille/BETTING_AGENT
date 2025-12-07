"""
Enhanced Parameter Search für Betting Agent
✅ Speichert alle Konfigurationen
✅ Vergleicht Performance-Metriken
✅ Findet optimale Settings
✅ Detaillierte Analyse & Visualisierung
"""

import sys
import yaml
import json
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add core to path
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

from core.adaptive_training import AdaptiveTrainingSystem


class AdvancedParameterSearch:
    """
    Systematische Parameter-Suche mit erweiterten Features:
    - Multi-Objective Optimization
    - Bayesian Optimization Support
    - Detaillierte Performance-Analyse
    - Automatische Best-Config Speicherung
    """
    
    def __init__(self, base_config_path, target_season=2024):
        self.base_config_path = Path(base_config_path)
        self.target_season = target_season
        
        with open(base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
        
        # Results storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = BASE_DIR / "parameter_search_results" / f"search_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = []
        self.best_configs = []
        
        # Performance tracking
        self.performance_history = {
            'roi': [],
            'winrate': [],
            'sharpe': [],
            'max_drawdown': []
        }
        
        print(f"\n{'='*80}")
        print(f"🔍 ADVANCED PARAMETER SEARCH")
        print(f"{'='*80}")
        print(f"Target Season: {target_season}")
        print(f"Results Dir:   {self.results_dir}\n")
    
    def define_search_space(self, mode='comprehensive'):
        """
        Definiert verschiedene Such-Modi.
        
        Args:
            mode: 'quick', 'balanced', 'comprehensive', 'custom'
        
        Returns:
            dict: Parameter-Grid
        """
        if mode == 'quick':
            # Schneller Test (wenige Kombinationen)
            return {
                "confidence_threshold": [0.60, 0.70],
                "min_edge_required": [0.02, 0.05],
                "max_bet_rate": [0.30, 0.40],
                "max_bet_amount": [15, 30],
                "use_kelly_criterion": [True],
                "kelly_fraction": [0.25],
                "global_timesteps": [500000],
                "learning_rate": [0.0001],
            }
        
        elif mode == 'balanced':
            # Ausgewogener Test
            return {
                "confidence_threshold": [0.55, 0.60, 0.65],
                "min_edge_required": [0.00, 0.02, 0.05],
                "max_bet_rate": [0.30, 0.40],
                "max_bet_amount": [10, 20, 30],
                "use_kelly_criterion": [True, False],
                "kelly_fraction": [0.25],
                "no_bet_reward_multiplier": [0.5],
                "draw_penalty_multiplier": [1.5],
                "global_timesteps": [1000000],
                "learning_rate": [0.0001],
            }
        
        elif mode == 'comprehensive':
            # Vollständiger Test
            return {
                "confidence_threshold": [0.55, 0.60, 0.65, 0.70],
                "min_edge_required": [0.00, 0.02, 0.05, 0.08],
                "max_bet_rate": [0.30, 0.40, 0.50],
                "max_bet_amount": [10, 15, 20, 30],
                "use_kelly_criterion": [True, False],
                "kelly_fraction": [0.15, 0.25, 0.35],
                "no_bet_reward_multiplier": [0.3, 0.5, 0.7],
                "draw_penalty_multiplier": [1.2, 1.5, 2.0],
                "global_timesteps": [500000, 1000000],
                "finetune_timesteps": [50000, 100000],
                "learning_rate": [0.0001, 0.0005],
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def create_param_combinations(
        self, 
        search_space: Dict, 
        max_combinations: int = None,
        sampling: str = 'random'
    ) -> List[Dict]:
        """
        Erstellt Parameter-Kombinationen.
        
        Args:
            search_space: Parameter-Grid
            max_combinations: Limit (None = alle)
            sampling: 'random', 'latin_hypercube', 'grid'
        
        Returns:
            List[Dict]: Parameter-Kombinationen
        """
        keys = search_space.keys()
        values = search_space.values()
        all_combinations = list(itertools.product(*values))
        
        print(f"📊 Suchraum:")
        for key, vals in search_space.items():
            print(f"   {key}: {len(vals)} Optionen")
        
        print(f"\n   Total Combinations: {len(all_combinations):,}")
        
        # Sampling
        if max_combinations and len(all_combinations) > max_combinations:
            if sampling == 'random':
                import random
                combinations = random.sample(all_combinations, max_combinations)
                print(f"   ⚡ Random Sampling: {max_combinations}")
            
            elif sampling == 'latin_hypercube':
                # Latin Hypercube Sampling für bessere Coverage
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=len(keys))
                sample = sampler.random(n=max_combinations)
                
                combinations = []
                for s in sample:
                    combo = []
                    for i, (key, vals) in enumerate(search_space.items()):
                        idx = int(s[i] * len(vals))
                        idx = min(idx, len(vals) - 1)
                        combo.append(vals[idx])
                    combinations.append(tuple(combo))
                
                print(f"   🎯 Latin Hypercube Sampling: {max_combinations}")
            
            else:
                combinations = all_combinations[:max_combinations]
                print(f"   📐 Grid Sampling: {max_combinations}")
        else:
            combinations = all_combinations
            print(f"   ✅ Using all combinations\n")
        
        # Convert to dicts
        param_dicts = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def run_single_experiment(
        self, 
        params: Dict, 
        experiment_id: int
    ) -> Dict:
        """Führt ein Experiment durch."""
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {experiment_id}")
        print(f"{'='*80}")
        
        for key, value in params.items():
            print(f"   {key}: {value}")
        print()
        
        # Create config
        config = self._create_config(params, experiment_id)
        
        # Save config
        config_file = self.results_dir / f"config_exp_{experiment_id}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        try:
            # Run training
            system = AdaptiveTrainingSystem(str(config_file))
            system.run_full_pipeline()
            
            # Extract results
            results = self._extract_comprehensive_results(
                system.tracker, 
                params, 
                experiment_id
            )
            
            # Save experiment results
            self._save_experiment_results(results, experiment_id)
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment {experiment_id} fehlgeschlagen: {e}\n")
            
            error_results = {
                "experiment_id": experiment_id,
                "params": params,
                "success": False,
                "error": str(e)
            }
            
            self._save_experiment_results(error_results, experiment_id)
            
            return error_results
    
    def _extract_comprehensive_results(
        self, 
        tracker, 
        params: Dict, 
        experiment_id: int
    ) -> Dict:
        """Extrahiert umfassende Metriken."""
        
        # Basic Metrics
        roi = tracker.cumulative_profit / max(1, tracker.cumulative_invested)
        winrate = tracker.cumulative_wins / max(1, tracker.cumulative_bets)
        avg_profit_per_bet = tracker.cumulative_profit / max(1, tracker.cumulative_bets)
        
        # Advanced Metrics
        max_drawdown = self._calculate_max_drawdown(tracker.gameday_results)
        sharpe_ratio = self._calculate_sharpe_ratio(tracker.gameday_results)
        profit_factor = self._calculate_profit_factor(tracker.all_decisions)
        sortino_ratio = self._calculate_sortino_ratio(tracker.gameday_results)
        calmar_ratio = roi / max(0.01, max_drawdown) if roi > 0 else 0
        
        # Betting Behavior
        avg_bet_size = tracker.cumulative_invested / max(1, tracker.cumulative_bets)
        bet_frequency = tracker.cumulative_bets / max(1, len(tracker.gameday_results))
        
        # Risk Metrics
        volatility = self._calculate_volatility(tracker.gameday_results)
        max_consecutive_losses = self._calculate_max_consecutive_losses(tracker.all_decisions)
        
        # Action Distribution
        action_dist = self._analyze_action_distribution(tracker.all_decisions)
        
        # Composite Score (Multi-Objective)
        composite_score = self._calculate_composite_score(
            roi, winrate, sharpe_ratio, max_drawdown, profit_factor
        )
        
        results = {
            # Meta
            "experiment_id": experiment_id,
            "params": params,
            "success": True,
            
            # Core Performance
            "roi": roi,
            "winrate": winrate,
            "total_profit": tracker.cumulative_profit,
            "total_invested": tracker.cumulative_invested,
            "total_bets": tracker.cumulative_bets,
            "total_wins": tracker.cumulative_wins,
            "avg_profit_per_bet": avg_profit_per_bet,
            
            # Risk Metrics
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar_ratio,
            "volatility": volatility,
            
            # Betting Behavior
            "avg_bet_size": avg_bet_size,
            "bet_frequency": bet_frequency,
            "max_consecutive_losses": max_consecutive_losses,
            
            # Action Analysis
            "action_distribution": action_dist,
            
            # Composite Score
            "composite_score": composite_score,
        }
        
        return results
    
    def _calculate_composite_score(
        self, 
        roi: float, 
        winrate: float, 
        sharpe: float, 
        max_dd: float, 
        profit_factor: float
    ) -> float:
        """
        Berechnet Composite Score für Multi-Objective Optimization.
        
        Gewichtung:
        - ROI: 30%
        - Winrate: 20%
        - Sharpe Ratio: 25%
        - Max Drawdown (inverted): 15%
        - Profit Factor: 10%
        """
        # Normalize Metriken
        roi_norm = np.clip(roi * 100, -50, 100) / 100  # -50% bis 100%
        winrate_norm = np.clip(winrate, 0, 1)
        sharpe_norm = np.clip(sharpe / 3, -1, 1)  # -3 bis 3
        drawdown_norm = 1 - np.clip(max_dd / 100, 0, 1)  # 0 bis 100€
        pf_norm = np.clip(profit_factor / 3, 0, 1)  # 0 bis 3
        
        # Weighted Score
        score = (
            0.30 * roi_norm +
            0.20 * winrate_norm +
            0.25 * sharpe_norm +
            0.15 * drawdown_norm +
            0.10 * pf_norm
        )
        
        return score
    
    def _calculate_sortino_ratio(self, gameday_results: List[Dict]) -> float:
        """Sortino Ratio (nur Downside Volatility)."""
        if not gameday_results or len(gameday_results) < 2:
            return 0.0
        
        df = pd.DataFrame(gameday_results)
        returns = df["TotalProfit"].values
        
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return mean_return / 0.01  # Very high if no losses
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        return mean_return / downside_std
    
    def _calculate_volatility(self, gameday_results: List[Dict]) -> float:
        """Volatilität der Returns."""
        if not gameday_results or len(gameday_results) < 2:
            return 0.0
        
        df = pd.DataFrame(gameday_results)
        returns = df["TotalProfit"].values
        
        return returns.std()
    
    def _calculate_max_consecutive_losses(self, decisions: List[Dict]) -> int:
        """Maximale Serie von Verlusten."""
        if not decisions:
            return 0
        
        df = pd.DataFrame(decisions)
        
        max_streak = 0
        current_streak = 0
        
        for won in df["Won"]:
            if not won:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _analyze_action_distribution(self, decisions: List[Dict]) -> Dict:
        """Analysiert Action-Verteilung."""
        if not decisions:
            return {}
        
        df = pd.DataFrame(decisions)
        
        dist = df.groupby("ActionName").agg({
            "BetAmount": "count",
            "Won": "sum",
            "Profit": "sum"
        }).to_dict('index')
        
        return dist
    
    def _calculate_max_drawdown(self, gameday_results: List[Dict]) -> float:
        """Max Drawdown."""
        if not gameday_results:
            return 0.0
        
        df = pd.DataFrame(gameday_results)
        cumulative = df["CumulativeProfit"].values
        
        running_max = cumulative[0]
        max_dd = 0.0
        
        for value in cumulative:
            running_max = max(running_max, value)
            drawdown = running_max - value
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, gameday_results: List[Dict]) -> float:
        """Sharpe Ratio."""
        if not gameday_results or len(gameday_results) < 2:
            return 0.0
        
        df = pd.DataFrame(gameday_results)
        returns = df["TotalProfit"].values
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _calculate_profit_factor(self, decisions: List[Dict]) -> float:
        """Profit Factor."""
        if not decisions:
            return 0.0
        
        df = pd.DataFrame(decisions)
        
        wins = df[df["Profit"] > 0]["Profit"].sum()
        losses = abs(df[df["Profit"] < 0]["Profit"].sum())
        
        if losses == 0:
            return wins if wins > 0 else 0.0
        
        return wins / losses
    
    def _create_config(self, params: Dict, exp_id: int) -> Dict:
        """Erstellt Config aus Parametern."""
        config = self.base_config.copy()
        
        # Training params
        config["training"]["target_season"] = self.target_season
        config["training"]["global_timesteps"] = params.get("global_timesteps", 1000000)
        config["training"]["finetune_timesteps"] = params.get("finetune_timesteps", 100000)
        
        # Model params
        config["model"]["dqn"]["learning_rate"] = params.get("learning_rate", 0.0001)
        
        # Environment params
        config["environment"]["confidence_threshold"] = params["confidence_threshold"]
        config["environment"]["min_edge_required"] = params["min_edge_required"]
        config["environment"]["max_bet_rate"] = params["max_bet_rate"]
        config["environment"]["max_bet_amount"] = params["max_bet_amount"]
        config["environment"]["use_kelly_criterion"] = params.get("use_kelly_criterion", True)
        config["environment"]["kelly_fraction"] = params.get("kelly_fraction", 0.25)
        config["environment"]["no_bet_reward_multiplier"] = params.get("no_bet_reward_multiplier", 0.5)
        config["environment"]["draw_penalty_multiplier"] = params.get("draw_penalty_multiplier", 1.5)
        
        # Paths
        run_dir = self.results_dir / f"exp_{exp_id}"
        
        config["paths"]["models_dir"] = str(run_dir / "models")
        config["paths"]["checkpoints_dir"] = str(run_dir / "checkpoints")
        config["paths"]["results_dir"] = str(run_dir / "results")
        config["paths"]["logs_dir"] = str(run_dir / "logs")
        
        return config
    
    def _save_experiment_results(self, results: Dict, exp_id: int):
        """Speichert Experiment-Ergebnisse."""
        results_file = self.results_dir / f"results_exp_{exp_id}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    def run_search(
        self, 
        mode: str = 'balanced', 
        max_experiments: int = None,
        sampling: str = 'random'
    ):
        """Führt komplette Suche durch."""
        
        # Define search space
        search_space = self.define_search_space(mode)
        
        # Create combinations
        param_combinations = self.create_param_combinations(
            search_space, 
            max_experiments,
            sampling
        )
        
        print(f"🚀 Starte {len(param_combinations)} Experimente...\n")
        
        # Run experiments
        for i, params in enumerate(param_combinations, 1):
            print(f"\n{'#'*80}")
            print(f"# EXPERIMENT {i}/{len(param_combinations)}")
            print(f"{'#'*80}\n")
            
            results = self.run_single_experiment(params, i)
            self.all_results.append(results)
            
            # Track performance
            if results.get("success"):
                self.performance_history['roi'].append(results['roi'])
                self.performance_history['winrate'].append(results['winrate'])
                self.performance_history['sharpe'].append(results['sharpe_ratio'])
                self.performance_history['max_drawdown'].append(results['max_drawdown'])
            
            # Save intermediate results
            self._save_all_results()
            
            # Update best configs
            self._update_best_configs(results)
        
        # Final analysis
        self._comprehensive_analysis()
    
    def _update_best_configs(self, results: Dict):
        """Updated Best Configs basierend auf verschiedenen Metriken."""
        if not results.get("success"):
            return
        
        # Keep top 5 for each metric
        metrics = ['roi', 'composite_score', 'sharpe_ratio', 'winrate']
        
        for metric in metrics:
            # Sort by metric
            sorted_results = sorted(
                [r for r in self.all_results if r.get("success")],
                key=lambda x: x.get(metric, -999),
                reverse=True
            )
            
            # Save top config
            if sorted_results:
                best = sorted_results[0]
                best_config = self._create_config(best['params'], best['experiment_id'])
                
                config_file = self.results_dir / f"best_config_{metric}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(best_config, f)
    
    def _save_all_results(self):
        """Speichert alle Ergebnisse."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = self.results_dir / f"all_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # CSV
        successful = [r for r in self.all_results if r.get("success")]
        
        if successful:
            # Flatten params
            flat_results = []
            for r in successful:
                flat = {**r}
                flat.update({f"param_{k}": v for k, v in r['params'].items()})
                del flat['params']
                if 'action_distribution' in flat:
                    del flat['action_distribution']
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            csv_path = self.results_dir / f"all_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False, sep=";")
    
    def _comprehensive_analysis(self):
        """Umfassende Analyse aller Ergebnisse."""
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}\n")
        
        successful = [r for r in self.all_results if r.get("success")]
        
        if not successful:
            print("❌ Keine erfolgreichen Experimente!\n")
            return
        
        print(f"✅ {len(successful)}/{len(self.all_results)} erfolgreiche Experimente\n")
        
        # Best Configs pro Metrik
        self._print_best_configs(successful)
        
        # Parameter Impact Analysis
        self._analyze_parameter_impact(successful)
        
        # Visualizations
        self._create_visualizations(successful)
        
        # Summary Report
        self._create_summary_report(successful)
    
    def _print_best_configs(self, results: List[Dict]):
        """Gibt beste Konfigurationen aus."""
        metrics = {
            'composite_score': 'Overall Best',
            'roi': 'Highest ROI',
            'winrate': 'Highest Winrate',
            'sharpe_ratio': 'Best Risk-Adjusted'
        }
        
        for metric, title in metrics.items():
            sorted_results = sorted(
                results,
                key=lambda x: x.get(metric, -999),
                reverse=True
            )
            
            best = sorted_results[0]
            
            print(f"\n{'='*80}")
            print(f"🏆 {title.upper()}")
            print(f"{'='*80}")
            print(f"Experiment ID:      {best['experiment_id']}")
            print(f"{metric.upper()}:       {best[metric]:.4f}")
            print(f"ROI:                {best['roi']*100:.2f}%")
            print(f"Winrate:            {best['winrate']*100:.2f}%")
            print(f"Sharpe Ratio:       {best['sharpe_ratio']:.2f}")
            print(f"Max Drawdown:       {best['max_drawdown']:.2f}€")
            print(f"Profit Factor:      {best['profit_factor']:.2f}")
            print(f"\nPARAMETERS:")
            for key, value in best['params'].items():
                print(f"   {key}: {value}")
    
    def _analyze_parameter_impact(self, results: List[Dict]):
        """Analysiert Parameter-Einfluss."""
        print(f"\n{'='*80}")
        print(f"📈 PARAMETER IMPACT ANALYSIS")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(results)
        
        # Extrahiere Parameter
        params_df = pd.DataFrame([r['params'] for r in results])
        
        # Korrelationen mit ROI
        print("Correlation with ROI:")
        for col in params_df.columns:
            if params_df[col].dtype in [np.float64, np.int64]:
                corr = params_df[col].corr(df['roi'])
                print(f"   {col:30} {corr:>8.3f}")
        
        print()
    
    def _create_visualizations(self, results: List[Dict]):
        """Erstellt Visualisierungen."""
        print(f"\n📊 Creating Visualizations...\n")
        
        df = pd.DataFrame(results)
        
        # Performance Evolution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Parameter Search Performance', fontsize=16, fontweight='bold')
        
        # ROI over experiments
        axes[0, 0].plot(range(len(df)), df['roi'] * 100, marker='o', alpha=0.6)
        axes[0, 0].set_title('ROI over Experiments')
        axes[0, 0].set_xlabel('Experiment')
        axes[0, 0].set_ylabel('ROI (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Winrate vs ROI
        axes[0, 1].scatter(df['winrate'] * 100, df['roi'] * 100, alpha=0.6, s=100)
        axes[0, 1].set_title('Winrate vs ROI')
        axes[0, 1].set_xlabel('Winrate (%)')
        axes[0, 1].set_ylabel('ROI (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe Ratio distribution
        axes[1, 0].hist(df['sharpe_ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Sharpe Ratio Distribution')
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max Drawdown vs ROI
        axes[1, 1].scatter(df['max_drawdown'], df['roi'] * 100, alpha=0.6, s=100)
        axes[1, 1].set_title('Max Drawdown vs ROI')
        axes[1, 1].set_xlabel('Max Drawdown (€)')
        axes[1, 1].set_ylabel('ROI (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / "performance_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}\n")
    
    def _create_summary_report(self, results: List[Dict]):
        """Erstellt Summary Report."""
        report_path = self.results_dir / "summary_report.txt"
        
        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("PARAMETER SEARCH SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Experiments: {len(self.all_results)}\n")
            f.write(f"Successful:        {len(results)}\n")
            f.write(f"Failed:            {len(self.all_results) - len(results)}\n\n")
            
            df = pd.DataFrame(results)
            
            f.write("="*80 + "\n")
            f.write("PERFORMANCE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            metrics = ['roi', 'winrate', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
            
            for metric in metrics:
                f.write(f"{metric.upper()}:\n")
                f.write(f"   Mean:   {df[metric].mean():.4f}\n")
                f.write(f"   Std:    {df[metric].std():.4f}\n")
                f.write(f"   Min:    {df[metric].min():.4f}\n")
                f.write(f"   Max:    {df[metric].max():.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATIONS\n")
            f.write("="*80 + "\n\n")
            
            # Best Overall
            best = df.loc[df['composite_score'].idxmax()]
            f.write("BEST OVERALL (Composite Score):\n")
            f.write(f"   Experiment ID: {best['experiment_id']}\n")
            f.write(f"   Composite Score: {best['composite_score']:.4f}\n")
            f.write(f"   ROI: {best['roi']*100:.2f}%\n")
            f.write(f"   Winrate: {best['winrate']*100:.2f}%\n")
            f.write(f"   Sharpe: {best['sharpe_ratio']:.2f}\n\n")
            
            f.write("Parameters:\n")
            for key, value in best['params'].items():
                f.write(f"   {key}: {value}\n")
        
        print(f"📄 Summary Report: {report_path}\n")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Parameter Search")
    parser.add_argument("--config", required=True, help="Base config file")
    parser.add_argument("--season", type=int, default=2024, help="Target season")
    parser.add_argument("--mode", choices=['quick', 'balanced', 'comprehensive'], 
                        default='balanced', help="Search mode")
    parser.add_argument("--max-experiments", type=int, help="Max experiments")
    parser.add_argument("--sampling", choices=['random', 'latin_hypercube', 'grid'],
                        default='random', help="Sampling method")
    
    args = parser.parse_args()
    
    search = AdvancedParameterSearch(args.config, args.season)
    search.run_search(
        mode=args.mode,
        max_experiments=args.max_experiments,
        sampling=args.sampling
    )