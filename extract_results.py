"""
Training Results Extractor
==========================
Extrahiert und konsolidiert alle Trainings-Ergebnisse für einfache Analyse
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np


class ResultsExtractor:
    """Extrahiert Ergebnisse aus Training-Runs"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.absolute()
        
        self.base_dir = base_dir
        self.runs_dir = base_dir / 'runs'
        self.meta_results_dir = base_dir / 'meta_combined_results'
        
    def extract_single_run(self, run_dir: Path):
        """
        Extrahiert Daten aus einem einzelnen Run
        
        Returns:
            dict mit allen wichtigen Metriken
        """
        results = {
            'run_id': run_dir.name,
            'run_path': str(run_dir),
            'timestamp': None,
            'config': {},
            'training_metrics': {},
            'evaluation_metrics': {},
            'files': {}
        }
        
        # 1. Config laden
        config_file = run_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                results['config'] = yaml.safe_load(f)
        
        # 2. Results Directory
        results_dir = run_dir / 'results'
        
        if results_dir.exists():
            # Excel File
            excel_files = list(results_dir.glob('evaluation_*.xlsx'))
            
            if excel_files:
                excel_file = excel_files[0]
                results['files']['excel'] = excel_file.name
                
                # Parse timestamp from filename
                parts = excel_file.stem.split('_')
                if len(parts) >= 3:
                    results['timestamp'] = '_'.join(parts[2:])
                
                # Read Excel
                try:
                    df_gamedays = pd.read_excel(
                        excel_file, 
                        sheet_name='Gameday_Summaries'
                    )
                    
                    # Final Metrics
                    if len(df_gamedays) > 0:
                        last_row = df_gamedays.iloc[-1]
                        
                        results['evaluation_metrics'] = {
                            'final_roi': float(last_row.get('CumulativeROI', 0)),
                            'final_winrate': float(last_row.get('CumulativeWinrate', 0)),
                            'total_bets': int(last_row.get('CumulativeBets', 0)),
                            'total_wins': int(last_row.get('CumulativeWins', 0)),
                            'total_profit': float(last_row.get('CumulativeProfit', 0)),
                            'final_balance': float(last_row.get('CumulativeBalance', 0)),
                            'gamedays_played': len(df_gamedays)
                        }
                        
                        # Calculate additional metrics
                        profits = df_gamedays['TotalProfit'].values
                        
                        # Sharpe Ratio
                        if len(profits) > 1 and profits.std() > 0:
                            sharpe = profits.mean() / profits.std()
                        else:
                            sharpe = 0
                        
                        # Max Drawdown
                        cumulative = df_gamedays['CumulativeProfit'].values
                        running_max = np.maximum.accumulate(cumulative)
                        drawdowns = running_max - cumulative
                        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
                        
                        results['evaluation_metrics']['sharpe_ratio'] = float(sharpe)
                        results['evaluation_metrics']['max_drawdown'] = float(max_dd)
                
                except Exception as e:
                    print(f"⚠️  Fehler beim Lesen von {excel_file.name}: {e}")
            
            # CSV Files
            csv_files = list(results_dir.glob('*.csv'))
            results['files']['csv_files'] = [f.name for f in csv_files]
        
        # 3. Training Log
        log_file = run_dir / 'logs' / 'training.log'
        
        if log_file.exists():
            results['files']['log'] = log_file.name
            
            # Parse log for training metrics
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # Extract key info
                if 'GLOBAL PRETRAINING' in log_content:
                    results['training_metrics']['has_global_training'] = True
                
                if 'FINE-TUNING' in log_content:
                    results['training_metrics']['has_finetuning'] = True
                
                if 'INCREMENTAL LEARNING' in log_content:
                    results['training_metrics']['has_incremental'] = True
                
            except Exception as e:
                print(f"⚠️  Fehler beim Lesen von {log_file.name}: {e}")
        
        # 4. Model Files
        models_dir = run_dir / 'models'
        
        if models_dir.exists():
            model_files = list(models_dir.glob('*.zip'))
            results['files']['models'] = [f.name for f in model_files]
        
        return results
    
    def extract_all_runs(self):
        """
        Extrahiert Daten aus allen Runs
        
        Returns:
            list of dicts
        """
        if not self.runs_dir.exists():
            print(f"❌ Runs directory nicht gefunden: {self.runs_dir}")
            return []
        
        all_runs = []
        
        for run_dir in self.runs_dir.glob('run_*'):
            if not run_dir.is_dir():
                continue
            
            print(f"📁 Extrahiere: {run_dir.name}")
            
            try:
                run_data = self.extract_single_run(run_dir)
                all_runs.append(run_data)
            except Exception as e:
                print(f"⚠️  Fehler bei {run_dir.name}: {e}")
        
        return all_runs
    
    def extract_meta_results(self):
        """
        Extrahiert Meta-Test Ergebnisse
        
        Returns:
            list of meta test runs
        """
        if not self.meta_results_dir.exists():
            print(f"❌ Meta results directory nicht gefunden")
            return []
        
        meta_runs = []
        
        for meta_dir in self.meta_results_dir.glob('meta_*'):
            if not meta_dir.is_dir():
                continue
            
            print(f"📊 Extrahiere Meta-Test: {meta_dir.name}")
            
            # Load summary
            summary_file = meta_dir / 'meta_experiment_summary.json'
            
            if not summary_file.exists():
                continue
            
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    experiments = json.load(f)
                
                # Calculate aggregated metrics
                successful = [
                    e for e in experiments 
                    if e.get('success') and e.get('roi') is not None
                ]
                
                if successful:
                    meta_run = {
                        'meta_id': meta_dir.name,
                        'total_experiments': len(experiments),
                        'successful': len(successful),
                        'failed': len(experiments) - len(successful),
                        'avg_roi': np.mean([e['roi'] for e in successful]),
                        'std_roi': np.std([e['roi'] for e in successful]),
                        'avg_winrate': np.mean([e['winrate'] for e in successful]),
                        'avg_sharpe': np.mean([
                            e['sharpe_ratio'] for e in successful
                        ]),
                        'best_roi': max([e['roi'] for e in successful]),
                        'best_winrate': max([e['winrate'] for e in successful]),
                        'experiments': experiments
                    }
                    
                    meta_runs.append(meta_run)
            
            except Exception as e:
                print(f"⚠️  Fehler bei {meta_dir.name}: {e}")
        
        return meta_runs
    
    def create_summary_report(self, output_file: str = 'analysis_summary.txt'):
        """
        Erstellt zusammenfassenden Report
        
        Args:
            output_file: Output-Datei
        """
        output_path = self.base_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BETTING AGENT - TRAINING RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. Single Runs
            f.write("\n" + "="*80 + "\n")
            f.write("SINGLE TRAINING RUNS\n")
            f.write("="*80 + "\n\n")
            
            runs = self.extract_all_runs()
            
            if not runs:
                f.write("❌ Keine Runs gefunden\n\n")
            else:
                for run in runs:
                    f.write(f"\n{'─'*80}\n")
                    f.write(f"Run: {run['run_id']}\n")
                    f.write(f"{'─'*80}\n\n")
                    
                    # Config
                    if run['config']:
                        env = run['config'].get('environment', {})
                        training = run['config'].get('training', {})
                        
                        f.write("CONFIGURATION:\n")
                        f.write(f"  Target Season:        {training.get('target_season', 'N/A')}\n")
                        f.write(f"  Confidence Threshold: {env.get('confidence_threshold', 'N/A')}\n")
                        f.write(f"  Min Edge:             {env.get('min_edge_required', 'N/A')}\n")
                        f.write(f"  Max Bet Rate:         {env.get('max_bet_rate', 'N/A')}\n")
                        f.write(f"  Use Kelly:            {env.get('use_kelly_criterion', 'N/A')}\n")
                        f.write(f"  Kelly Fraction:       {env.get('kelly_fraction', 'N/A')}\n")
                        f.write(f"  Reward Shaping:       {env.get('reward_shaping', 'N/A')}\n")
                        f.write("\n")
                    
                    # Evaluation Metrics
                    if run['evaluation_metrics']:
                        metrics = run['evaluation_metrics']
                        
                        f.write("EVALUATION RESULTS:\n")
                        f.write(f"  Final ROI:            {metrics.get('final_roi', 0)*100:>8.2f}%\n")
                        f.write(f"  Final Winrate:        {metrics.get('final_winrate', 0)*100:>8.2f}%\n")
                        f.write(f"  Total Bets:           {metrics.get('total_bets', 0):>8}\n")
                        f.write(f"  Total Wins:           {metrics.get('total_wins', 0):>8}\n")
                        f.write(f"  Total Profit:         {metrics.get('total_profit', 0):>8.2f}€\n")
                        f.write(f"  Final Balance:        {metrics.get('final_balance', 0):>8.2f}€\n")
                        f.write(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>8.2f}\n")
                        f.write(f"  Max Drawdown:         {metrics.get('max_drawdown', 0):>8.2f}€\n")
                        f.write(f"  Gamedays Played:      {metrics.get('gamedays_played', 0):>8}\n")
                        f.write("\n")
                    
                    # Files
                    if run['files']:
                        f.write("FILES:\n")
                        
                        if run['files'].get('excel'):
                            f.write(f"  Excel:   {run['files']['excel']}\n")
                        
                        if run['files'].get('models'):
                            f.write(f"  Models:  {', '.join(run['files']['models'])}\n")
                        
                        if run['files'].get('csv_files'):
                            f.write(f"  CSVs:    {len(run['files']['csv_files'])} files\n")
                        
                        f.write("\n")
            
            # 2. Meta Tests
            f.write("\n" + "="*80 + "\n")
            f.write("META-TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            meta_runs = self.extract_meta_results()
            
            if not meta_runs:
                f.write("❌ Keine Meta-Tests gefunden\n\n")
            else:
                for meta in meta_runs:
                    f.write(f"\n{'─'*80}\n")
                    f.write(f"Meta-Test: {meta['meta_id']}\n")
                    f.write(f"{'─'*80}\n\n")
                    
                    f.write("OVERVIEW:\n")
                    f.write(f"  Total Experiments:    {meta['total_experiments']:>8}\n")
                    f.write(f"  Successful:           {meta['successful']:>8}\n")
                    f.write(f"  Failed:               {meta['failed']:>8}\n")
                    f.write(f"  Success Rate:         {meta['successful']/meta['total_experiments']*100:>8.2f}%\n")
                    f.write("\n")
                    
                    f.write("AGGREGATED METRICS:\n")
                    f.write(f"  Average ROI:          {meta['avg_roi']*100:>8.2f}% (±{meta['std_roi']*100:.2f}%)\n")
                    f.write(f"  Average Winrate:      {meta['avg_winrate']*100:>8.2f}%\n")
                    f.write(f"  Average Sharpe:       {meta['avg_sharpe']:>8.2f}\n")
                    f.write(f"  Best ROI:             {meta['best_roi']*100:>8.2f}%\n")
                    f.write(f"  Best Winrate:         {meta['best_winrate']*100:>8.2f}%\n")
                    f.write("\n")
                    
                    # Top 5 Experiments
                    successful = [
                        e for e in meta['experiments']
                        if e.get('success') and e.get('roi') is not None
                    ]
                    
                    if successful:
                        top5 = sorted(
                            successful, 
                            key=lambda x: x['roi'], 
                            reverse=True
                        )[:5]
                        
                        f.write("TOP 5 EXPERIMENTS BY ROI:\n")
                        
                        for i, exp in enumerate(top5, 1):
                            f.write(f"\n  #{i}:\n")
                            f.write(f"    ROI:        {exp['roi']*100:>8.2f}%\n")
                            f.write(f"    Winrate:    {exp['winrate']*100:>8.2f}%\n")
                            f.write(f"    Confidence: {exp['confidence']:>8.2f}\n")
                            f.write(f"    Min Edge:   {exp['min_edge']:>8.2f}\n")
                            f.write(f"    Max Bet:    {exp['max_bet_rate']*100:>8.2f}%\n")
                            
                            if exp.get('use_kelly'):
                                f.write(f"    Kelly:      {exp['kelly_fraction']:>8.2f}\n")
                        
                        f.write("\n")
        
        print(f"\n✅ Summary report saved: {output_path}")
        return output_path
    
    def export_to_json(self, output_file: str = 'results_export.json'):
        """
        Exportiert alle Daten als JSON
        
        Args:
            output_file: Output-Datei
        """
        output_path = self.base_dir / output_file
        
        data = {
            'generated': datetime.now().isoformat(),
            'single_runs': self.extract_all_runs(),
            'meta_tests': self.extract_meta_results()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON export saved: {output_path}")
        return output_path
    
    def export_to_csv(self, output_file: str = 'results_export.csv'):
        """
        Exportiert Run-Metriken als CSV
        
        Args:
            output_file: Output-Datei
        """
        output_path = self.base_dir / output_file
        
        runs = self.extract_all_runs()
        
        if not runs:
            print("❌ Keine Runs zum Exportieren")
            return None
        
        # Flatten data
        rows = []
        
        for run in runs:
            row = {
                'run_id': run['run_id'],
                'timestamp': run.get('timestamp', ''),
            }
            
            # Config
            if run['config']:
                env = run['config'].get('environment', {})
                training = run['config'].get('training', {})
                
                row['target_season'] = training.get('target_season', '')
                row['confidence_threshold'] = env.get('confidence_threshold', '')
                row['min_edge'] = env.get('min_edge_required', '')
                row['max_bet_rate'] = env.get('max_bet_rate', '')
                row['use_kelly'] = env.get('use_kelly_criterion', '')
                row['kelly_fraction'] = env.get('kelly_fraction', '')
                row['reward_shaping'] = env.get('reward_shaping', '')
            
            # Metrics
            if run['evaluation_metrics']:
                metrics = run['evaluation_metrics']
                
                row['final_roi'] = metrics.get('final_roi', 0)
                row['final_winrate'] = metrics.get('final_winrate', 0)
                row['total_bets'] = metrics.get('total_bets', 0)
                row['total_wins'] = metrics.get('total_wins', 0)
                row['total_profit'] = metrics.get('total_profit', 0)
                row['final_balance'] = metrics.get('final_balance', 0)
                row['sharpe_ratio'] = metrics.get('sharpe_ratio', 0)
                row['max_drawdown'] = metrics.get('max_drawdown', 0)
                row['gamedays_played'] = metrics.get('gamedays_played', 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, sep=';')
        
        print(f"✅ CSV export saved: {output_path}")
        return output_path


# =====================================================================
# CLI INTERFACE
# =====================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extrahiert Trainings-Ergebnisse für Analyse'
    )
    
    parser.add_argument(
        '--format',
        choices=['txt', 'json', 'csv', 'all'],
        default='all',
        help='Export-Format'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output-Verzeichnis'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ResultsExtractor()
    
    # Create output dir
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = extractor.base_dir
    
    print("\n" + "="*80)
    print("🔍 RESULTS EXTRACTOR")
    print("="*80 + "\n")
    
    # Export based on format
    if args.format in ['txt', 'all']:
        print("\n📄 Creating text summary...")
        extractor.create_summary_report(
            str(output_dir / 'analysis_summary.txt')
        )
    
    if args.format in ['json', 'all']:
        print("\n📋 Exporting to JSON...")
        extractor.export_to_json(
            str(output_dir / 'results_export.json')
        )
    
    if args.format in ['csv', 'all']:
        print("\n📊 Exporting to CSV...")
        extractor.export_to_csv(
            str(output_dir / 'results_export.csv')
        )
    
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80 + "\n")
    
    # Quick stats
    runs = extractor.extract_all_runs()
    meta_tests = extractor.extract_meta_results()
    
    print(f"📊 Summary:")
    print(f"   Single Runs:  {len(runs)}")
    print(f"   Meta Tests:   {len(meta_tests)}")
    
    if runs:
        successful_runs = [
            r for r in runs 
            if r.get('evaluation_metrics', {}).get('final_roi') is not None
        ]
        
        if successful_runs:
            avg_roi = np.mean([
                r['evaluation_metrics']['final_roi'] 
                for r in successful_runs
            ])
            
            print(f"\n   Average ROI:  {avg_roi*100:.2f}%")
    
    print()


if __name__ == '__main__':
    main()