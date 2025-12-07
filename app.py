"""
Flask Web Application for Betting Agent

"""

import os
import sys
import json
import yaml
import threading
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# =====================================================================
# SETUP PATHS
# =====================================================================
BASE_DIR = Path(__file__).parent.absolute()
SCRIPTS_DIR = BASE_DIR / 'scripts'
CORE_DIR = BASE_DIR / 'core'

# Add to Python path
for path in [SCRIPTS_DIR, CORE_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# =====================================================================
# FLASK APP
# =====================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# =====================================================================
# GLOBAL VARIABLES
# =====================================================================

# Training status
training_status = {
    'running': False,
    'progress': 0,
    'message': 'Bereit',
    'phase': 'idle',
    'current_run': None,
    'log_output': ''
}

# Download status
download_status = {
    'running': False,
    'progress': 0,
    'message': 'Bereit zum Download',
    'phase': 'idle'
}

# Training process
training_process = None

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_config():
    """Load config.yaml"""
    config_path = BASE_DIR / 'config' / 'config.yaml'
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config):
    """Save config.yaml"""
    config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_training_runs():
    """Get list of training runs - searches in all run_* directories"""
    
    runs_base = Path('runs')
    
    if not runs_base.exists():
        print("❌ 'runs' directory not found")
        return []
    
    runs = []
    seen_ids = set()
    
    print(f"🔍 Scanning: {runs_base.absolute()}")
    
    # Search in all run_* directories
    for run_dir in runs_base.glob('run_*'):
        if not run_dir.is_dir():
            continue
        
        # Look for results subdirectory
        results_dir = run_dir / 'results'
        
        if not results_dir.exists():
            continue
        
        print(f"  📁 Checking: {run_dir.name}/results")
        
        # Look for Excel files
        for excel_file in results_dir.glob('evaluation_*.xlsx'):
            
            # Parse filename: evaluation_2025_20251115_164715.xlsx
            parts = excel_file.stem.split('_')
            if len(parts) >= 3:
                season = parts[1]
                timestamp = '_'.join(parts[2:])
                run_id = excel_file.stem
                
                # Skip duplicates
                if run_id in seen_ids:
                    continue
                seen_ids.add(run_id)
                
                # Check for corresponding CSV files
                csv_gameday = results_dir / f'gameday_results_{season}_{timestamp}.csv'
                csv_performance = results_dir / f'performance_{season}_{timestamp}.csv'
                
                print(f"    ✅ Found: {excel_file.name}")
                
                runs.append({
                    'id': run_id,
                    'season': season,
                    'timestamp': timestamp,
                    'run_dir': run_dir.name,
                    'excel_file': excel_file.name,
                    'csv_gameday': csv_gameday.name if csv_gameday.exists() else None,
                    'csv_performance': csv_performance.name if csv_performance.exists() else None,
                    'has_results': True,
                    'path': str(results_dir),
                    'full_path': str(excel_file)
                })
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    print(f"📊 Total runs found: {len(runs)}")
    
    return runs


def get_data_files():
    """Get list of data files"""
    config = load_config()
    
    # Raw data
    raw_dir = Path(config.get('paths', {}).get('raw_data_dir', 'data/raw'))
    raw_files = []
    
    if raw_dir.exists():
        for csv_file in sorted(raw_dir.glob('D1_*.csv')):
            # Extract season from filename: D1_2023.csv
            season = csv_file.stem.replace('D1_', '')
            raw_files.append({
                'season': season,
                'filename': csv_file.name,
                'size': csv_file.stat().st_size / 1024,  # KB
                'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # Processed data
    processed_dir = Path(config.get('paths', {}).get('data_dir', 'data/processed'))
    processed_files = []
    
    if processed_dir.exists():
        for csv_file in sorted(processed_dir.glob('Bundesliga_*.csv')):
            processed_files.append({
                'filename': csv_file.name,
                'size': csv_file.stat().st_size / 1024,  # KB
                'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    return {
        'raw': raw_files,
        'processed': processed_files
    }


def get_feature_data():
    """Get feature metadata from config"""
    config = load_config()
    features_config = config.get('features', {})
    
    categories = features_config.get('categories', [])
    
    # Enrich with examples and formulas
    feature_data = {}
    
    for category in categories:
        cat_name = category.get('name', '')
        features_list = []
        
        for feature_name in category.get('features', []):
            feature_info = {
                'name': feature_name,
                'description': get_feature_description(feature_name),
                'formula': get_feature_formula(feature_name),
                'example': get_feature_example(feature_name),
                'type': get_feature_type(feature_name),
                'source': 'Calculated' if 'L4' in feature_name or 'Season' in feature_name else 'Raw Data'
            }
            features_list.append(feature_info)
        
        feature_data[cat_name] = {
            'icon': category.get('icon', '📊'),
            'description': category.get('description', ''),
            'features': features_list
        }
    
    return feature_data


def get_feature_description(name):
    """Get feature description"""
    descriptions = {
        'Goals_Home_L4': 'Durchschnittliche Tore des Heimteams in den letzten 4 Heimspielen',
        'Goals_Away_L4': 'Durchschnittliche Tore des Auswärtsteams in den letzten 4 Auswärtsspielen',
        'Conceded_Home_L4': 'Durchschnittlich kassierte Tore (Heimteam) in letzten 4 Spielen',
        'Conceded_Away_L4': 'Durchschnittlich kassierte Tore (Auswärtsteam) in letzten 4 Spielen',
        'Expected_Goals': 'Erwartete Tore basierend auf Offensive/Defensive beider Teams',
        'Points_Home_L4': 'Durchschnittliche Punkte (Heimteam) letzte 4 Spiele',
        'Points_Away_L4': 'Durchschnittliche Punkte (Auswärtsteam) letzte 4 Spiele',
        'ImpProb_H': 'Implizite Wahrscheinlichkeit für Heimsieg (aus Buchmacher-Quoten)',
        'ImpProb_D': 'Implizite Wahrscheinlichkeit für Unentschieden',
        'ImpProb_A': 'Implizite Wahrscheinlichkeit für Auswärtssieg',
        'MaxQuote_H': 'Beste verfügbare Quote für Heimsieg',
        'MaxQuote_A': 'Beste verfügbare Quote für Auswärtssieg',
        'Value_Home': 'Value-Indikator: Model-Prob minus Implied-Prob (Heimsieg)',
        'Value_Away': 'Value-Indikator: Model-Prob minus Implied-Prob (Auswärtssieg)',
        'BookieMargin': 'Buchmacher-Marge (Overround)',
        'Draw_Risk': 'Binärer Indikator für erhöhtes Unentschieden-Risiko',
        'Momentum_Home': 'Kurzfristige Form relativ zu Saison-Durchschnitt (Heim)',
        'Momentum_Away': 'Kurzfristige Form relativ zu Saison-Durchschnitt (Auswärts)',
    }
    return descriptions.get(name, 'Keine Beschreibung verfügbar')


def get_feature_formula(name):
    """Get feature formula"""
    formulas = {
        'Goals_Home_L4': 'mean(goals_scored in last 4 home games)',
        'Expected_Goals': '(Goals_Home_L4 + Conceded_Away_L4 + Goals_Away_L4 + Conceded_Home_L4) / 2',
        'ImpProb_H': '1 / MaxQuote_H (normalized)',
        'Value_Home': 'Model_Probability_H - Implied_Probability_H',
        'BookieMargin': '1 - 1/((1/Quote_H) + (1/Quote_D) + (1/Quote_A))',
        'Points_Diff_L4': 'Points_Home_L4 - Points_Away_L4',
        'Form_Ratio_L4': 'Points_Home_L4 / (Points_Away_L4 + 0.01)',
        'Momentum_Home': 'Points_Home_L4 - (Points_Home_Season / Gameday)',
    }
    return formulas.get(name, '')


def get_feature_example(name):
    """Get feature example"""
    examples = {
        'Goals_Home_L4': 'Bayern: 2.5 Tore/Spiel<br>→ Starke Offensive',
        'ImpProb_H': 'Quote 1.50 → 67% Wahrscheinlichkeit',
        'Value_Home': 'Model: 72%, Bookie: 67%<br>→ Edge: +5%',
        'Expected_Goals': '(2.5 + 1.2 + 1.8 + 1.5) / 2 = 3.5 Tore',
    }
    return examples.get(name, '')


def get_feature_type(name):
    """Get feature type"""
    if 'L4' in name:
        return 'L4'
    elif 'Season' in name:
        return 'Season'
    elif any(x in name for x in ['Value', 'Momentum', 'Form', 'Draw_Risk', 'Expected']):
        return 'Calculated'
    else:
        return 'Raw'


# =====================================================================
# ROUTES: PAGES
# =====================================================================

@app.route('/')
def index():
    """Home page"""
    config = load_config()
    data_files = get_data_files()
    runs = get_training_runs()
    
    return render_template(
        'index.html',
        data_available=len(data_files['processed']) > 0,
        training_running=training_status['running'],
        feature_count=config.get('features', {}).get('feature_count', 42),
        runs=runs
    )


@app.route('/data')
def data_management():
    """Data management page"""
    data_files = get_data_files()
    
    return render_template(
        'data_management.html',
        raw_data=data_files['raw'],
        processed_data=data_files['processed']
    )


@app.route('/model')
def model_info():
    """Model info page with tabs"""
    config = load_config()
    feature_data = get_feature_data()
    
    # Profile type analysis
    env = config.get('environment', {})
    confidence = env.get('confidence_threshold', 0.60)
    min_edge = env.get('min_edge_required', 0.05)
    max_bet_rate = env.get('max_bet_rate', 0.30)
    
    # Determine profile type
    if confidence >= 0.65 and min_edge >= 0.05:
        profile_type = {
            'name': 'Conservative',
            'emoji': '🛡️',
            'class': 'conservative',
            'description': 'Fokus auf Sicherheit und hohe Winrate'
        }
    elif confidence <= 0.58 and min_edge <= 0.03:
        profile_type = {
            'name': 'Aggressive',
            'emoji': '⚡',
            'class': 'aggressive',
            'description': 'Fokus auf Value und viele Wetten'
        }
    else:
        profile_type = {
            'name': 'Balanced',
            'emoji': '⚖️',
            'class': 'balanced',
            'description': 'Ausgewogen zwischen Risiko und Sicherheit'
        }
    
    return render_template(
        'model_info.html',
        config=config,
        feature_data=feature_data,
        features=sum([len(cat['features']) for cat in feature_data.values()]),
        feature_count=config.get('features', {}).get('feature_count', 42),
        params={
            'confidence_threshold': confidence,
            'min_edge': min_edge,
            'max_bet_rate': max_bet_rate,
            'bet_amount': env.get('bet_amount', 10.0),
            'max_bet_amount': env.get('max_bet_amount', 30.0),
            'use_kelly': env.get('use_kelly_criterion', True),
            'kelly_fraction': env.get('kelly_fraction', 0.25),
            'reward_shaping': env.get('reward_shaping', 'conservative'),
            'no_bet_multiplier': env.get('no_bet_reward_multiplier', 0.5),
            'draw_penalty': env.get('draw_penalty_multiplier', 1.5),
            'net_arch': config.get('model', {}).get('policy_kwargs', {}).get('net_arch', [256, 256, 128])
        },
        profile_type=profile_type,
        expected_time={
            'global': '~45-60 Min',
            'finetune': '~5-10 Min',
            'incremental': '~1-2 Min/Spieltag',
            'total': '~60-90 Min'
        }
    )



@app.route('/features')
def feature_overview():
    """Feature overview page"""
    config = load_config()
    feature_data = get_feature_data()
    
    # ✅ FIX: Berechne total features count
    total_features = sum([len(cat['features']) for cat in feature_data.values()])
    
    return render_template(
        'feature_overview.html',
        feature_data=feature_data,
        features=total_features,  # ✅ Jetzt als Integer
        feature_count=total_features,  # ✅ Auch als feature_count
        feature_categories=list(feature_data.keys())
    )

@app.route('/training')
def training_config():
    """Training configuration selection page"""
    # Load main config (config.yaml)
    config = load_config()
    
    # Check data availability
    data_files = get_data_files()
    data_available = len(data_files['processed']) > 0
    
    # Get runs
    runs = get_training_runs()
    
    # Extract nested values for easier template access
    training_cfg = config.get('training', {})
    global_seasons = training_cfg.get('global_seasons', {})
    environment_cfg = config.get('environment', {})
    model_cfg = config.get('model', {})
    dqn_cfg = model_cfg.get('dqn', {})
    
    return render_template(
        'training_config.html',
        config=config,
        data_available=data_available,
        runs=runs,
        # Training settings (from config.yaml)
        target_season=training_cfg.get('target_season', 2024),
        fine_tune_gamedays=training_cfg.get('fine_tune_gamedays', 9),
        global_start=global_seasons.get('start', 2016),
        global_end=global_seasons.get('end', 2023),
        global_timesteps=training_cfg.get('global_timesteps', 1000000),
        finetune_timesteps=training_cfg.get('finetune_timesteps', 100000),
        incremental_learning=training_cfg.get('incremental_learning', True),
        # Environment settings (from config.yaml)
        confidence_threshold=environment_cfg.get('confidence_threshold', 0.60),
        min_edge=environment_cfg.get('min_edge_required', 0.05),
        max_bet_rate=int(environment_cfg.get('max_bet_rate', 0.30) * 100),  # Convert to percentage
        bet_amount=environment_cfg.get('bet_amount', 10.0),
        max_bet_amount=environment_cfg.get('max_bet_amount', 30.0),
        use_kelly=environment_cfg.get('use_kelly_criterion', True),
        kelly_fraction=environment_cfg.get('kelly_fraction', 0.25),
        # Model settings (from config.yaml)
        learning_rate=dqn_cfg.get('learning_rate', 0.0001),
        gamma=dqn_cfg.get('gamma', 0.95)
    )


@app.route('/status')
def status():
    """Training status page"""
    return render_template('status.html')


@app.route('/results')
def results():
    """Results page"""
    runs = get_training_runs()
    
    return render_template(
        'results.html',
        runs=runs
    )

# =====================================================================
# META TESTING ROUTES
# =====================================================================


@app.route('/meta_testing')
def meta_testing():
    """Meta Testing page"""
    return render_template('meta_testing.html')


# ✅ Global Meta-Test Status mit Log
meta_test_status = {
    'running': False,
    'progress': 0,
    'completed': 0,
    'failed': 0,
    'total': 0,
    'current_test': None,
    'status': 'idle',
    'run_id': None,
    'log_output': '',  # ✅ NEU
    'log_file': None   # ✅ NEU
}

@app.route('/api/start_meta_test', methods=['POST'])
def api_start_meta_test():
    """Start meta testing mit LIVE Log-Tracking"""
    global meta_test_status
    
    if meta_test_status['running']:
        return jsonify({'error': 'Tests laufen bereits'}), 400
    
    try:
        data = request.get_json()
        
        mode = data.get('mode', 'balanced')
        max_experiments = data.get('max_experiments')
        custom_config = data.get('config_path')
        
        # Generate run ID
        run_id = f"meta_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # ✅ Determine log file path (wird vom MetaTestRunner erstellt)
        log_file = BASE_DIR / 'meta_combined_results' / f"meta_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "meta_test.log"
        
        # Update status
        meta_test_status = {
            'running': True,
            'progress': 0,
            'completed': 0,
            'failed': 0,
            'total': 0,
            'current_test': 'Initialisiere...',
            'status': 'running',
            'run_id': run_id,
            'log_output': '',
            'log_file': str(log_file)  # ✅ Store log file path
        }
        
        # Start in background thread
        def run_meta_test():
            global meta_test_status
            
            try:
                cmd = [
                    sys.executable,
                    str(BASE_DIR / 'meta_combined_test.py'),
                    '--mode', mode
                ]
                
                if max_experiments:
                    cmd.extend(['--max-experiments', str(max_experiments)])
                
                if custom_config:
                    cmd.extend(['--config', custom_config])
                
                # ✅ Run with LIVE LOG CAPTURE
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # ✅ Monitor output LINE BY LINE
                log_lines = []
                for line in process.stdout:
                    log_lines.append(line)
                    
                    # ✅ Update log output (last 200 lines)
                    meta_test_status['log_output'] = ''.join(log_lines[-200:])
                    
                    # Parse progress from line
                    if '🧪 EXPERIMENT' in line:
                        try:
                            import re
                            match = re.search(r'EXPERIMENT (\d+)', line)
                            if match:
                                current = int(match.group(1))
                                meta_test_status['completed'] = current - 1
                                meta_test_status['current_test'] = f"Experiment {current}"
                        except:
                            pass
                    
                    # Parse total from "Total Combinations:"
                    if 'Total Combinations:' in line:
                        try:
                            import re
                            match = re.search(r'Total Combinations:\s*([\d,]+)', line)
                            if match:
                                total = int(match.group(1).replace(',', ''))
                                meta_test_status['total'] = total
                        except:
                            pass
                    
                    # Update progress percentage
                    if meta_test_status['total'] > 0 and meta_test_status['completed'] > 0:
                        meta_test_status['progress'] = (
                            meta_test_status['completed'] / meta_test_status['total'] * 100
                        )
                
                process.wait()
                
                # ✅ Final log read (falls etwas fehlt)
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            full_log = f.read()
                            meta_test_status['log_output'] = full_log[-50000:]  # Last 50KB
                    except:
                        pass
                
                if process.returncode == 0:
                    meta_test_status['status'] = 'completed'
                    meta_test_status['progress'] = 100
                    meta_test_status['current_test'] = 'Abgeschlossen'
                else:
                    meta_test_status['status'] = 'failed'
                
                meta_test_status['running'] = False
                
            except Exception as e:
                meta_test_status['running'] = False
                meta_test_status['status'] = 'failed'
                meta_test_status['current_test'] = f'Fehler: {str(e)}'
                meta_test_status['log_output'] += f'\n\n❌ ERROR: {str(e)}'
                print(f"Meta test error: {e}")
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=run_meta_test, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Meta-Testing gestartet',
            'run_id': run_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/meta_test_status', methods=['GET'])
def api_meta_test_status():
    """Get meta test status mit Log"""
    return jsonify(meta_test_status)


@app.route('/api/meta_test_results', methods=['GET'])
def api_meta_test_results():
    """Get all meta test results"""
    try:
        meta_results_dir = BASE_DIR / 'meta_combined_results'
        
        if not meta_results_dir.exists():
            return jsonify({'runs': []})
        
        runs = []
        
        # Scan for experiment summary files
        for summary_file in meta_results_dir.glob('meta_experiment_summary.json'):
            run_dir = summary_file.parent
            
            with open(summary_file, 'r') as f:
                experiments = json.load(f)
            
            if not experiments:
                continue
            
            # Extract metadata
            first_exp = experiments[0]
            
            # Calculate metrics
            successful = [e for e in experiments if e.get('success') and e.get('roi') is not None]
            
            avg_roi = np.mean([e['roi'] for e in successful]) if successful else 0
            avg_winrate = np.mean([e['winrate'] for e in successful]) if successful else 0
            
            runs.append({
                'run_id': run_dir.name,
                'mode': 'balanced',  # Could be extracted from filename
                'season': first_exp.get('target_season', 'Unknown'),
                'experiments': len(experiments),
                'started_at': datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                'status': 'completed' if len(successful) > 0 else 'failed',
                'metrics': {
                    'roi': avg_roi,
                    'winrate': avg_winrate
                }
            })
        
        # Sort by date
        runs.sort(key=lambda x: x['started_at'], reverse=True)
        
        return jsonify({'runs': runs})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/meta_test_details/<run_id>')
def meta_test_details(run_id):
    """Detailed view of meta test results"""
    try:
        meta_results_dir = BASE_DIR / 'meta_combined_results' / run_id
        
        if not meta_results_dir.exists():
            return "Run nicht gefunden", 404
        
        # Load summary
        summary_file = meta_results_dir / 'meta_experiment_summary.json'
        
        if not summary_file.exists():
            return "Keine Ergebnisse gefunden", 404
        
        with open(summary_file, 'r') as f:
            experiments = json.load(f)
        
        # Find CSV/Excel files
        csv_files = list(meta_results_dir.glob('all_results_*.csv'))
        excel_files = list(meta_results_dir.glob('analysis_*.xlsx'))
        
        return render_template(
            'meta_test_details.html',
            run_id=run_id,
            experiments=experiments,
            csv_files=[f.name for f in csv_files],
            excel_files=[f.name for f in excel_files]
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Fehler: {str(e)}", 500


@app.route('/api/meta_test_log', methods=['GET'])
def api_meta_test_log():
    """Read log file directly - Fallback method"""
    global meta_test_status
    
    log_file_path = meta_test_status.get('log_file')
    
    if not log_file_path:
        return jsonify({'log': 'Kein Log-File gefunden'}), 404
    
    log_file = Path(log_file_path)
    
    if not log_file.exists():
        return jsonify({'log': 'Log-File existiert noch nicht...'}), 404
    
    try:
        # Read last 100KB of log
        with open(log_file, 'r', encoding='utf-8') as f:
            f.seek(0, 2)  # Go to end
            size = f.tell()
            
            # Read last 100KB
            start = max(0, size - 100000)
            f.seek(start)
            log_content = f.read()
        
        return jsonify({
            'log': log_content,
            'size': size
        })
        
    except Exception as e:
        return jsonify({
            'log': f'Fehler beim Lesen: {str(e)}'
        }), 500
# =====================================================================
# API: DOWNLOAD
# =====================================================================


@app.route('/api/download_status', methods=['GET'])
def api_download_status():
    """Get current download status"""
    return jsonify(download_status)


@app.route('/api/download_data', methods=['POST'])
def api_download_data():
    """
    Download Bundesliga data from football-data.co.uk
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        seasons = data.get('seasons', [])
        
        if not seasons:
            return jsonify({'error': 'Keine Saisons ausgewählt'}), 400
        
        # Load config
        config = load_config()
        raw_dir = Path(config.get('paths', {}).get('raw_data_dir', 'data/raw'))
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ FIX: Importiere aus CORE Verzeichnis
        import sys
        core_path = str(CORE_DIR)
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        
        from core.data_downloader import download_all_seasons
        
        # Update global status
        global download_status
        download_status = {
            'running': True,
            'progress': 0,
            'message': 'Download wird gestartet...',
            'phase': 'downloading'
        }
        
        # Start download in background thread
        def download_task():
            global download_status
            try:
                seasons_str = [str(s) for s in seasons]
                total = len(seasons_str)
                success_count = 0
                
                for i, season in enumerate(seasons_str):
                    download_status['message'] = f'Lade Saison {season}...'
                    download_status['progress'] = int((i / total) * 100)
                    
                    # Download single season
                    result = download_all_seasons(raw_dir, [season])
                    if result > 0:
                        success_count += 1
                
                download_status['progress'] = 100
                download_status['message'] = f'✅ {success_count}/{total} Saisons heruntergeladen'
                download_status['running'] = False
                
            except Exception as e:
                download_status['running'] = False
                download_status['message'] = f'Fehler: {str(e)}'
                download_status['progress'] = 0
                print(f"Download Error: {e}")
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=download_task, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Download gestartet für {len(seasons)} Saisons'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess_data', methods=['POST'])
def api_preprocess_data():
    """
    Preprocess downloaded data
    """
    try:
        config = load_config()
        raw_dir = Path(config.get('paths', {}).get('raw_data_dir', 'data/raw'))
        processed_dir = Path(config.get('paths', {}).get('data_dir', 'data/processed'))
        
        # ✅ FIX: Importiere aus CORE Verzeichnis
        import sys
        core_path = str(CORE_DIR)
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        
        from core.data_downloader import preprocess_data
        
        # Update status
        global download_status
        download_status = {
            'running': True,
            'progress': 50,
            'message': 'Preprocessing wird gestartet...',
            'phase': 'preprocessing'
        }
        
        # Start in background
        def preprocess_task():
            global download_status
            try:
                download_status['message'] = 'Verarbeite Daten...'
                
                output_file = preprocess_data(raw_dir, processed_dir)
                
                download_status['progress'] = 100
                download_status['message'] = f'✅ Daten verarbeitet: {output_file.name}'
                download_status['running'] = False
                
            except Exception as e:
                download_status['running'] = False
                download_status['message'] = f'Fehler: {str(e)}'
                download_status['progress'] = 0
                print(f"Preprocessing Error: {e}")
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=preprocess_task, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Preprocessing gestartet'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =====================================================================
# API: TRAINING
# =====================================================================

@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    """Start training with config from form"""
    global training_status, training_process
    
    if training_status['running']:
        return jsonify({'error': 'Training läuft bereits'}), 400
    
    try:
        # Parse form data
        data = request.form.to_dict()
        
        # Update config
        config = load_config()
        
        # Update training settings
        config['training']['target_season'] = int(data.get('target_season', 2024))
        config['training']['global_seasons']['start'] = int(data.get('global_start', 2016))
        config['training']['global_seasons']['end'] = int(data.get('global_end', 2023))
        config['training']['fine_tune_gamedays'] = int(data.get('fine_tune_gamedays', 9))
        config['training']['global_timesteps'] = int(data.get('global_timesteps', 1000000))
        config['training']['finetune_timesteps'] = int(data.get('finetune_timesteps', 100000))
        config['training']['incremental_learning'] = data.get('incremental_learning') == 'on'
        
        # Update environment settings
        config['environment']['confidence_threshold'] = float(data.get('confidence_threshold', 0.60))
        config['environment']['min_edge_required'] = float(data.get('min_edge', 0.05))
        config['environment']['max_bet_rate'] = float(data.get('max_bet_rate', 30)) / 100
        config['environment']['bet_amount'] = float(data.get('bet_amount', 10.0))
        config['environment']['max_bet_amount'] = float(data.get('max_bet_amount', 30.0))
        config['environment']['use_kelly_criterion'] = data.get('use_kelly') == 'on'
        config['environment']['kelly_fraction'] = float(data.get('kelly_fraction', 0.25))
        
        # Update model settings
        config['model']['dqn']['learning_rate'] = float(data.get('learning_rate', 0.0001))
        config['model']['dqn']['gamma'] = float(data.get('gamma', 0.95))
        
        # Save updated config
        save_config(config)
        
        # Update training status
        training_status = {
            'running': True,
            'progress': 0,
            'message': 'Training wird gestartet...',
            'phase': 'starting',
            'current_run': f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'log_output': ''
        }
        
        # Start training in subprocess
        def run_training():
            global training_process, training_status
            
            try:
                # Run adaptive training script
                cmd = [
                    sys.executable,
                    str(CORE_DIR / 'adaptive_training.py'),
                    str(BASE_DIR / 'config' / 'config.yaml')
                ]
                
                training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Read output
                for line in training_process.stdout:
                    training_status['log_output'] += line
                    
                    # Update status based on output
                    if 'PHASE 1' in line or 'GLOBAL PRETRAINING' in line:
                        training_status['phase'] = 'global_training'
                        training_status['progress'] = 20
                        training_status['message'] = 'Global Training läuft...'
                    elif 'PHASE 2' in line or 'FINE-TUNING' in line:
                        training_status['phase'] = 'fine_tuning'
                        training_status['progress'] = 60
                        training_status['message'] = 'Fine-Tuning läuft...'
                    elif 'PHASE 3' in line or 'DEPLOYMENT' in line:
                        training_status['phase'] = 'evaluation'
                        training_status['progress'] = 80
                        training_status['message'] = 'Evaluation läuft...'
                    elif 'ABGESCHLOSSEN' in line or 'completed' in line.lower():
                        training_status['phase'] = 'completed'
                        training_status['progress'] = 100
                        training_status['message'] = '✅ Training abgeschlossen!'
                
                training_process.wait()
                
                if training_process.returncode == 0:
                    training_status['phase'] = 'completed'
                    training_status['progress'] = 100
                    training_status['message'] = '✅ Training erfolgreich abgeschlossen!'
                else:
                    training_status['phase'] = 'error'
                    training_status['message'] = f'❌ Training fehlgeschlagen (Code: {training_process.returncode})'
                
                training_status['running'] = False
                
            except Exception as e:
                training_status['running'] = False
                training_status['phase'] = 'error'
                training_status['message'] = f'❌ Fehler: {str(e)}'
                training_status['log_output'] += f'\n\nERROR: {str(e)}'
        
        # Start thread
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return redirect(url_for('status'))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start_multi_agent_training', methods=['POST'])
def api_start_multi_agent_training():
    """Start multi-agent training"""
    global training_status, training_process
    
    if training_status['running']:
        return jsonify({'error': 'Training läuft bereits'}), 400
    
    try:
        # Parse form data
        data = request.form.to_dict()
        
        # Get selected agents
        agent_types = request.form.getlist('agents')
        
        if not agent_types:
            return jsonify({'error': 'Keine Agenten ausgewählt'}), 400
        
        # Update config
        config = load_config()
        
        # Enable multi-agent
        config['multi_agent']['enabled'] = True
        config['multi_agent']['agent_types'] = agent_types
        config['multi_agent']['ensemble_strategy'] = data.get('ensemble_strategy', 'voting')
        
        # Update training settings
        config['training']['target_season'] = int(data.get('target_season', 2024))
        config['training']['global_timesteps'] = int(data.get('global_timesteps', 1000000))
        
        # Save config
        save_config(config)
        
        # Update status
        training_status = {
            'running': True,
            'progress': 0,
            'message': 'Multi-Agent Training wird gestartet...',
            'phase': 'starting',
            'current_run': f"multi_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'log_output': ''
        }
        
        # Start training
        def run_multi_agent_training():
            global training_process, training_status
            
            try:
                cmd = [
                    sys.executable,
                    str(CORE_DIR / 'multi_agent_system.py'),
                    str(BASE_DIR / 'config.yaml')
                ]
                
                training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                for line in training_process.stdout:
                    training_status['log_output'] += line
                
                training_process.wait()
                
                if training_process.returncode == 0:
                    training_status['phase'] = 'completed'
                    training_status['progress'] = 100
                    training_status['message'] = '✅ Multi-Agent Training abgeschlossen!'
                else:
                    training_status['phase'] = 'error'
                    training_status['message'] = f'❌ Training fehlgeschlagen'
                
                training_status['running'] = False
                
            except Exception as e:
                training_status['running'] = False
                training_status['phase'] = 'error'
                training_status['message'] = f'❌ Fehler: {str(e)}'
        
        thread = threading.Thread(target=run_multi_agent_training, daemon=True)
        thread.start()
        
        return redirect(url_for('status'))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get current training status"""
    return jsonify(training_status)


# =====================================================================
# API: RESULTS
# =====================================================================

@app.route('/api/results/<run_id>', methods=['GET'])
def api_results(run_id):
    """Get results for a specific run"""
    try:
        # Search for Excel file in all run directories
        excel_file = None
        
        for run_dir in Path('runs').glob('run_*'):
            results_dir = run_dir / 'results'
            if results_dir.exists():
                candidate = results_dir / f"{run_id}.xlsx"
                if candidate.exists():
                    excel_file = candidate
                    break
        
        if not excel_file:
            return jsonify({
                'error': 'Results not found',
                'run_id': run_id
            }), 404
        
        print(f"✅ Found: {excel_file}")
        
        # Read Excel
        df_gamedays = pd.read_excel(excel_file, sheet_name='Gameday_Summaries')
        gameday_data = df_gamedays.to_dict('records')
        
        # Get config
        config = load_config()
        max_bet = config.get('environment', {}).get('max_bet_amount', 30.0)
        
        # Parse filename for CSV names
        parts = run_id.split('_')
        season = parts[1] if len(parts) > 1 else '2025'
        timestamp = '_'.join(parts[2:]) if len(parts) > 2 else ''
        
        # Check for CSV files
        results_dir = excel_file.parent
        csv_gameday = results_dir / f'gameday_results_{season}_{timestamp}.csv'
        csv_performance = results_dir / f'performance_{season}_{timestamp}.csv'
        
        return jsonify({
            'success': True,
            'gameday_data': gameday_data,
            'excel_file': excel_file.name,
            'max_bet_amount': max_bet,
            'csv_gameday': csv_gameday.name if csv_gameday.exists() else None,
            'csv_performance': csv_performance.name if csv_performance.exists() else None,
            'path': str(results_dir)
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<run_id>/<filename>')
def download_results(run_id, filename):
    """Download results file (Excel or CSV)"""
    try:
        # Search for file in all run directories
        file_path = None
        
        for run_dir in Path('runs').glob('run_*'):
            results_dir = run_dir / 'results'
            if results_dir.exists():
                candidate = results_dir / filename
                if candidate.exists():
                    file_path = candidate
                    break
        
        if not file_path:
            return f"File not found: {filename}", 404
        
        print(f"✅ Sending: {file_path}")
        
        # Determine MIME type
        if filename.endswith('.xlsx'):
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif filename.endswith('.csv'):
            mimetype = 'text/csv'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"Error: {str(e)}", 500


# =====================================================================
# API: FEATURE IMPORTANCE
# =====================================================================

@app.route('/api/feature_importance', methods=['GET'])
def api_feature_importance():
    """
    Get feature importance from trained model.
    ✅ FIX: Sucht in allen run_* directories nach neuestem Model
    """
    try:
        # ✅ Suche in allen run_* directories
        runs_dir = BASE_DIR / 'runs'
        
        if not runs_dir.exists():
            return jsonify({
                'error': 'Keine Trainings-Runs gefunden',
                'hint': 'Bitte zuerst ein Training durchführen'
            }), 404
        
        # Sammle alle Model-Dateien
        all_models = []
        
        for run_dir in runs_dir.glob('run_*'):
            models_dir = run_dir / 'models'
            
            if not models_dir.exists():
                continue
            
            for model_file in models_dir.glob('*.zip'):
                all_models.append({
                    'path': model_file,
                    'mtime': model_file.stat().st_mtime,
                    'run': run_dir.name,
                    'name': model_file.name
                })
        
        if not all_models:
            return jsonify({
                'error': 'Kein trainiertes Modell gefunden',
                'hint': 'Bitte zuerst ein Training durchführen',
                'searched_in': str(runs_dir)
            }), 404
        
        # ✅ Sortiere nach Modify-Zeit (neuestes zuerst)
        all_models.sort(key=lambda x: x['mtime'], reverse=True)
        latest = all_models[0]
        
        print(f"🔍 Feature Importance - Lade Model:")
        print(f"   Run:  {latest['run']}")
        print(f"   File: {latest['name']}")
        print(f"   Path: {latest['path']}")
        
        # ✅ Lade Model und extrahiere Weights
        try:
            from stable_baselines3 import DQN
            import torch
            
            model = DQN.load(str(latest['path']))
            
            # Get Q-Network
            q_net = model.policy.q_net
            
            # Get first layer weights
            first_layer = q_net[0]  # Erster Linear Layer
            weights = first_layer.weight.detach().cpu().numpy()
            
            # Calculate importance (mean absolute weight per feature)
            importance = np.abs(weights).mean(axis=0)
            
            # Normalize to sum to 1
            importance = importance / importance.sum()
            
            # ✅ Get feature names from config
            config = load_config()
            feature_names = []
            
            for category in config.get('features', {}).get('categories', []):
                feature_names.extend(category.get('features', []))
            
            # ✅ Match features to importance scores
            importance_dict = {}
            
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[name] = float(importance[i])
            
            # ✅ Add metadata
            response = {
                'success': True,
                'importance': importance_dict,
                'metadata': {
                    'model_run': latest['run'],
                    'model_name': latest['name'],
                    'features_count': len(importance_dict),
                    'model_modified': datetime.fromtimestamp(latest['mtime']).isoformat(),
                    'total_models_found': len(all_models)
                }
            }
            
            print(f"✅ Feature Importance berechnet: {len(importance_dict)} Features")
            
            return jsonify(response)
            
        except ImportError as e:
            return jsonify({
                'error': 'Stable-Baselines3 nicht installiert',
                'details': str(e)
            }), 500
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Fehler beim Laden des Modells: {str(e)}',
                'model_path': str(latest['path'])
            }), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Unerwarteter Fehler: {str(e)}'
        }), 500


# =====================================================================
# API: PROFILE MANAGEMENT
# =====================================================================

@app.route('/load-profile/<profile_name>', methods=['GET'])
def load_profile(profile_name):
    """Load a betting profile"""
    try:
        # ✅ FIX: Profile sind in config/, nicht profiles/
        profiles_dir = BASE_DIR / 'config'
        profile_file = profiles_dir / f"{profile_name}.yaml"
        
        print(f"🔍 DEBUG: Lade Profil von: {profile_file}")
        print(f"🔍 DEBUG: Existiert? {profile_file.exists()}")
        
        if not profile_file.exists():
            print(f"❌ DEBUG: Datei nicht gefunden!")
            return jsonify({'error': f'Profile not found: {profile_file}'}), 404
        
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = yaml.safe_load(f)
        
        print(f"✅ DEBUG: Profil geladen: {profile.get('name', 'Unknown')}")
        print(f"🔍 DEBUG: Environment: {profile.get('environment', {})}")
        
        return jsonify({
            'success': True,
            'profile': profile,
            'environment': profile.get('environment', {})
        })
        
    except Exception as e:
        print(f"❌ DEBUG: Fehler beim Laden: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
# =====================================================================
# RUN APP
# =====================================================================

if __name__ == '__main__':
    import socket
    
    print("="*80)
    print("🤖 BETTING AGENT WEB APP")
    print("="*80)
    print(f"📂 Base Dir: {BASE_DIR}")
    print(f"🔧 Scripts Dir: {SCRIPTS_DIR}")
    print(f"🧠 Core Dir: {CORE_DIR}")
    
    # ✅ Find available port
    def find_free_port(start_port=5000, max_attempts=10):
        """Find next available port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                # Try to bind to port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return None
    
    # Find free port
    port = find_free_port(5000)
    
    if port is None:
        print("❌ Kein freier Port gefunden!")
        sys.exit(1)
    
    print(f"🌐 Starting Flask on http://localhost:{port}")
    
    if port != 5000:
        print(f"⚠️  Port 5000 war belegt, nutze Port {port} stattdessen")
    
    print("="*80 + "\n")
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            use_reloader=False  # Disable reloader to prevent double-execution
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server gestoppt")
    except Exception as e:
        print(f"\n❌ Fehler beim Starten: {e}")