"""
Feature Engineering & Selection System
======================================
Testet verschiedene Feature-Kombinationen und Zeitfenster
für optimale RL-Performance.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Setup Paths
BASE_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BASE_DIR))


class FeatureEngineer:
    """Erstellt Features mit verschiedenen Zeitfenstern."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_groups = {}
        
    def create_rolling_features(self, windows=[3, 4, 5, 6, 10]):
        """Erstellt Features für verschiedene Zeitfenster."""
        print("\n🔄 Erstelle Rolling Features...")
        
        df = self.df.copy()
        df = df.sort_values(['Season', 'Date']).reset_index(drop=True)
        
        for window in windows:
            print(f"   Window: L{window}")
            
            # Initialize columns
            for prefix in ['Home', 'Away']:
                for metric in ['Points', 'Goals', 'Conceded', 'Shots', 
                               'ShotsTarget', 'Corners', 'Fouls']:
                    df[f'{metric}_{prefix}_L{window}'] = np.nan
            
            # Calculate per season and team
            for season in df['Season'].unique():
                season_mask = df['Season'] == season
                teams = pd.unique(
                    df.loc[season_mask, ['HomeTeam', 'AwayTeam']].values.ravel()
                )
                
                for team in teams:
                    # Home games
                    home_mask = season_mask & (df['HomeTeam'] == team)
                    home_games = df[home_mask].copy()
                    
                    if len(home_games) >= window:
                        home_games['Points_tmp'] = home_games.apply(
                            lambda r: 3 if r['FTR']=='H' else (1 if r['FTR']=='D' else 0),
                            axis=1
                        )
                        
                        df.loc[home_games.index, f'Points_Home_L{window}'] = \
                            home_games['Points_tmp'].shift(1).rolling(window, min_periods=1).mean()
                        
                        df.loc[home_games.index, f'Goals_Home_L{window}'] = \
                            home_games['FTHG'].shift(1).rolling(window, min_periods=1).mean()
                        
                        df.loc[home_games.index, f'Conceded_Home_L{window}'] = \
                            home_games['FTAG'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'HS' in home_games.columns:
                            df.loc[home_games.index, f'Shots_Home_L{window}'] = \
                                home_games['HS'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'HST' in home_games.columns:
                            df.loc[home_games.index, f'ShotsTarget_Home_L{window}'] = \
                                home_games['HST'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'HC' in home_games.columns:
                            df.loc[home_games.index, f'Corners_Home_L{window}'] = \
                                home_games['HC'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'HF' in home_games.columns:
                            df.loc[home_games.index, f'Fouls_Home_L{window}'] = \
                                home_games['HF'].shift(1).rolling(window, min_periods=1).mean()
                    
                    # Away games
                    away_mask = season_mask & (df['AwayTeam'] == team)
                    away_games = df[away_mask].copy()
                    
                    if len(away_games) >= window:
                        away_games['Points_tmp'] = away_games.apply(
                            lambda r: 3 if r['FTR']=='A' else (1 if r['FTR']=='D' else 0),
                            axis=1
                        )
                        
                        df.loc[away_games.index, f'Points_Away_L{window}'] = \
                            away_games['Points_tmp'].shift(1).rolling(window, min_periods=1).mean()
                        
                        df.loc[away_games.index, f'Goals_Away_L{window}'] = \
                            away_games['FTAG'].shift(1).rolling(window, min_periods=1).mean()
                        
                        df.loc[away_games.index, f'Conceded_Away_L{window}'] = \
                            away_games['FTHG'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'AS' in away_games.columns:
                            df.loc[away_games.index, f'Shots_Away_L{window}'] = \
                                away_games['AS'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'AST' in away_games.columns:
                            df.loc[away_games.index, f'ShotsTarget_Away_L{window}'] = \
                                away_games['AST'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'AC' in away_games.columns:
                            df.loc[away_games.index, f'Corners_Away_L{window}'] = \
                                away_games['AC'].shift(1).rolling(window, min_periods=1).mean()
                        
                        if 'AF' in away_games.columns:
                            df.loc[away_games.index, f'Fouls_Away_L{window}'] = \
                                away_games['AF'].shift(1).rolling(window, min_periods=1).mean()
            
            # Fill NaN
            for col in df.columns:
                if f'_L{window}' in col:
                    if 'Points' in col:
                        df[col].fillna(1.0, inplace=True)
                    elif 'Goals' in col or 'Conceded' in col:
                        df[col].fillna(1.5, inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
        
        self.df = df
        return df
    
    def create_advanced_features(self):
        """Erstellt abgeleitete Features."""
        print("\n🎯 Erstelle Advanced Features...")
        
        df = self.df.copy()
        
        # Find all window sizes
        windows = []
        for col in df.columns:
            if '_L' in col:
                try:
                    window = int(col.split('_L')[1])
                    if window not in windows:
                        windows.append(window)
                except:
                    pass
        
        windows = sorted(windows)
        
        for window in windows:
            print(f"   Window: L{window}")
            
            # Points Difference
            if f'Points_Home_L{window}' in df.columns:
                df[f'Points_Diff_L{window}'] = (
                    df[f'Points_Home_L{window}'] - df[f'Points_Away_L{window}']
                )
            
            # Goal Intensity
            if all(col in df.columns for col in [
                f'Goals_Home_L{window}', f'Conceded_Away_L{window}',
                f'Goals_Away_L{window}', f'Conceded_Home_L{window}'
            ]):
                df[f'Goal_Intensity_L{window}'] = (
                    df[f'Goals_Home_L{window}'] + df[f'Conceded_Away_L{window}'] +
                    df[f'Goals_Away_L{window}'] + df[f'Conceded_Home_L{window}']
                ) / 2
            
            # Form Difference
            if f'Points_Home_L{window}' in df.columns:
                df[f'Form_Ratio_L{window}'] = (
                    df[f'Points_Home_L{window}'] / 
                    (df[f'Points_Away_L{window}'] + 0.01)
                )
            
            # Attack vs Defense
            if all(col in df.columns for col in [
                f'Goals_Home_L{window}', f'Conceded_Home_L{window}'
            ]):
                df[f'Attack_Defense_Home_L{window}'] = (
                    df[f'Goals_Home_L{window}'] / 
                    (df[f'Conceded_Home_L{window}'] + 0.01)
                )
                
                df[f'Attack_Defense_Away_L{window}'] = (
                    df[f'Goals_Away_L{window}'] / 
                    (df[f'Conceded_Away_L{window}'] + 0.01)
                )
            
            # Shot Efficiency
            if all(col in df.columns for col in [
                f'Goals_Home_L{window}', f'Shots_Home_L{window}'
            ]):
                df[f'Shot_Efficiency_Home_L{window}'] = (
                    df[f'Goals_Home_L{window}'] / 
                    (df[f'Shots_Home_L{window}'] + 0.01)
                )
                
                df[f'Shot_Efficiency_Away_L{window}'] = (
                    df[f'Goals_Away_L{window}'] / 
                    (df[f'Shots_Away_L{window}'] + 0.01)
                )
        
        # Replace inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        self.df = df
        return df
    
    def group_features(self):
        """Gruppiert Features nach Typ."""
        print("\n📊 Gruppiere Features...")
        
        groups = {
            'odds': [],
            'form': [],
            'goals': [],
            'shots': [],
            'discipline': [],
            'advanced': []
        }
        
        for col in self.df.columns:
            if any(x in col for x in ['Quote', 'Prob', 'OU_', 'Bookie']):
                groups['odds'].append(col)
            elif 'Points' in col:
                groups['form'].append(col)
            elif any(x in col for x in ['Goals', 'Conceded', 'Goal_Intensity']):
                groups['goals'].append(col)
            elif any(x in col for x in ['Shots', 'Shot_Efficiency']):
                groups['shots'].append(col)
            elif any(x in col for x in ['Fouls', 'Yellow', 'Red', 'Corners']):
                groups['discipline'].append(col)
            elif any(x in col for x in ['Diff', 'Ratio', 'Attack_Defense', 'Value', 'Momentum']):
                groups['advanced'].append(col)
        
        self.feature_groups = groups
        
        print("\nFeature Groups:")
        for group, features in groups.items():
            print(f"  {group:12}: {len(features):>3} features")
        
        return groups


class FeatureSelector:
    """Testet und selektiert beste Features."""
    
    def __init__(self, df: pd.DataFrame, target_col='FTR'):
        self.df = df.copy()
        self.target_col = target_col
        self.results = []
        
    def prepare_data(self, feature_cols):
        """Bereitet Daten für Training vor."""
        # Filter valid games
        df = self.df[self.df['Gameday'] >= 4].copy()
        
        # Prepare X and y
        X = df[feature_cols].copy()
        
        # Encode target
        if self.target_col == 'FTR':
            y = df[self.target_col].map({'H': 0, 'D': 1, 'A': 2})
        else:
            y = df[self.target_col]
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, mask.sum()
    
    def test_feature_importance(self, feature_cols, name="features"):
        """Testet Feature Importance mit Random Forest."""
        print(f"\n🧪 Teste: {name} ({len(feature_cols)} features)")
        
        try:
            X, y, n_samples = self.prepare_data(feature_cols)
            
            if n_samples < 100:
                print(f"   ⚠️  Zu wenig Samples: {n_samples}")
                return None
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation
            scores = cross_val_score(
                rf, X, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            # Fit for feature importance
            rf.fit(X, y)
            
            # Get importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                'name': name,
                'n_features': len(feature_cols),
                'n_samples': n_samples,
                'accuracy_mean': scores.mean(),
                'accuracy_std': scores.std(),
                'top_10_features': importance.head(10)['feature'].tolist(),
                'top_10_importance': importance.head(10)['importance'].tolist(),
            }
            
            self.results.append(result)
            
            print(f"   ✅ Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
            print(f"   Top 3: {', '.join(importance.head(3)['feature'].tolist())}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def test_window_sizes(self, base_metrics=['Points', 'Goals', 'Conceded']):
        """Testet verschiedene Zeitfenster."""
        print("\n" + "="*80)
        print("🔍 TESTE ZEITFENSTER")
        print("="*80)
        
        windows = [3, 4, 5, 6, 10]
        
        for window in windows:
            feature_cols = []
            
            for metric in base_metrics:
                for side in ['Home', 'Away']:
                    col = f'{metric}_{side}_L{window}'
                    if col in self.df.columns:
                        feature_cols.append(col)
            
            if feature_cols:
                self.test_feature_importance(
                    feature_cols, 
                    name=f"Window_L{window}"
                )
    
    def test_feature_groups(self, feature_groups):
        """Testet Feature-Gruppen einzeln."""
        print("\n" + "="*80)
        print("🔍 TESTE FEATURE-GRUPPEN")
        print("="*80)
        
        for group_name, features in feature_groups.items():
            if features:
                available = [f for f in features if f in self.df.columns]
                if available:
                    self.test_feature_importance(
                        available, 
                        name=f"Group_{group_name}"
                    )
    
    def test_combinations(self, feature_groups, max_groups=3):
        """Testet Kombinationen von Feature-Gruppen."""
        print("\n" + "="*80)
        print("🔍 TESTE FEATURE-KOMBINATIONEN")
        print("="*80)
        
        group_names = list(feature_groups.keys())
        
        for r in range(2, min(max_groups + 1, len(group_names) + 1)):
            for combo in combinations(group_names, r):
                feature_cols = []
                for group in combo:
                    feature_cols.extend([
                        f for f in feature_groups[group] 
                        if f in self.df.columns
                    ])
                
                if feature_cols:
                    combo_name = "+".join(combo)
                    self.test_feature_importance(
                        feature_cols, 
                        name=f"Combo_{combo_name}"
                    )
    
    def get_best_features(self, top_n=50):
        """Ermittelt beste Features basierend auf Tests."""
        print("\n" + "="*80)
        print("🏆 BESTE FEATURES")
        print("="*80)
        
        if not self.results:
            print("❌ Keine Ergebnisse vorhanden!")
            return []
        
        # Sortiere nach Accuracy
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('accuracy_mean', ascending=False)
        
        print("\nTop 5 Feature-Sets:")
        print(results_df[['name', 'n_features', 'accuracy_mean', 'accuracy_std']].head())
        
        # Sammle Features nach Importance
        feature_scores = {}
        
        for result in self.results:
            for feat, imp in zip(
                result['top_10_features'], 
                result['top_10_importance']
            ):
                if feat not in feature_scores:
                    feature_scores[feat] = []
                feature_scores[feat].append(imp * result['accuracy_mean'])
        
        # Durchschnittliche Scores
        feature_ranking = pd.DataFrame([
            {'feature': feat, 'score': np.mean(scores)}
            for feat, scores in feature_scores.items()
        ]).sort_values('score', ascending=False)
        
        print(f"\nTop {top_n} Features (nach aggregiertem Score):")
        print(feature_ranking.head(top_n))
        
        return feature_ranking.head(top_n)['feature'].tolist()
    
    def plot_results(self, output_dir=Path("results/feature_analysis")):
        """Erstellt Visualisierungen."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Plot 1: Accuracy Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        results_sorted = results_df.sort_values('accuracy_mean', ascending=True)
        
        ax.barh(
            range(len(results_sorted)), 
            results_sorted['accuracy_mean'],
            xerr=results_sorted['accuracy_std'],
            capsize=3
        )
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['name'])
        ax.set_xlabel('Accuracy')
        ax.set_title('Feature Set Performance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_comparison.png", dpi=150)
        plt.close()
        
        print(f"\n📊 Plot gespeichert: {output_dir / 'feature_comparison.png'}")


def main():
    """Main Feature Engineering Pipeline."""
    
    # Load data
    data_dir = Path("data/processed")
    csv_files = sorted(data_dir.glob("Bundesliga_*.csv"))
    
    if not csv_files:
        print("❌ Keine Daten gefunden!")
        return
    
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"📂 Lade Daten: {latest_csv.name}")
    
    df = pd.read_csv(latest_csv, sep=";")
    
    print(f"   Spiele: {len(df):,}")
    print(f"   Zeitraum: {df['Season'].min()} - {df['Season'].max()}")
    
    # ========== FEATURE ENGINEERING ==========
    engineer = FeatureEngineer(df)
    
    # Erstelle Features für verschiedene Zeitfenster
    df = engineer.create_rolling_features(windows=[3, 4, 5, 6, 10])
    df = engineer.create_advanced_features()
    feature_groups = engineer.group_features()
    
    # ========== FEATURE SELECTION ==========
    selector = FeatureSelector(df)
    
    # Test 1: Zeitfenster
    selector.test_window_sizes()
    
    # Test 2: Feature-Gruppen
    selector.test_feature_groups(feature_groups)
    
    # Test 3: Kombinationen
    selector.test_combinations(feature_groups, max_groups=3)
    
    # ========== RESULTS ==========
    best_features = selector.get_best_features(top_n=50)
    
    # Speichere Ergebnisse
    output_dir = Path("results/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best features
    with open(output_dir / "best_features.txt", "w") as f:
        f.write("# Best Features for RL Agent\n")
        f.write("# Generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
        for i, feat in enumerate(best_features, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"\n💾 Liste gespeichert: {output_dir / 'best_features.txt'}")
    
    # Plot
    selector.plot_results(output_dir)
    
    # Save detailed results
    results_df = pd.DataFrame(selector.results)
    results_df.to_csv(output_dir / "feature_test_results.csv", index=False)
    print(f"💾 Detaillierte Ergebnisse: {output_dir / 'feature_test_results.csv'}")
    
    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING ABGESCHLOSSEN")
    print("="*80)


if __name__ == "__main__":
    main()