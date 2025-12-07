"""
Data Downloader & Preprocessor
✅ Optimiert für L4-L6 Features (basierend auf Feature Engineering)
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =====================================================================
# SEASON URLS
# =====================================================================
SEASON_URLS = {
    "2016": "https://www.football-data.co.uk/mmz4281/1617/D1.csv",
    "2017": "https://www.football-data.co.uk/mmz4281/1718/D1.csv",
    "2018": "https://www.football-data.co.uk/mmz4281/1819/D1.csv",
    "2019": "https://www.football-data.co.uk/mmz4281/1920/D1.csv",
    "2020": "https://www.football-data.co.uk/mmz4281/2021/D1.csv",
    "2021": "https://www.football-data.co.uk/mmz4281/2122/D1.csv",
    "2022": "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
    "2023": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "2024": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "2025": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}


# =====================================================================
# DOWNLOAD
# =====================================================================
def download_season(season: str, output_dir: Path) -> bool:
    """Lädt CSV für eine Saison herunter."""
    if season not in SEASON_URLS:
        print(f"❌ Saison {season} nicht verfügbar")
        return False
    
    url = SEASON_URLS[season]
    output_path = output_dir / f"D1_{season}.csv"
    
    try:
        print(f"⬇️  Lade Saison {season}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"   ✅ Gespeichert: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False


def download_all_seasons(output_dir: Path, seasons=None):
    """Lädt alle oder ausgewählte Saisons herunter."""
    output_dir.mkdir(exist_ok=True)
    
    if seasons is None:
        seasons = list(SEASON_URLS.keys())
    
    success_count = 0
    
    for season in seasons:
        if download_season(season, output_dir):
            success_count += 1
    
    print(f"\n✅ {success_count}/{len(seasons)} Saisons erfolgreich heruntergeladen\n")
    return success_count


# =====================================================================
# PREPROCESSING
# =====================================================================
def preprocess_data(input_dir: Path, output_dir: Path) -> Path:
    """Bereitet Daten auf mit optimierten Features."""
    print("\n" + "="*70)
    print("🔧 PREPROCESSING (OPTIMIERT)")
    print("="*70 + "\n")
    
    csv_files = sorted(input_dir.glob("D1_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"❌ Keine CSV-Dateien in {input_dir}")
    
    print(f"📂 Gefunden: {len(csv_files)} Saisons")
    
    # Lade und merge
    all_data = []
    
    for csv_file in csv_files:
        season = csv_file.stem.replace("D1_", "")
        print(f"   📄 Verarbeite Saison {season}...")
        
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
            df.insert(0, "Season", int(season))
            
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            
            all_data.append(df)
            
        except Exception as e:
            print(f"   ⚠️  Fehler bei Saison {season}: {e}")
    
    if not all_data:
        raise ValueError("❌ Keine Daten erfolgreich geladen!")
    
    print("\n🔗 Kombiniere Saisons...")
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    
    # Gameday
    print("📊 Berechne Spieltage...")
    combined["Gameday"] = 0
    
    for season in combined["Season"].unique():
        season_mask = combined["Season"] == season
        season_data = combined[season_mask]
        gamedays = (season_data.index - season_data.index.min()) // 9 + 1
        combined.loc[season_mask, "Gameday"] = gamedays
    
    combined.insert(0, "Index", range(1, len(combined) + 1))
    
    # ✅ OPTIMIERTES Feature Engineering
    print("🎯 Feature Engineering (OPTIMIERT)...")
    combined = compute_features(combined)
    
    print("✂️  Spalten-Selektion...")
    combined = select_relevant_columns(combined)
    
    # Speichern
    output_dir.mkdir(exist_ok=True)
    
    seasons_str = f"{combined['Season'].min()}-{combined['Season'].max()}"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"Bundesliga_{seasons_str}_{timestamp}_optimized.csv"
    
    combined.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*70}")
    print(f"💾 DATENSATZ GESPEICHERT")
    print(f"{'='*70}")
    print(f"Pfad:     {output_file}")
    print(f"Spiele:   {len(combined):,}")
    print(f"Features: {len(combined.columns)}")
    print(f"Zeitraum: {combined['Date'].min().date()} bis {combined['Date'].max().date()}")
    print(f"{'='*70}\n")
    
    return output_file


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle Features."""
    print("   🔢 Konvertiere Quoten...")
    df = convert_odds_columns(df)
    
    print("   💰 Berechne beste Quoten...")
    df = compute_best_odds(df)
    
    print("   📊 Berechne Team-Statistiken...")
    df = compute_team_statistics(df)
    
    print("   🔥 Berechne aggregierte Features...")
    df = compute_aggregated_features(df)
    
    return df


def convert_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert Quote-Spalten."""
    odds_cols = [c for c in df.columns if any(
        k in c for k in ["BbMx", "BbAv", "B365", "Max", "Avg", "P>", "P<", ">2.5", "<2.5"]
    )]
    
    for col in odds_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def compute_best_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet beste Quoten und Probabilities."""
    # Over/Under
    over_cols = [c for c in df.columns if ">2.5" in c or "Over" in c]
    under_cols = [c for c in df.columns if "<2.5" in c or "Under" in c]
    
    if over_cols:
        df["OU_Over"] = df[over_cols].max(axis=1, skipna=True)
    else:
        df["OU_Over"] = np.nan
    
    if under_cols:
        df["OU_Under"] = df[under_cols].max(axis=1, skipna=True)
    else:
        df["OU_Under"] = np.nan
    
    df["OU_Spread"] = df["OU_Over"] - df["OU_Under"]
    
    # 1X2
    for side, cols in {
        "H": ["B365H", "BbMxH", "BbAvH", "MaxH", "AvgH", "PSH"],
        "D": ["B365D", "BbMxD", "BbAvD", "MaxD", "AvgD", "PSD"],
        "A": ["B365A", "BbMxA", "BbAvA", "MaxA", "AvgA", "PSA"],
    }.items():
        present = [c for c in cols if c in df.columns]
        if present:
            df[f"MaxQuote_{side}"] = df[present].max(axis=1, skipna=True)
        else:
            df[f"MaxQuote_{side}"] = np.nan
    
    # Implied Probabilities
    df["ImpProb_H"] = 1 / df["MaxQuote_H"]
    df["ImpProb_D"] = 1 / df["MaxQuote_D"]
    df["ImpProb_A"] = 1 / df["MaxQuote_A"]
    
    prob_sum = df[["ImpProb_H", "ImpProb_D", "ImpProb_A"]].sum(axis=1)
    df["ImpProb_H"] = df["ImpProb_H"] / prob_sum
    df["ImpProb_D"] = df["ImpProb_D"] / prob_sum
    df["ImpProb_A"] = df["ImpProb_A"] / prob_sum
    
    df["BookieMargin"] = 1 - 1 / ((1/df["MaxQuote_H"]) + (1/df["MaxQuote_D"]) + (1/df["MaxQuote_A"]))
    
    return df


def compute_team_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ NUR NOCH L4-FEATURES (statt L3-L10)
    """
    
    def get_points(res, is_home):
        if res == "H": return 3 if is_home else 0
        if res == "A": return 0 if is_home else 3
        if res == "D": return 1
        return 0
    
    # ✅ NUR NOCH WINDOW 4
    window = 4
    
    print(f"      Erstelle Features für Window: L{window}")
    
    # Initialize columns
    for prefix in ["Home", "Away"]:
        for suffix in ["Points", "Goals", "Conceded"]:
            df[f"{suffix}_{prefix}_L{window}"] = np.nan
        
        # Match Stats
        for feat in ["Shots", "ShotsTarget", "Fouls", "Corners", "Yellow", "Red"]:
            df[f"{feat}_{prefix}_L{window}"] = np.nan
    
    # Season Stats (bleiben)
    for prefix in ["Home", "Away"]:
        df[f"Points_{prefix}_Season"] = np.nan
        df[f"Goals_{prefix}_Season"] = np.nan
        df[f"Conceded_{prefix}_Season"] = np.nan
        df[f"GoalDiff_{prefix}_Season"] = np.nan
    
    # Calculate per season and team
    df = df.sort_values(["Season", "Date"]).reset_index(drop=True)
    
    for season in df["Season"].unique():
        season_mask = df["Season"] == season
        teams = pd.unique(df.loc[season_mask, ["HomeTeam", "AwayTeam"]].values.ravel())
        
        for team in teams:
            # HOME GAMES
            home_mask = season_mask & (df["HomeTeam"] == team)
            home_games = df[home_mask].copy()
            
            if len(home_games) > 0:
                home_games["Points"] = home_games.apply(
                    lambda row: get_points(row["FTR"], True), axis=1
                )
                
                # Core Stats
                df.loc[home_games.index, f'Points_Home_L{window}'] = \
                    home_games['Points'].shift(1).rolling(window, min_periods=1).mean()
                
                df.loc[home_games.index, f'Goals_Home_L{window}'] = \
                    home_games['FTHG'].shift(1).rolling(window, min_periods=1).mean()
                
                df.loc[home_games.index, f'Conceded_Home_L{window}'] = \
                    home_games['FTAG'].shift(1).rolling(window, min_periods=1).mean()
                
                # Match Stats
                if "HS" in home_games.columns:
                    df.loc[home_games.index, f'Shots_Home_L{window}'] = \
                        home_games['HS'].shift(1).rolling(window, min_periods=1).mean()
                
                if "HST" in home_games.columns:
                    df.loc[home_games.index, f'ShotsTarget_Home_L{window}'] = \
                        home_games['HST'].shift(1).rolling(window, min_periods=1).mean()
                
                if "HF" in home_games.columns:
                    df.loc[home_games.index, f'Fouls_Home_L{window}'] = \
                        home_games['HF'].shift(1).rolling(window, min_periods=1).mean()
                
                if "HC" in home_games.columns:
                    df.loc[home_games.index, f'Corners_Home_L{window}'] = \
                        home_games['HC'].shift(1).rolling(window, min_periods=1).mean()
                
                if "HY" in home_games.columns:
                    df.loc[home_games.index, f'Yellow_Home_L{window}'] = \
                        home_games['HY'].shift(1).rolling(window, min_periods=1).mean()
                
                if "HR" in home_games.columns:
                    df.loc[home_games.index, f'Red_Home_L{window}'] = \
                        home_games['HR'].shift(1).rolling(window, min_periods=1).mean()
                
                # Season Stats
                df.loc[home_games.index, 'Points_Home_Season'] = \
                    home_games['Points'].shift(1).cumsum()
                
                df.loc[home_games.index, 'Goals_Home_Season'] = \
                    home_games['FTHG'].shift(1).cumsum()
                
                df.loc[home_games.index, 'Conceded_Home_Season'] = \
                    home_games['FTAG'].shift(1).cumsum()
                
                df.loc[home_games.index, 'GoalDiff_Home_Season'] = \
                    df.loc[home_games.index, 'Goals_Home_Season'] - \
                    df.loc[home_games.index, 'Conceded_Home_Season']
            
            # AWAY GAMES (analog)
            away_mask = season_mask & (df["AwayTeam"] == team)
            away_games = df[away_mask].copy()
            
            if len(away_games) > 0:
                away_games["Points"] = away_games.apply(
                    lambda row: get_points(row["FTR"], False), axis=1
                )
                
                # Rolling Features
                df.loc[away_games.index, f'Points_Away_L{window}'] = \
                    away_games['Points'].shift(1).rolling(window, min_periods=1).mean()
                
                df.loc[away_games.index, f'Goals_Away_L{window}'] = \
                    away_games['FTAG'].shift(1).rolling(window, min_periods=1).mean()
                
                df.loc[away_games.index, f'Conceded_Away_L{window}'] = \
                    away_games['FTHG'].shift(1).rolling(window, min_periods=1).mean()
                
                # Match Stats
                if "AS" in away_games.columns:
                    df.loc[away_games.index, f'Shots_Away_L{window}'] = \
                        away_games['AS'].shift(1).rolling(window, min_periods=1).mean()
                
                if "AST" in away_games.columns:
                    df.loc[away_games.index, f'ShotsTarget_Away_L{window}'] = \
                        away_games['AST'].shift(1).rolling(window, min_periods=1).mean()
                
                if "AF" in away_games.columns:
                    df.loc[away_games.index, f'Fouls_Away_L{window}'] = \
                        away_games['AF'].shift(1).rolling(window, min_periods=1).mean()
                
                if "AC" in away_games.columns:
                    df.loc[away_games.index, f'Corners_Away_L{window}'] = \
                        away_games['AC'].shift(1).rolling(window, min_periods=1).mean()
                
                if "AY" in away_games.columns:
                    df.loc[away_games.index, f'Yellow_Away_L{window}'] = \
                        away_games['AY'].shift(1).rolling(window, min_periods=1).mean()
                
                if "AR" in away_games.columns:
                    df.loc[away_games.index, f'Red_Away_L{window}'] = \
                        away_games['AR'].shift(1).rolling(window, min_periods=1).mean()
                
                # Season Stats
                df.loc[away_games.index, 'Points_Away_Season'] = \
                    away_games['Points'].shift(1).cumsum()
                
                df.loc[away_games.index, 'Goals_Away_Season'] = \
                    away_games['FTAG'].shift(1).cumsum()
                
                df.loc[away_games.index, 'Conceded_Away_Season'] = \
                    away_games['FTHG'].shift(1).cumsum()
                
                df.loc[away_games.index, 'GoalDiff_Away_Season'] = \
                    df.loc[away_games.index, 'Goals_Away_Season'] - \
                    df.loc[away_games.index, 'Conceded_Away_Season']
    
    # Fill NaN
    df[f"Points_Home_L{window}"].fillna(1.0, inplace=True)
    df[f"Points_Away_L{window}"].fillna(1.0, inplace=True)
    df[f"Goals_Home_L{window}"].fillna(1.5, inplace=True)
    df[f"Goals_Away_L{window}"].fillna(1.5, inplace=True)
    df[f"Conceded_Home_L{window}"].fillna(1.5, inplace=True)
    df[f"Conceded_Away_L{window}"].fillna(1.5, inplace=True)
    
    for col in df.columns:
        if "_Season" in col:
            df[col].fillna(0, inplace=True)
    
    return df


def compute_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ NUR NOCH L4-AGGREGIERTE FEATURES
    """
    
    window = 4
    
    # Goal Intensity
    if all(col in df.columns for col in [
        f'Goals_Home_L{window}', f'Conceded_Away_L{window}',
        f'Goals_Away_L{window}', f'Conceded_Home_L{window}'
    ]):
        df[f'Goal_Intensity_L{window}'] = (
            df[f'Goals_Home_L{window}'] + df[f'Conceded_Away_L{window}'] +
            df[f'Goals_Away_L{window}'] + df[f'Conceded_Home_L{window}']
        ) / 2
    
    # Points Difference
    if all(col in df.columns for col in [
        f'Points_Home_L{window}', f'Points_Away_L{window}'
    ]):
        df[f'Points_Diff_L{window}'] = (
            df[f'Points_Home_L{window}'] - df[f'Points_Away_L{window}']
        )
    
    # Form Ratio
    if all(col in df.columns for col in [
        f'Points_Home_L{window}', f'Points_Away_L{window}'
    ]):
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
    
    # Season Aggregates (bleiben)
    df['Points_Total_Season'] = df['Points_Home_Season'] + df['Points_Away_Season']
    df['Goals_Total_Season'] = df['Goals_Home_Season'] + df['Goals_Away_Season']
    df['Conceded_Total_Season'] = df['Conceded_Home_Season'] + df['Conceded_Away_Season']
    df['GoalDiff_Total_Season'] = df['GoalDiff_Home_Season'] + df['GoalDiff_Away_Season']
    
    # ✅ Draw Risk jetzt mit L4
    df['Draw_Risk'] = (
        (df[f'Points_Diff_L{window}'].abs() < 2) &
        (df[f'Goals_Home_L{window}'] < 2) &
        (df[f'Goals_Away_L{window}'] < 2) &
        (df[f'Goal_Intensity_L{window}'] < 2.5)
    ).astype(float)
    
    # Value Indicators (bleiben)
    df['Value_Home'] = np.where(
        df['MaxQuote_H'].notna(),
        (1 / df['MaxQuote_H']) - df['ImpProb_H'],
        0.0
    )
    
    df['Value_Away'] = np.where(
        df['MaxQuote_A'].notna(),
        (1 / df['MaxQuote_A']) - df['ImpProb_A'],
        0.0
    )
    
    # ✅ Momentum jetzt mit L4
    df['Momentum_Home'] = df.apply(
        lambda row: (
            row[f'Points_Home_L{window}'] - 
            (row['Points_Home_Season'] / max(1, row['Gameday']))
        ) if row['Gameday'] > 0 else 0,
        axis=1
    )
    
    df['Momentum_Away'] = df.apply(
        lambda row: (
            row[f'Points_Away_L{window}'] - 
            (row['Points_Away_Season'] / max(1, row['Gameday']))
        ) if row['Gameday'] > 0 else 0,
        axis=1
    )
    
    # ✅ Expected Goals jetzt mit L4
    df['Expected_Goals'] = (
        (df[f'Goals_Home_L{window}'] + df[f'Conceded_Away_L{window}'] +
         df[f'Goals_Away_L{window}'] + df[f'Conceded_Home_L{window}']) / 2
    )
    
    df['Expected_Goals_Weighted'] = (
        0.7 * df['Expected_Goals'] + 
        0.3 * (df['Goals_Total_Season'] / df['Gameday'].clip(lower=1))
    )
    
    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Behält nur relevante Spalten."""
    mandatory = [
        "Index", "Season", "Date", "HomeTeam", "AwayTeam", 
        "FTHG", "FTAG", "FTR", "Gameday"
    ]
    
    # Alle L3-L10 Features behalten
    feature_patterns = [
        "_L3", "_L4", "_L5", "_L6", "_L10",
        "Quote", "Prob", "OU_", "Bookie",
        "_Season", "Value_", "Momentum_", 
        "Expected_", "Draw_Risk"
    ]
    
    feature_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in feature_patterns):
            feature_cols.append(col)
    
    # In-Game Stats
    ingame = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
    
    keep_cols = mandatory + feature_cols + ingame
    existing_cols = [c for c in keep_cols if c in df.columns]
    
    return df[existing_cols].copy()


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bundesliga Data Downloader & Preprocessor")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess data")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    
    if args.download:
        download_all_seasons(raw_dir, args.seasons)
    
    if args.preprocess:
        preprocess_data(raw_dir, processed_dir)