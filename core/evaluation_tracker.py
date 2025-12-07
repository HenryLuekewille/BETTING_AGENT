"""
Enhanced Evaluation Tracker mit Excel-Export.
✅ Angepasst für Flask Web-App
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ Backend für Web-Server
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class EvaluationTracker:
    """
    Verfolgt Performance über Spieltage mit detailliertem Excel-Export.
    
    Erstellt:
    - Spieltag-Zusammenfassungen
    - Einzelne Wett-Entscheidungen
    - Action-Statistiken
    - Team-Statistiken
    - Performance-Plots
    """
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Storage
        self.gameday_results = []
        self.all_decisions = []
        
        # Tracking
        self.cumulative_profit = 0.0
        self.cumulative_invested = 0.0
        self.cumulative_bets = 0
        self.cumulative_wins = 0
        
        print(f"📊 Evaluation Tracker initialisiert")
        print(f"   Output Dir: {self.results_dir}\n")
    
    def log_gameday(self, gameday, results, predictions=None, gameday_data=None):
        """
        Logge kompletten Spieltag.
        
        Args:
            gameday: Spieltag-Nummer
            results: Dict mit Metriken (roi, winrate, etc.)
            predictions: Liste der Vorhersagen (optional)
            gameday_data: DataFrame der Spiele (optional)
        """
        # Update Kumulative Werte
        self.cumulative_profit += results['total_profit']
        self.cumulative_invested += results['total_invested']
        self.cumulative_bets += results['total_bets']
        self.cumulative_wins += results['total_wins']
        
        # Gameday Summary
        self.gameday_results.append({
            'Gameday': gameday,
            'BetsPlaced': results['total_bets'],
            'Wins': results['total_wins'],
            'Losses': results['total_bets'] - results['total_wins'],
            'TotalProfit': results['total_profit'],
            'TotalInvested': results['total_invested'],
            'ROI': results['roi'],
            'Winrate': results['winrate'],
            'CumulativeProfit': self.cumulative_profit,
            'CumulativeInvested': self.cumulative_invested,
            'CumulativeBets': self.cumulative_bets,
            'CumulativeWins': self.cumulative_wins,
            'CumulativeROI': self.cumulative_profit / max(1, self.cumulative_invested),
            'CumulativeWinrate': self.cumulative_wins / max(1, self.cumulative_bets),
        })
        
        # Einzelne Entscheidungen (falls verfügbar)
        if predictions and gameday_data is not None:
            self._log_decisions(gameday, predictions, gameday_data)
    
    def _log_decisions(self, gameday, predictions, gameday_data):
        """✅ Logge einzelne Wett-Entscheidungen mit variabler Bet Size."""
        action_names = ["No Bet", "Home", "Away", "Over", "Under"]
        
        # ✅ Lade Base/Max Bet (falls verfügbar)
        base_bet = 10.0
        max_bet = 30.0
        
        for pred in predictions:
            row = gameday_data.iloc[pred['match_idx']]
            action = pred['action']
            
            # Quote und Probability ermitteln
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
                quote = np.nan
                prob = 0.33
            
            # Ergebnis auswerten
            ftr = pred['actual_ftr']
            total_goals = pred['actual_fthg'] + pred['actual_ftag']
            
            if action == 0:
                won = False
                profit = 0.0
                bet_amount = 0.0
            else:
                # ✅ VARIABLE BET SIZE (vereinfacht für Tracker)
                # Nutze Confidence-Based Scaling
                if prob > 0.75:
                    bet_amount = min(base_bet * 2.5, max_bet)
                elif prob > 0.70:
                    bet_amount = min(base_bet * 2.0, max_bet)
                elif prob > 0.65:
                    bet_amount = min(base_bet * 1.5, max_bet)
                elif prob > 0.60:
                    bet_amount = min(base_bet * 1.2, max_bet)
                else:
                    bet_amount = base_bet
                
                # Value Bonus
                if action in [1, 2]:
                    value = self._safe_float(row.get(f'Value_{"Home" if action == 1 else "Away"}', 0))
                    if value > 0.08:
                        bet_amount = min(bet_amount * 1.2, max_bet)
                    elif value > 0.05:
                        bet_amount = min(bet_amount * 1.1, max_bet)
                
                bet_amount = round(bet_amount, 2)
                
                if action == 1:
                    won = (ftr == 'H')
                elif action == 2:
                    won = (ftr == 'A')
                elif action == 3:
                    won = (total_goals > 2.5)
                elif action == 4:
                    won = (total_goals <= 2.5)
                
                if won:
                    profit = (quote - 1.0) * bet_amount
                else:
                    profit = -bet_amount
            
            # Speichern
            self.all_decisions.append({
                'Gameday': gameday,
                'Date': row.get('Date', ''),
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
                'Action': action,
                'ActionName': action_names[action],
                'Quote': quote,
                'BetAmount': bet_amount,  # ✅ Variable Bet Size
                'Probability': prob,       # ✅ NEU
                'ActualResult': ftr,
                'ActualGoalsHome': pred['actual_fthg'],
                'ActualGoalsAway': pred['actual_ftag'],
                'TotalGoals': total_goals,
                'Won': won,
                'Profit': profit,
                # Features
                'ImpProb_H': self._safe_float(row.get('ImpProb_H', np.nan)),
                'ImpProb_D': self._safe_float(row.get('ImpProb_D', np.nan)),
                'ImpProb_A': self._safe_float(row.get('ImpProb_A', np.nan)),
                'Points_Home_L4': self._safe_float(row.get('Points_Home_L4', np.nan)),
                'Points_Away_L4': self._safe_float(row.get('Points_Away_L4', np.nan)),
                'Goals_Home_L4': self._safe_float(row.get('Goals_Home_L4', np.nan)),
                'Goals_Away_L4': self._safe_float(row.get('Goals_Away_L4', np.nan)),
            })
        
    def save_final_report(self, season):
        """Speichere finalen Report als Excel mit mehreren Sheets."""
        print("\n" + "="*80)
        print("💾 ERSTELLE EXCEL-REPORT")
        print("="*80 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = self.results_dir / f"evaluation_{season}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Gameday Summaries
            df_gamedays = pd.DataFrame(self.gameday_results)
            df_gamedays.to_excel(writer, sheet_name='Gameday_Summaries', index=False)
            print(f"   ✅ Sheet 'Gameday_Summaries': {len(df_gamedays)} Spieltage")
            
            # Sheet 2: All Decisions
            if self.all_decisions:
                df_decisions = pd.DataFrame(self.all_decisions)
                df_decisions.to_excel(writer, sheet_name='All_Decisions', index=False)
                print(f"   ✅ Sheet 'All_Decisions': {len(df_decisions)} Entscheidungen")
            
            # Sheet 3: Action Stats
            df_actions = self._create_action_stats()
            if not df_actions.empty:
                df_actions.to_excel(writer, sheet_name='Action_Stats', index=False)
                print(f"   ✅ Sheet 'Action_Stats': {len(df_actions)} Aktionen")
            
            # Sheet 4: Team Stats
            df_teams = self._create_team_stats()
            if not df_teams.empty:
                df_teams.to_excel(writer, sheet_name='Team_Stats', index=False)
                print(f"   ✅ Sheet 'Team_Stats': {len(df_teams)} Teams")
            
            # Sheet 5: Summary
            df_summary = self._create_summary(season)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✅ Sheet 'Summary': Gesamt-Übersicht")
        
        print(f"\n💾 Excel gespeichert: {excel_path}\n")
        
        # CSV-Export (als Backup)
        csv_path = self.results_dir / f"gameday_results_{season}_{timestamp}.csv"
        df_gamedays.to_csv(csv_path, index=False, sep=';')
        print(f"💾 CSV gespeichert: {csv_path}\n")
        
        # Plot
        self._plot_results(df_gamedays, season, timestamp)
        
        return excel_path
    
    def _create_action_stats(self):
        """Erstelle Action-Statistiken."""
        if not self.all_decisions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_decisions)
        
        # Filtere nur Wetten (Action != 0)
        df_bets = df[df['Action'] != 0].copy()
        
        if len(df_bets) == 0:
            return pd.DataFrame()
        
        stats = df_bets.groupby('ActionName').agg({
            'BetAmount': ['count', 'sum'],
            'Won': 'sum',
            'Profit': ['sum', 'mean', 'std'],
            'Quote': 'mean'
        }).round(2)
        
        stats.columns = [
            'Count', 'TotalBet', 'Wins', 
            'TotalProfit', 'AvgProfit', 'StdProfit', 'AvgQuote'
        ]
        
        stats['Losses'] = stats['Count'] - stats['Wins']
        stats['Winrate'] = (stats['Wins'] / stats['Count'] * 100).round(2)
        stats['ROI'] = (stats['TotalProfit'] / stats['TotalBet'] * 100).round(2)
        
        return stats.reset_index()
    
    def _create_team_stats(self):
        """Erstelle Team-Statistiken."""
        if not self.all_decisions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_decisions)
        
        # Home-Team Stats
        home_stats = df[df['Action'] == 1].groupby('HomeTeam').agg({
            'BetAmount': 'count',
            'Won': 'sum',
            'Profit': 'sum'
        })
        home_stats.columns = ['BetsHome', 'WinsHome', 'ProfitHome']
        
        # Away-Team Stats
        away_stats = df[df['Action'] == 2].groupby('AwayTeam').agg({
            'BetAmount': 'count',
            'Won': 'sum',
            'Profit': 'sum'
        })
        away_stats.columns = ['BetsAway', 'WinsAway', 'ProfitAway']
        
        # Kombiniere
        team_stats = pd.concat([home_stats, away_stats], axis=1).fillna(0)
        
        team_stats['TotalBets'] = (
            team_stats['BetsHome'] + team_stats['BetsAway']
        )
        team_stats['TotalWins'] = (
            team_stats['WinsHome'] + team_stats['WinsAway']
        )
        team_stats['TotalProfit'] = (
            team_stats['ProfitHome'] + team_stats['ProfitAway']
        )
        
        # Nur Teams mit Wetten
        team_stats = team_stats[team_stats['TotalBets'] > 0]
        
        if len(team_stats) > 0:
            team_stats['Winrate'] = (
                team_stats['TotalWins'] / team_stats['TotalBets'] * 100
            ).round(2)
        
        return team_stats.reset_index().sort_values(
            'TotalProfit', 
            ascending=False
        )
    
    def _create_summary(self, season):
        """Erstelle Gesamt-Zusammenfassung."""
        summary_data = {
            'Metric': [
                'Season',
                'Total Gamedays',
                'Total Bets',
                'Total Wins',
                'Total Losses',
                'Winrate (%)',
                'Total Invested (€)',
                'Total Profit (€)',
                'ROI (%)',
                'Average Profit per Bet (€)',
                'Best Gameday (ROI)',
                'Worst Gameday (ROI)',
            ],
            'Value': []
        }
        
        df_gd = pd.DataFrame(self.gameday_results)
        
        summary_data['Value'] = [
            season,
            len(df_gd),
            self.cumulative_bets,
            self.cumulative_wins,
            self.cumulative_bets - self.cumulative_wins,
            f"{(self.cumulative_wins / max(1, self.cumulative_bets) * 100):.2f}",
            f"{self.cumulative_invested:.2f}",
            f"{self.cumulative_profit:.2f}",
            f"{(self.cumulative_profit / max(1, self.cumulative_invested) * 100):.2f}",
            f"{(self.cumulative_profit / max(1, self.cumulative_bets)):.2f}",
            f"GD {df_gd.loc[df_gd['ROI'].idxmax(), 'Gameday']} ({df_gd['ROI'].max()*100:.2f}%)" if len(df_gd) > 0 else "N/A",
            f"GD {df_gd.loc[df_gd['ROI'].idxmin(), 'Gameday']} ({df_gd['ROI'].min()*100:.2f}%)" if len(df_gd) > 0 else "N/A",
        ]
        
        return pd.DataFrame(summary_data)
    
    def _plot_results(self, df, season, timestamp):
        """Erstelle Performance-Plots."""
        if len(df) == 0:
            print("⚠️  Keine Daten für Plot verfügbar\n")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Performance Analysis - Season {season}', 
                     fontsize=16, fontweight='bold')
        
        # 1. ROI over Gamedays
        axes[0, 0].plot(df['Gameday'], df['ROI'] * 100, 
                        marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('ROI über Spieltage', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Spieltag')
        axes[0, 0].set_ylabel('ROI (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Cumulative ROI
        axes[0, 1].plot(df['Gameday'], df['CumulativeROI'] * 100, 
                        marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_title('Kumulativer ROI', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Spieltag')
        axes[0, 1].set_ylabel('Kumulativer ROI (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Cumulative Profit
        axes[1, 0].plot(df['Gameday'], df['CumulativeProfit'], 
                        marker='o', linewidth=2, markersize=6, color='blue')
        axes[1, 0].set_title('Kumulierter Gewinn', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Spieltag')
        axes[1, 0].set_ylabel('Profit (€)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Winrate
        axes[1, 1].plot(df['Gameday'], df['Winrate'] * 100, 
                        marker='o', linewidth=2, markersize=6, color='orange')
        axes[1, 1].set_title('Winrate über Spieltage', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Spieltag')
        axes[1, 1].set_ylabel('Winrate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(50, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = self.results_dir / f"performance_{season}_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot gespeichert: {plot_path}\n")
    
    @staticmethod
    def _safe_float(x):
        """Konvertiert zu float, gibt NaN zurück bei Fehler."""
        try:
            val = float(x)
            return val if not (np.isnan(val) or np.isinf(val)) else np.nan
        except (TypeError, ValueError):
            return np.nan