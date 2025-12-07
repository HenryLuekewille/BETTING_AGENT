"""
Optimized Betting Environment for Reinforcement Learning
✅ L4-Focus (Last 4 Games)
✅ Variable Bet Sizes (Kelly Criterion)
✅ Enhanced Reward Shaping
✅ NEU: Confidence-Based Bet Scaling (4 Modi)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class BettingEnvOptimized(gym.Env):
    """
    Optimierte RL-Umgebung für Sportwetten.
    
    ✅ Fokus auf L4-Features (letzte 4 Spiele)
    ✅ Variable Bet Sizes (Basis bis Maximum)
    ✅ Kelly Criterion Support
    ✅ Verbesserte Reward-Struktur
    ✅ NEU: Confidence-Based Bet Scaling
    """

    metadata = {"render_modes": []}
    
    # ✅ OPTIMIERTE FEATURE-LISTE (NUR L4 + Odds + Season)
    OPTIMAL_FEATURES = [
        # Goals L4 (Primary Focus)
        'Goals_Home_L4', 'Goals_Away_L4',
        'Conceded_Away_L4', 'Conceded_Home_L4',
        
        # Points L4
        'Points_Home_L4', 'Points_Away_L4',
        
        # Odds (KRITISCH - höchste Feature Importance!)
        'ImpProb_H', 'ImpProb_A', 'ImpProb_D',
        'MaxQuote_H', 'MaxQuote_A',
        'OU_Over', 'OU_Under', 'OU_Spread',
        'BookieMargin',
        
        # Season Context
        'Points_Total_Season', 'Points_Home_Season', 'Points_Away_Season',
        
        # Advanced L4
        'Points_Diff_L4',
        'Form_Ratio_L4',
        'Attack_Defense_Home_L4',
        'Attack_Defense_Away_L4',
        
        # Discipline L4
        'Corners_Home_L4', 'Corners_Away_L4',
        'Fouls_Home_L4', 'Fouls_Away_L4',
        
        # Momentum & Value
        'Momentum_Home', 'Momentum_Away',
        'Value_Home', 'Value_Away',
        'Expected_Goals', 'Expected_Goals_Weighted',
        'Draw_Risk',
    ]

    def __init__(
        self, 
        data: pd.DataFrame,
        use_betting_odds: bool = True,
        confidence_threshold: float = 0.60,
        bet_amount: float = 10.0,
        max_bet_amount: float = 30.0,
        min_gameday: int = 4,
        apply_gameday_filter: bool = True,
        reward_shaping: str = "conservative",
        no_bet_reward_multiplier: float = 0.5,
        draw_penalty_multiplier: float = 1.5,
        min_edge_required: float = 0.02,
        max_bet_rate: float = 0.40,
        use_kelly_criterion: bool = True,
        kelly_fraction: float = 0.25,
        confidence_scaling_mode: str = "none",  # ✅ NEU!
        debug_mode: bool = False,
    ):
        """
        Initialisiert die Betting Environment.
        
        Args:
            data: DataFrame mit Spieldaten
            use_betting_odds: Nutze Wettquoten als Features
            confidence_threshold: Min. Confidence für Wetten (0.5-0.95)
            bet_amount: Basis-Wetteinsatz in Euro
            max_bet_amount: Maximaler Wetteinsatz in Euro
            min_gameday: Frühester Spieltag für Training
            apply_gameday_filter: Filter nach min_gameday
            reward_shaping: "conservative" oder "aggressive"
            no_bet_reward_multiplier: Reward-Multiplikator für korrekte No-Bets
            draw_penalty_multiplier: Straf-Multiplikator für Draw-Wetten
            min_edge_required: Minimum Value-Edge für Wetten
            max_bet_rate: Maximum % der Spiele auf die gewettet wird
            use_kelly_criterion: Nutze Kelly Criterion für Bet Sizes
            kelly_fraction: Kelly Fraction (0.1-0.5, typisch 0.25)
            confidence_scaling_mode: "none", "linear", "quadratic", "threshold"  # ✅ NEU!
            debug_mode: Aktiviere Debug-Ausgaben
        """
        super().__init__()
        
        # Store parameters
        self.use_betting_odds = use_betting_odds
        self.confidence_threshold = confidence_threshold
        self.base_bet_amount = bet_amount
        self.max_bet_amount = max_bet_amount
        self.reward_shaping = reward_shaping
        self.debug_mode = debug_mode
        
        self.no_bet_reward_multiplier = no_bet_reward_multiplier
        self.draw_penalty_multiplier = draw_penalty_multiplier
        self.min_edge_required = min_edge_required
        self.max_bet_rate = max_bet_rate
        self.use_kelly_criterion = use_kelly_criterion
        self.kelly_fraction = kelly_fraction
        self.confidence_scaling_mode = confidence_scaling_mode  # ✅ NEU!
        
        # ==================== DATA FILTERING ====================
        self.original_data = data.copy()
        
        if apply_gameday_filter:
            filtered = []
            for season in sorted(data["Season"].unique()):
                season_data = data[data["Season"] == season]
                season_filtered = season_data[
                    season_data["Gameday"] >= min_gameday
                ]
                filtered.append(season_filtered)
            
            self.data = pd.concat(filtered, ignore_index=True)
            
            if self.debug_mode:
                print(
                    f"   🎯 Gameday-Filter (>= {min_gameday}):\n"
                    f"      Original:  {len(data):>5} Spiele\n"
                    f"      Filtered:  {len(self.data):>5} Spiele\n"
                    f"      Entfernt:  {len(data) - len(self.data):>5} Spiele"
                )
        else:
            self.data = data.reset_index(drop=True)
        
        if self.data.empty:
            raise ValueError(
                f"❌ Keine Spiele nach Filter (Gameday >= {min_gameday})!"
            )
        
        # ==================== FEATURE SELECTION ====================
        if use_betting_odds:
            # Mit Quoten: Nutze optimale Feature-Liste
            self.features = [
                f for f in self.OPTIMAL_FEATURES 
                if f in self.data.columns
            ]
        else:
            # Ohne Quoten: Nur Stats (keine Odds)
            pure_stats = [
                f for f in self.OPTIMAL_FEATURES 
                if not any(x in f for x in ['Quote', 'Prob', 'OU_', 'Bookie', 'Value'])
            ]
            self.features = [f for f in pure_stats if f in self.data.columns]
        
        # ==================== SPACES ====================
        # Actions: 0=No Bet, 1=Home, 2=Away, 3=Over, 4=Under
        self.action_space = spaces.Discrete(5)
        
        # Observation: Features + draw_signal + bet_budget_ratio
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features) + 2,),
            dtype=np.float32
        )
        
        if self.debug_mode:
            print(
                f"   🎮 Action Space: {self.action_space.n} Aktionen\n"
                f"   🔭 Observation Space: {self.observation_space.shape}\n"
                f"   🎯 Features: {len(self.features)} (L4-optimiert)\n"
                f"   💰 Bet Amount: {self.base_bet_amount}€ - {self.max_bet_amount}€\n"
                f"   📊 Confidence Threshold: {self.confidence_threshold}\n"
                f"   🚫 Min Edge: {self.min_edge_required}\n"
                f"   📈 Max Bet Rate: {self.max_bet_rate*100:.0f}%\n"
                f"   🎲 Kelly Criterion: {use_kelly_criterion} (Fraction: {kelly_fraction})\n"
                f"   ⚙️  Confidence Scaling: {confidence_scaling_mode}"  # ✅ NEU!
            )
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Environment zurücksetzen."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = 0.0
        self.total_bets = 0
        self.success_bets = 0
        self.total_profit = 0.0
        self.total_invested = 0.0
        self.invalid_actions = 0
        self.draw_avoided = 0
        self.no_bet_correct = 0
        self.no_bet_wrong = 0
        self.last_action = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Erstellt Observation Vector.
        
        Returns:
            np.array: [features..., draw_signal, bet_budget_ratio]
        """
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Core Features
        obs_features = row[self.features].values.astype(np.float32)
        obs_features = np.nan_to_num(obs_features, nan=0.0)
        
        # ✅ Draw Signal
        if self.use_betting_odds and 'ImpProb_D' in self.data.columns:
            draw_prob = self._safe_float(row.get('ImpProb_D', 0.33))
        else:
            prob_h = self._safe_float(row.get('ImpProb_H', 0.4))
            prob_a = self._safe_float(row.get('ImpProb_A', 0.4))
            draw_prob = max(0.0, 1.0 - prob_h - prob_a)
        
        # ✅ Bet Budget Ratio (wie viel % des Budgets wurde bereits investiert)
        bet_budget_ratio = self.total_invested / max(1, self.current_step * self.base_bet_amount)
        
        # Combine
        obs = np.append(obs_features, [draw_prob, bet_budget_ratio])
        
        return obs.astype(np.float32)

    def _should_place_bet(self, action: int, row: pd.Series) -> bool:
        """
        ✅ STRENGE VERSION: Entscheidet ob Wette platziert werden soll.
        
        Args:
            action: Gewählte Aktion (1-4)
            row: Spiel-Daten
            
        Returns:
            bool: True wenn Wette erlaubt
        """
        if action == 0:
            return False
        
        # ✅ CHECK BET RATE LIMIT (STRENG!)
        current_bet_rate = self.total_bets / max(1, self.current_step)
        if current_bet_rate >= self.max_bet_rate:
            if self.debug_mode:
                print(f"⚠️  Bet Rate Limit erreicht: {current_bet_rate*100:.1f}% >= {self.max_bet_rate*100:.1f}%")
            return False
        
        # Hole Wahrscheinlichkeiten
        prob_h = self._safe_float(row.get('ImpProb_H', 0.33))
        prob_d = self._safe_float(row.get('ImpProb_D', 0.33))
        prob_a = self._safe_float(row.get('ImpProb_A', 0.33))
        
        # ✅ Draw-Adjusted Threshold
        base_threshold = self.confidence_threshold
        draw_adjustment = prob_d * 0.10
        dynamic_threshold = min(0.75, base_threshold + draw_adjustment)
        
        # Home Bet
        if action == 1:
            quote_h = self._safe_float(row.get('MaxQuote_H', np.nan))
            if np.isnan(quote_h):
                return False
            
            implied_prob = 1 / quote_h
            edge = prob_h - implied_prob
            
            # ✅ STRENGERE ANFORDERUNGEN
            return (
                (prob_h > dynamic_threshold) or           # Hohe Confidence ODER
                (edge > self.min_edge_required * 1.5) or  # Sehr guter Value ODER
                (prob_h > 0.60 and edge > self.min_edge_required)  # Gute Conf + Edge
            )
        
        # Away Bet
        elif action == 2:
            quote_a = self._safe_float(row.get('MaxQuote_A', np.nan))
            if np.isnan(quote_a):
                return False
            
            implied_prob = 1 / quote_a
            edge = prob_a - implied_prob
            
            return (
                (prob_a > dynamic_threshold) or 
                (edge > self.min_edge_required * 1.5) or
                (prob_a > 0.60 and edge > self.min_edge_required)
            )
        
        # Over/Under
        elif action in [3, 4]:
            expected_goals = self._safe_float(row.get('Expected_Goals', 2.5))
            
            if action == 3:  # Over
                return expected_goals > 2.8  # ✅ Höherer Threshold
            else:  # Under
                return expected_goals < 2.2  # ✅ Niedriger Threshold
        
        return False

    def _apply_confidence_scaling(self, bet_size: float, prob: float, mode: str) -> float:
        """
        ✅ NEU: Wendet Confidence-Based Scaling auf Bet Size an.
        
        Args:
            bet_size: Basis Bet Size
            prob: Win Probability
            mode: 'none', 'linear', 'quadratic', 'threshold'
            
        Returns:
            float: Skalierte Bet Size
        """
        if mode == "none":
            return bet_size
        
        elif mode == "linear":
            # Linear scaling: 0.5 -> 0.5x, 1.0 -> 2.0x
            factor = max(0.5, min(2.0, prob * 2))
            return bet_size * factor
        
        elif mode == "quadratic":
            # Quadratic scaling: stärkere Reaktion auf hohe Confidence
            # Formel: factor = (prob ^ 1.5) * 2
            # Bei prob=0.5: factor=0.71, bei prob=0.75: factor=1.30, bei prob=1.0: factor=2.0
            factor = max(0.5, min(2.5, (prob ** 1.5) * 2))
            return bet_size * factor
        
        elif mode == "threshold":
            # Threshold-based: Sprünge bei bestimmten Levels
            if prob >= 0.75:
                factor = 2.5
            elif prob >= 0.70:
                factor = 2.0
            elif prob >= 0.65:
                factor = 1.5
            elif prob >= 0.60:
                factor = 1.0
            else:
                factor = 0.5
            
            return bet_size * factor
        
        return bet_size

    def _calculate_bet_size(self, action: int, quote: float, row: pd.Series) -> float:
        """
        ✅ Berechnet variable Bet Size basierend auf Confidence.
        ✅ NEU: Mit Confidence-Based Scaling
        
        Args:
            action: Gewählte Aktion
            quote: Wettquote
            row: Spiel-Daten
            
        Returns:
            float: Bet Size in Euro
        """
        if np.isnan(quote):
            return self.base_bet_amount
        
        # ✅ Win-Probability ermitteln
        if action == 1:
            prob = self._safe_float(row.get('ImpProb_H', 0.33))
        elif action == 2:
            prob = self._safe_float(row.get('ImpProb_A', 0.33))
        elif action == 3:
            expected_goals = self._safe_float(row.get('Expected_Goals', 2.5))
            prob = min(0.8, expected_goals / 5.0)
        elif action == 4:
            expected_goals = self._safe_float(row.get('Expected_Goals', 2.5))
            prob = max(0.2, 1 - (expected_goals / 5.0))
        else:
            return self.base_bet_amount
        
        # ✅ KELLY CRITERION (wenn aktiviert)
        if self.use_kelly_criterion and prob > 0.5:
            # Kelly Formula: f* = (bp - q) / b
            # b = net odds (quote - 1)
            # p = win probability
            # q = loss probability (1 - p)
            
            b = quote - 1
            q = 1 - prob
            
            kelly_fraction = (b * prob - q) / b
            kelly_fraction = max(0, kelly_fraction)
            
            # ✅ Fractional Kelly (safer)
            kelly_fraction *= self.kelly_fraction
            
            # Bet Size = Base + Kelly Bonus (bis zu 3x Base)
            bet_size = self.base_bet_amount * (1 + kelly_fraction * 3)
            
        else:
            # ✅ Confidence-Based Scaling (ohne Kelly)
            if prob > 0.75:
                multiplier = 2.5
            elif prob > 0.70:
                multiplier = 2.0
            elif prob > 0.65:
                multiplier = 1.5
            elif prob > 0.60:
                multiplier = 1.2
            else:
                multiplier = 1.0
            
            bet_size = self.base_bet_amount * multiplier
        
        # ✅ Value Bonus
        if action in [1, 2]:
            value = self._safe_float(row.get(f'Value_{"Home" if action == 1 else "Away"}', 0))
            if value > 0.08:
                bet_size = bet_size * 1.2
            elif value > 0.05:
                bet_size = bet_size * 1.1
        
        # ✅ NEU: Apply Confidence-Based Scaling
        bet_size = self._apply_confidence_scaling(bet_size, prob, self.confidence_scaling_mode)
        
        # ✅ Cap bei Maximum
        bet_size = min(bet_size, self.max_bet_amount)
        
        return round(bet_size, 2)

    def step(self, action: int):
        """
        Führt Aktion aus und gibt Reward zurück.
        
        Args:
            action: 0=No Bet, 1=Home, 2=Away, 3=Over, 4=Under
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        if self.current_step >= len(self.data):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, self._get_info()
        
        row = self.data.iloc[self.current_step]
        self.last_action = action
        
        # Spielergebnis
        ftr = row["FTR"]
        total_goals = row["FTHG"] + row["FTAG"]
        is_draw = (ftr == "D")
        
        # Quoten
        qH = self._safe_float(row.get("MaxQuote_H", np.nan))
        qA = self._safe_float(row.get("MaxQuote_A", np.nan))
        qOver = self._safe_float(row.get("OU_Over", np.nan))
        qUnder = self._safe_float(row.get("OU_Under", np.nan))
        
        reward = 0.0
        bet_placed = False
        
        # ============== NO BET ==============
        if action == 0:
            draw_prob = self._safe_float(row.get('ImpProb_D', 0.33))
            
            if is_draw:
                # ✅ Korrekt: Kein Bet bei Draw
                reward = 0.1 * self.base_bet_amount * (1 + draw_prob * 0.3)
                self.draw_avoided += 1
                self.no_bet_correct += 1
            else:
                # Check ob gute Gelegenheit verpasst wurde
                max_prob = max(
                    self._safe_float(row.get('ImpProb_H', 0.33)),
                    self._safe_float(row.get('ImpProb_A', 0.33))
                )
                
                value_h = self._safe_float(row.get('Value_Home', 0))
                value_a = self._safe_float(row.get('Value_Away', 0))
                max_value = max(value_h, value_a)
                
                if max_prob > 0.70 and max_value > 0.08:
                    # Klare Gelegenheit verpasst
                    reward = -0.4 * self.base_bet_amount
                    self.no_bet_wrong += 1
                elif max_prob > 0.60:
                    reward = -0.15 * self.base_bet_amount
                    self.no_bet_wrong += 1
                else:
                    # OK, keine klare Gelegenheit
                    reward = 0.03 * self.base_bet_amount
                    self.no_bet_correct += 1
        
        # ============== BETTING ==============
        else:
            # Check ob Wette erlaubt ist
            if not self._should_place_bet(action, row):
                # Wette abgelehnt
                if is_draw:
                    reward = 0.05 * self.base_bet_amount
                    self.draw_avoided += 1
                else:
                    reward = 0.0
                
                self.current_step += 1
                obs = self._get_obs()
                done = (self.current_step >= len(self.data))
                return obs, reward, done, False, self._get_info()
            
            # ✅ Wette wird platziert
            bet_placed = True
            
            # ✅ Berechne variable Bet Size
            if action == 1:
                bet_size = self._calculate_bet_size(action, qH, row)
                quote = qH
            elif action == 2:
                bet_size = self._calculate_bet_size(action, qA, row)
                quote = qA
            elif action == 3:
                bet_size = self._calculate_bet_size(action, qOver, row)
                quote = qOver
            else:
                bet_size = self._calculate_bet_size(action, qUnder, row)
                quote = qUnder
            
            self.total_bets += 1
            self.total_invested += bet_size
            
            # ✅ Evaluate Bet
            if action == 1:
                won = (ftr == "H") if not is_draw else False
                is_draw_trap = is_draw
            elif action == 2:
                won = (ftr == "A") if not is_draw else False
                is_draw_trap = is_draw
            elif action == 3:
                won = (total_goals > 2.5)
                is_draw_trap = False
            elif action == 4:
                won = (total_goals <= 2.5)
                is_draw_trap = False
            
            # ✅ Calculate Reward
            if won:
                profit = (quote - 1.0) * bet_size
                
                # Bonus für Value Bets
                if action in [1, 2]:
                    value = self._safe_float(
                        row.get('Value_Home' if action == 1 else 'Value_Away', 0)
                    )
                    if value > 0.08:
                        profit *= 1.2
                    elif value > 0.04:
                        profit *= 1.1
                
                # Bonus für hohe Quoten
                if quote >= 2.5:
                    profit *= 1.1
                elif quote >= 2.0:
                    profit *= 1.05
                
                reward = profit
                self.success_bets += 1
            else:
                if is_draw_trap:
                    # Draw-Verlust härter bestrafen
                    reward = -self.draw_penalty_multiplier * bet_size
                else:
                    # Normaler Verlust
                    reward = -bet_size
        
        # Update Balance
        self.balance += reward
        self.total_profit += reward
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= len(self.data))
        
        # Get next observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _get_info(self) -> dict:
        """
        Erstellt Info-Dictionary.
        
        Returns:
            dict: Metriken und Statistiken
        """
        roi = self.total_profit / max(1, self.total_invested)
        winrate = self.success_bets / max(1, self.total_bets)
        bet_rate = self.total_bets / max(1, self.current_step)
        no_bet_accuracy = self.no_bet_correct / max(1, self.no_bet_correct + self.no_bet_wrong)
        avg_bet_size = self.total_invested / max(1, self.total_bets)
        
        return {
            "roi": roi,
            "balance": self.balance,
            "bets": self.total_bets,
            "wins": self.success_bets,
            "winrate": winrate,
            "invalid_actions": self.invalid_actions,
            "draw_avoided": self.draw_avoided,
            "draw_avoidance_rate": self.draw_avoided / max(1, self.current_step),
            "step": self.current_step,
            "total_invested": self.total_invested,
            "total_profit": self.total_profit,
            "avg_profit_per_bet": self.total_profit / max(1, self.total_bets),
            "bet_rate": bet_rate,
            "no_bet_correct": self.no_bet_correct,
            "no_bet_wrong": self.no_bet_wrong,
            "no_bet_accuracy": no_bet_accuracy,
            "avg_bet_size": avg_bet_size,
            "max_bet_amount": self.max_bet_amount,
            "confidence_scaling_mode": self.confidence_scaling_mode,  # ✅ NEU!
        }

    @staticmethod
    def _safe_float(x):
        """
        Konvertiert zu float mit Error-Handling.
        
        Args:
            x: Input value
            
        Returns:
            float: Konvertierter Wert oder NaN
        """
        try:
            val = float(x)
            return val if not (np.isnan(val) or np.isinf(val)) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def render(self):
        """Rendering (nicht implementiert)."""
        pass

    def close(self):
        """Cleanup (nicht implementiert)."""
        pass


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def test_environment():
    """Test-Funktion für die Environment."""
    import pandas as pd
    from pathlib import Path
    
    # Lade Test-Daten
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    csv_files = list(data_dir.glob("Bundesliga_*.csv"))
    
    if not csv_files:
        print("❌ Keine Daten gefunden!")
        return
    
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"📂 Lade Daten: {latest_csv.name}")
    
    df = pd.read_csv(latest_csv, sep=";")
    df = df[df["Season"] == 2023].reset_index(drop=True)
    
    print(f"📊 Test-Daten: {len(df)} Spiele (Saison 2023)\n")
    
    # Teste verschiedene Scaling-Modi
    for scaling_mode in ["none", "linear", "quadratic", "threshold"]:
        print(f"\n{'='*70}")
        print(f"🧪 TEST: Confidence Scaling = {scaling_mode}")
        print(f"{'='*70}\n")
        
        # Erstelle Environment
        env = BettingEnvOptimized(
            df,
            use_betting_odds=True,
            confidence_threshold=0.60,
            bet_amount=10.0,
            max_bet_amount=30.0,
            min_gameday=4,
            apply_gameday_filter=True,
            use_kelly_criterion=True,
            kelly_fraction=0.25,
            confidence_scaling_mode=scaling_mode,  # ✅ NEU!
            debug_mode=False
        )
        
        # Test Episode
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < 20:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if step % 5 == 0:
                print(f"Step {step:3} | Action: {action} | Reward: {reward:>8.2f} | "
                      f"ROI: {info['roi']*100:>6.2f}% | Avg Bet: {info['avg_bet_size']:>5.2f}€")
            
            step += 1
        
        print(f"\n{'='*70}")
        print(f"FINAL STATS - Scaling: {scaling_mode}")
        print(f"{'='*70}")
        print(f"Total Bets:        {info['bets']}")
        print(f"Avg Bet Size:      {info['avg_bet_size']:.2f}€")
        print(f"Total Invested:    {info['total_invested']:.2f}€")
        print(f"Total Profit:      {info['total_profit']:.2f}€")
        print(f"ROI:               {info['roi']*100:.2f}%")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    test_environment()