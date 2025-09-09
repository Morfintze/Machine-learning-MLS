#!/usr/bin/env python3
"""
enhanced_ml_soccer_predictor.py
Uitgebreide versie met meer features: winst/verlies streaks, rustdagen, seizoenscontext, etc.
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------
# Enhanced Config
# -------------------------
EMA_SPAN = 7
SCALE_TO_SCORE = 4.0
MODEL_FILE = "enhanced_soccer_model.pkl"
SCALER_FILE = "enhanced_scaler.pkl"
WEIGHTS_FILE = "enhanced_dynamic_weights.pkl"

# Uitgebreide minimum standard deviations
MIN_STD_VALUES = {
    'xG90': 0.3, 'Sh90': 1.0, 'SoT90': 0.5, 'ShotQual': 0.05, 'ConvRatio90': 0.1,
    'Goals': 0.5, 'Prog90': 5.0, 'PrgDist90': 100.0, 'Att3rd90': 10.0,
    'Possession': 0.05, 'FieldTilt': 0.05, 'HighPress': 1.0, 'AerialMismatch': 5.0,
    'KeeperPSxGdiff': 0.2, 'TkldPct_possession': 0.05, 'WonPct_misc': 0.05,
    'Att_3rd_defense': 1.0, 'SavePct_keeper': 0.05,
    # Nieuwe features
    'WinStreak': 0.5, 'UnbeatenStreak': 0.5, 'LossStreak': 0.5,
    'WinRate5': 0.1, 'WinRate10': 0.1, 'PointsRate5': 0.1, 'PointsRate10': 0.1,
    'RestDays': 1.0, 'RecentForm': 0.2, 'GoalDifference': 0.5,
    'CleanSheetRate': 0.05, 'ScoringRate': 0.05, 'AvgGoalsFor': 0.2,
    'HomeAdvantage': 0.1, 'SeasonProgress': 0.1
}

# Uitgebreide initiele gewichten
WEIGHTS = {
    # Bestaande weights
    'xG90': 1.1, 'Sh90': 1.6, 'SoT90': 0.8, 'ShotQual': 1.5, 'ConvRatio90': 1.8,
    'Goals': 0.8, 'Prog90': 0.35, 'PrgDist90': 0.25, 'Att3rd90': 0.6,
    'FieldTilt': 0.8, 'HighPress': 0.95, 'AerialMismatch': 0.6, 'Possession': 0.35,
    'KeeperPSxGdiff': -0.44, 'GoalsAgainst': -2.481, 'TkldPct_possession': 0.4,
    'WonPct_misc': 0.4, 'Att_3rd_defense': 0.8, 'SavePct_keeper': 0.2,
    # Nieuwe weights
    'WinStreak': 1.2, 'UnbeatenStreak': 0.8, 'LossStreak': -1.0,
    'WinRate5': 1.5, 'WinRate10': 1.0, 'PointsRate5': 1.3, 'PointsRate10': 0.9,
    'RestDays': 0.3, 'RecentForm': 1.1, 'GoalDifference': 0.9,
    'CleanSheetRate': 0.7, 'ScoringRate': 0.8, 'AvgGoalsFor': 1.0,
    'HomeAdvantage': 0.5, 'SeasonProgress': 0.2
}

ML_WEIGHTS = WEIGHTS.copy()

# -------------------------
# Enhanced Helper Functions
# -------------------------
def choose_file(prompt):
    Tk().withdraw()
    path = askopenfilename(title=prompt)
    print(f"{prompt}: {path}")
    return path

def col_to_str(col):
    if isinstance(col, tuple):
        return " ".join(str(p) for p in col if p is not None).lower()
    return str(col).lower()

def series_to_numeric(series):
    def parse_val(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == '':
            return 0.0
        m = re.search(r'-?\d+\.?\d*', s)
        if m:
            return float(m.group(0))
        return 0.0
    return series.map(parse_val).astype(float)

def detect_header_rows(path, max_check=6, non_numeric_threshold=0.6):
    try:
        preview = pd.read_csv(path, header=None, nrows=max_check, low_memory=False)
    except:
        return 1
    header_n = 0
    for i in range(len(preview)):
        row = preview.iloc[i].astype(str).fillna('')
        cnt_non_numeric = sum(1 for v in row if v.strip()=='' or re.search(r'[A-Za-z]', v))
        if cnt_non_numeric / max(1, len(row)) >= non_numeric_threshold:
            header_n += 1
        else:
            break
    return max(1, header_n)

def find_column(df, keywords):
    """Zoek een kolom die alle keywords bevat (case-insensitive)."""
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keywords):
            return c
    return None

def find_column_flexible(df, keyword_groups):
    """Probeer meerdere keyword groepen om een kolom te vinden."""
    for keywords in keyword_groups:
        col = find_column(df, keywords)
        if col:
            return col
    return None

def parse_date(date_str):
    """Probeer verschillende datumformaten te parseren."""
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
        '%Y/%m/%d', '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    if len(date_str) >= 10:
        try:
            return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except ValueError:
            pass
    print(f"Kon datum niet parseren: {date_str}")
    return None

def extract_match_result_from_string(result_str, goals_for=None):
    """Extraheert resultaat uit result string."""
    result_str = str(result_str).upper().strip()
    
    if result_str in ['W', 'WIN', 'WON']:
        return 'W'
    elif result_str in ['D', 'DRAW', 'TIE']:
        return 'D'
    elif result_str in ['L', 'LOSS', 'LOST']:
        return 'L'
    else:
        # Probeer scoreformaten zoals "2-1"
        score_match = re.match(r'(\d+)\s*[-—:\u2013]\s*(\d+)', result_str)
        if score_match:
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))
            
            if goals_for is not None:
                # Bepaal welke score bij dit team hoort
                if abs(goals_for - home_goals) < 1e-6:
                    our_goals, opp_goals = home_goals, away_goals
                else:
                    our_goals, opp_goals = away_goals, home_goals
            else:
                # Aanname: eerste getal is voor dit team
                our_goals, opp_goals = home_goals, away_goals
            
            if our_goals > opp_goals:
                return 'W'
            elif our_goals == opp_goals:
                return 'D'
            else:
                return 'L'
    
    return None

# -------------------------
# Enhanced Feature Engineering
# -------------------------
def calculate_historical_features(df):
    """Berekent historische prestatie-features."""
    features = {}
    n = len(df)
    
    if n == 0:
        # Return default values for empty dataframe
        default_features = {
            'WinStreak': 0, 'UnbeatenStreak': 0, 'LossStreak': 0,
            'WinRate5': 0, 'WinRate10': 0, 'PointsRate5': 0, 'PointsRate10': 0,
            'RestDays': 7, 'RecentForm': 0, 'GoalDifference': 0,
            'CleanSheetRate': 0, 'ScoringRate': 0, 'AvgGoalsFor': 0
        }
        return {k: pd.Series([v]) for k, v in default_features.items()}
    
    # Zorg ervoor dat df gesorteerd is op datum
    date_col = find_column(df, ['date'])
    if date_col:
        df = df.copy()
        df['parsed_date'] = df[date_col].apply(parse_date)
        df = df.sort_values('parsed_date').reset_index(drop=True)
    
    # Extract results en goals
    result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
    goals_col = find_column_flexible(df, [['gf', 'shooting'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    ga_col = find_column_flexible(df, [['ga', 'shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_']])
    
    results = []
    goals_for = []
    goals_against = []
    
    for i in range(n):
        # Extract result
        if result_col:
            result_str = df[result_col].iloc[i]
            goals_for_val = series_to_numeric(df[goals_col]).iloc[i] if goals_col else None
            result = extract_match_result_from_string(result_str, goals_for_val)
            results.append(result)
        else:
            results.append(None)
        
        # Extract goals
        goals_for.append(series_to_numeric(df[goals_col]).iloc[i] if goals_col else 0)
        goals_against.append(series_to_numeric(df[ga_col]).iloc[i] if ga_col else 0)
    
    # Calculate features for each match (rolling)
    win_streaks = []
    unbeaten_streaks = []
    loss_streaks = []
    win_rates_5 = []
    win_rates_10 = []
    points_rates_5 = []
    points_rates_10 = []
    recent_forms = []
    goal_diffs = []
    clean_sheet_rates = []
    scoring_rates = []
    avg_goals_for = []
    rest_days = []
    
    for i in range(n):
        # Current streak calculations (looking backwards from position i)
        current_streak_w = 0
        current_streak_u = 0
        current_streak_l = 0
        
        # Count streaks going backwards
        for j in range(i-1, -1, -1):
            if results[j] == 'W':
                if current_streak_w == 0 and current_streak_l == 0:
                    current_streak_w += 1
                    current_streak_u += 1
                elif current_streak_w > 0 or current_streak_u > 0:
                    current_streak_w += 1
                    current_streak_u += 1
                else:
                    break
            elif results[j] == 'D':
                if current_streak_l == 0:
                    current_streak_u += 1
                    if current_streak_w > 0:
                        current_streak_w = 0
                else:
                    break
            elif results[j] == 'L':
                if current_streak_w == 0 and current_streak_u == 0:
                    current_streak_l += 1
                else:
                    break
            else:
                break  # Unknown result
        
        win_streaks.append(current_streak_w)
        unbeaten_streaks.append(current_streak_u)
        loss_streaks.append(current_streak_l)
        
        # Win rates and points rates
        last_5_results = results[max(0, i-4):i+1] if i >= 0 else []
        last_10_results = results[max(0, i-9):i+1] if i >= 0 else []
        
        wins_5 = sum(1 for r in last_5_results if r == 'W')
        wins_10 = sum(1 for r in last_10_results if r == 'W')
        
        win_rates_5.append(wins_5 / max(1, len(last_5_results)))
        win_rates_10.append(wins_10 / max(1, len(last_10_results)))
        
        # Points (3 for win, 1 for draw, 0 for loss)
        points_5 = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in last_5_results)
        points_10 = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in last_10_results)
        
        points_rates_5.append(points_5 / max(1, len(last_5_results) * 3))
        points_rates_10.append(points_10 / max(1, len(last_10_results) * 3))
        
        # Recent form (weighted recent results)
        weights = [0.4, 0.3, 0.2, 0.1]  # More weight to recent games
        form_score = 0
        for j, r in enumerate(reversed(last_5_results[-4:])):
            if j < len(weights):
                if r == 'W':
                    form_score += 3 * weights[j]
                elif r == 'D':
                    form_score += 1 * weights[j]
        recent_forms.append(form_score)
        
        # Goal difference (only for the last 7 matches, not cumulative)
        start_idx = max(0, i - 6)  # Last 7 matches including current
        gf_sum = sum(goals_for[start_idx:i+1])
        ga_sum = sum(goals_against[start_idx:i+1])
        goal_diffs.append(gf_sum - ga_sum)
        
        # Clean sheet rate (only for the last 7 matches)
        clean_sheets = sum(1 for j in range(start_idx, i+1) if goals_against[j] == 0)
        clean_sheet_rates.append(clean_sheets / max(1, i+1 - start_idx))
        
        # Scoring rate (games with goals, only for the last 7 matches)
        scoring_games = sum(1 for j in range(start_idx, i+1) if goals_for[j] > 0)
        scoring_rates.append(scoring_games / max(1, i+1 - start_idx))
        
        # Average goals for (only for the last 7 matches)
        avg_goals_for.append(sum(goals_for[start_idx:i+1]) / max(1, i+1 - start_idx))
        
        # Rest days
        if date_col and i > 0:
            current_date = df['parsed_date'].iloc[i]
            previous_date = df['parsed_date'].iloc[i-1]
            if pd.notna(current_date) and pd.notna(previous_date):
                rest_days.append((current_date - previous_date).days)
            else:
                rest_days.append(7)  # Default 1 week
        else:
            rest_days.append(7)  # Default for first game
    
    # Convert to pandas Series
    features['WinStreak'] = pd.Series(win_streaks)
    features['UnbeatenStreak'] = pd.Series(unbeaten_streaks)
    features['LossStreak'] = pd.Series(loss_streaks)
    features['WinRate5'] = pd.Series(win_rates_5)
    features['WinRate10'] = pd.Series(win_rates_10)
    features['PointsRate5'] = pd.Series(points_rates_5)
    features['PointsRate10'] = pd.Series(points_rates_10)
    features['RestDays'] = pd.Series(rest_days)
    features['RecentForm'] = pd.Series(recent_forms)
    features['GoalDifference'] = pd.Series(goal_diffs)
    features['CleanSheetRate'] = pd.Series(clean_sheet_rates)
    features['ScoringRate'] = pd.Series(scoring_rates)
    features['AvgGoalsFor'] = pd.Series(avg_goals_for)
    
    return features

def calculate_seasonal_context_features(df):
    """Berekent seizoenscontext features."""
    features = {}
    n = len(df)
    
    if n == 0:
        return {
            'SeasonProgress': pd.Series([0.0]),
            'HomeAdvantage': pd.Series([0.0])
        }
    
    # Season progress (assumes ~34 game season)
    season_progress = [(i + 1) / 34.0 for i in range(n)]
    features['SeasonProgress'] = pd.Series(season_progress)
    
    # Home advantage (recent home performance vs away)
    venue_col = find_column(df, ['venue'])
    if venue_col:
        home_results = []
        for i in range(n):
            venue = str(df[venue_col].iloc[i]).lower()
            is_home = 'home' in venue
            
            # Calculate home advantage based on recent venue performance
            home_points = 0
            away_points = 0
            home_games = 0
            away_games = 0
            
            # Look at last 10 games
            start_idx = max(0, i - 9)
            for j in range(start_idx, i + 1):
                venue_j = str(df[venue_col].iloc[j]).lower()
                result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
                
                if result_col:
                    result = extract_match_result_from_string(df[result_col].iloc[j])
                    points = 3 if result == 'W' else 1 if result == 'D' else 0
                    
                    if 'home' in venue_j:
                        home_points += points
                        home_games += 1
                    elif 'away' in venue_j:
                        away_points += points
                        away_games += 1
            
            home_avg = home_points / max(1, home_games)
            away_avg = away_points / max(1, away_games)
            home_advantage = (home_avg - away_avg) / 3.0  # Normalize to 0-1 scale
            
            home_results.append(home_advantage if is_home else -home_advantage)
        
        features['HomeAdvantage'] = pd.Series(home_results)
    else:
        features['HomeAdvantage'] = pd.Series([0.0] * n)
    
    return features

def apply_venue_filter(df, venue='home'):
    """Enhanced venue filter with better date handling."""
    date_col = find_column(df, ['date'])
    if date_col is None:
        print("Geen datumkolom gevonden! Kan niet sorteren op datum.")
        return df
        
    df['parsed_date'] = df[date_col].apply(parse_date)
    original_len = len(df)
    df = df[df['parsed_date'].notna()].copy()
    
    if len(df) < original_len:
        print(f"{original_len - len(df)} wedstrijden verwijderd vanwege ongeldige datums.")
    
    df.sort_values('parsed_date', ascending=True, inplace=True)
    
    venue_col = find_column(df, ['venue'])
    if venue_col is None:
        print("Geen Venue-kolom gevonden!")
        return df
        
    df[venue_col] = df[venue_col].astype(str).str.lower()
    mapped = df[venue_col].map(lambda x: 'home' if 'home' in x else ('away' if 'away' in x else 'other'))
    df['mapped_venue'] = mapped
    
    filtered = df[df['mapped_venue'] == venue].reset_index(drop=True)
    
    if len(filtered) > EMA_SPAN:
        filtered = filtered.tail(EMA_SPAN).reset_index(drop=True)
    
    print(f"{venue.capitalize()}-team: {len(filtered)}/{len(df)} wedstrijden na filter (Venue == {venue}).")
    
    if len(filtered) > 0:
        print(f"Datumbereik: {filtered['parsed_date'].min().strftime('%Y-%m-%d')} tot {filtered['parsed_date'].max().strftime('%Y-%m-%d')}")
    
    return filtered

def build_enhanced_feature_series(df, team_name):
    """Enhanced feature building with all new features."""
    n = len(df)
    feats = {}
    minutes = pd.Series([90.0] * n)

    # Verbeterde kolomdetectie met meer specifieke zoekpatronen
    xg_col = find_column_flexible(df, [['expected_xg_shooting'], ['xg_shooting'], ['xg'], ['npxg_shooting'], ['npxg']])
    sh_col = find_column_flexible(df, [['standard_sh_shooting'], ['sh_shooting'], ['sh'], ['shots'], ['total', 'shots']])
    sot_col = find_column_flexible(df, [['standard_sot_shooting'], ['sot_shooting'], ['sot'], ['shots on target'], ['on target']])
    goals_col = find_column_flexible(df, [['standard_gls_shooting'], ['gls_shooting'], ['gf_shooting'], ['gf'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    
    # Progressive passing - verbeterde detectie
    prgp_col = find_column_flexible(df, [['prgp_passing'], ['prgr_possession'], ['prgp'], ['progressive', 'passes']])
    prgdist_col = find_column_flexible(df, [['total_prgdist_passing'], ['prgdist_possession'], ['prgdist'], ['progressive', 'distance'], ['pass', 'prgdist']])
    
    # Set pieces - focus op corners
    setpieces_col = find_column_flexible(df, [['pass types_ck_passing_types'], ['ck_passing_types'], ['corner', 'kicks'], ['corners'], ['ck']])
    
    # Attacking third touches
    att3rd_col = find_column_flexible(df, [['touches_att 3rd_possession'], ['touches_att_3rd_possession'], ['att_3rd_possession'], ['att_3rd'], ['att', '3rd']])
    
    # Possession - verbeterde detectie
    poss_col = find_column_flexible(df, [['poss_possession'], ['possession'], ['poss', '%'], ['possession', '%']])
    
    # Goalkeeper features - verbeterde detectie
    sota_col = find_column_flexible(df, [['performance_sota_keeper'], ['sota_keeper'], ['sota'], ['shots on target against']])
    saves_col = find_column_flexible(df, [['performance_saves_keeper'], ['saves_keeper'], ['saves']])
    psxg_col = find_column_flexible(df, [['performance_psxg_keeper'], ['psxg_keeper'], ['psxg']])
    
    # Aerial duels - verbeterde detectie
    aerial_win_col = find_column_flexible(df, [
        ['aerial duels_won%_misc'], 
        ['won%_misc'], 
        ['aerial', 'won%'], 
        ['aerial', '%'], 
        ['duel', 'won%'],
        ['aerial', 'won', '%'],
        ['aerial', 'success', '%'],
        ['aerial', 'duels', '%']
    ])
    
    # Defensive actions
    def3rd_col = find_column_flexible(df, [['tackles_def 3rd_defense'], ['def_3rd_defense'], ['defensive', '3rd']])
    int_col = find_column_flexible(df, [['performance_int_misc'], ['int_misc'], ['interceptions'], ['int']])
    
    # Tackled percentage
    tkldpct_col = find_column_flexible(df, [['tkldpct_possession'], ['tkld%_possession'], ['tackled', '%']])
    
    # Won percentage
    wonpct_col = find_column_flexible(df, [['won%_misc'], ['duels', 'won%'], ['won', '%']])
    
    # Attacking third defense
    att3rddef_col = find_column_flexible(df, [['tackles_att 3rd_defense'], ['att_3rd_defense'], ['attacking', '3rd', 'tackles']])
    
    # Save percentage - verbeterde detectie
    savepct_col = find_column_flexible(df, [['save%_keeper'], ['savepct_keeper'], ['save', '%'], ['performance_save%_keeper']])

    # Basic features (per 90 minutes)
    def per90(col):
        return series_to_numeric(df[col]) / minutes * 90.0 if col else pd.Series([0.0] * n)
    
    feats['xG90'] = per90(xg_col)
    feats['Sh90'] = per90(sh_col)
    feats['SoT90'] = per90(sot_col)

    # Shot quality calculation
    sh_safe = series_to_numeric(df[sh_col].replace(0, np.nan)) if sh_col else pd.Series([1.0] * n)
    feats['ShotQual'] = 0.6 * (series_to_numeric(df[xg_col]) / sh_safe).fillna(0.0) + 0.4 * (series_to_numeric(df[sot_col]) / sh_safe).fillna(0.0)
    feats['Goals'] = series_to_numeric(df[goals_col]).fillna(0.0) if goals_col else pd.Series([0.0] * n)

    # Goals against calculation
    GA_WEIGHT = 0.25
    ga_col = find_column_flexible(df, [['standard_ga_shooting'], ['ga_shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_'], ['ga']])
    if ga_col:
        feats['GoalsAgainst'] = series_to_numeric(df[ga_col]).fillna(0.0)
    else:
        # Fallback: try to extract from result column
        res_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
        parsed_ga = []
        if res_col:
            goals_series = series_to_numeric(df[goals_col]).fillna(np.nan) if goals_col else pd.Series([np.nan] * n, index=df.index)
            for idx_row, row in df.iterrows():
                res_val = row.get(res_col, '') if isinstance(row, dict) or isinstance(row, pd.Series) else ''
                if pd.isna(res_val): 
                    res_val = ''
                res_str = str(res_val)
                mres = re.search(r'(\d+)\s*[-—:\u2013]\s*(\d+)', res_str)
                if mres:
                    a = float(mres.group(1))
                    b = float(mres.group(2))
                    if not np.isnan(goals_series.iloc[idx_row]):
                        g = goals_series.iloc[idx_row]
                        if abs(g - a) < 1e-6:
                            ga_val = b
                        elif abs(g - b) < 1e-6:
                            ga_val = a
                        else:
                            ga_val = b
                    else:
                        ga_val = b
                    parsed_ga.append(ga_val)
                else:
                    parsed_ga.append(0.0)
            feats['GoalsAgainst'] = pd.Series(parsed_ga, index=df.index)
        else:
            feats['GoalsAgainst'] = pd.Series([0.0] * n, index=df.index)

    feats['GoalsAgainstWeighted'] = feats['GoalsAgainst'] * GA_WEIGHT
    feats['ConvRatio90'] = (series_to_numeric(df[goals_col]) / series_to_numeric(df[sot_col])).fillna(0.0) if goals_col and sot_col else pd.Series([0.0] * n)
    feats['Prog90'] = per90(prgp_col)
    feats['PrgDist90'] = per90(prgdist_col)
    feats['SetPieces90'] = per90(setpieces_col)
    feats['Att3rd90'] = per90(att3rd_col)

    # Possession handling
    poss_raw = series_to_numeric(df[poss_col]) if poss_col else pd.Series([0.0] * n)
    if poss_raw.max() > 1.5:  # Likely percentage (0-100)
        feats['Possession'] = poss_raw / 100.0
    else:  # Already decimal (0-1)
        feats['Possession'] = poss_raw

    feats['SoTA90'] = per90(sota_col)
    feats['SaveRate'] = (series_to_numeric(df[saves_col]) / series_to_numeric(df[sota_col])).fillna(0.0) if saves_col and sota_col else pd.Series([0.0] * n)
    feats['PSxG'] = series_to_numeric(df[psxg_col]).fillna(0.0) if psxg_col else pd.Series([0.0] * n)

    # Aerial win percentage handling
    if aerial_win_col:
        aerial_raw = series_to_numeric(df[aerial_win_col])
        # Check if values are percentages (typically 0-100) or decimals (0-1)
        if aerial_raw.max() > 1.5:  # Likely percentage
            feats['AerialWin%'] = aerial_raw / 100.0
        else:  # Already decimal
            feats['AerialWin%'] = aerial_raw
    else:
                # Als aerial win percentage niet wordt gevonden, probeer dan wonpct_col als fallback
        if wonpct_col:
            won_raw = series_to_numeric(df[wonpct_col])
            if won_raw.max() > 1.5:  # Likely percentage
                feats['AerialWin%'] = won_raw / 100.0
            else:  # Already decimal
                feats['AerialWin%'] = won_raw
        else:
            feats['AerialWin%'] = pd.Series([0.0] * n)

    # High press - use defensive actions in attacking third
    feats['HighPress'] = per90(att3rddef_col) if att3rddef_col else pd.Series([0.0] * n)

    # Additional basic features with percentage handling
    if tkldpct_col:
        tkld_raw = series_to_numeric(df[tkldpct_col])
        feats['TkldPct_possession'] = tkld_raw / 100.0 if tkld_raw.max() > 1.5 else tkld_raw
    else:
        feats['TkldPct_possession'] = pd.Series([0.0] * n)

    if wonpct_col:
        won_raw = series_to_numeric(df[wonpct_col])
        feats['WonPct_misc'] = won_raw / 100.0 if won_raw.max() > 1.5 else won_raw
    else:
        feats['WonPct_misc'] = pd.Series([0.0] * n)

    feats['Att_3rd_defense'] = per90(att3rddef_col) if att3rddef_col else pd.Series([0.0] * n)

    if savepct_col:
        save_raw = series_to_numeric(df[savepct_col])
        feats['SavePct_keeper'] = save_raw / 100.0 if save_raw.max() > 1.5 else save_raw
    else:
        feats['SavePct_keeper'] = pd.Series([0.0] * n)

    # Add historical performance features
    historical_feats = calculate_historical_features(df)
    feats.update(historical_feats)
    
    # Add seasonal context features
    seasonal_feats = calculate_seasonal_context_features(df)
    feats.update(seasonal_feats)

    return feats

# -------------------------
# Enhanced EMA and Delta Functions
# -------------------------
def ema(series, span=EMA_SPAN):
    """Enhanced EMA with better handling of edge cases."""
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return series.iloc[0] if hasattr(series, 'iloc') else series[0]
    return pd.Series(series).ewm(span=span, adjust=False).mean().iloc[-1]

def make_enhanced_delta(team_feats, opp_feats):
    """Enhanced delta calculation with all new features."""
    delta = {}
    
    # Basic features
    keys = [
        'xG90', 'Sh90', 'SoT90', 'ShotQual', 'ConvRatio90', 'Goals', 'GoalsAgainst',
        'Prog90', 'PrgDist90', 'SetPieces90', 'Att3rd90', 'Possession',
        'TkldPct_possession', 'WonPct_misc', 'Att_3rd_defense', 'SavePct_keeper',
        # New historical features
        'WinStreak', 'UnbeatenStreak', 'LossStreak', 'WinRate5', 'WinRate10', 
        'PointsRate5', 'PointsRate10', 'RestDays', 'RecentForm', 'GoalDifference',
        'CleanSheetRate', 'ScoringRate', 'AvgGoalsFor',
        'SeasonProgress', 'HomeAdvantage'
    ]
    
    for k in keys:
        t = team_feats.get(k, pd.Series([0.0]))
        o = opp_feats.get(k, pd.Series([0.0]))
        delta[k] = (ema(t), ema(o))

    # Calculated features
    t_att = team_feats.get('Att3rd90', pd.Series([0.0]))
    o_att = opp_feats.get('Att3rd90', pd.Series([0.0]))
    t_att_val = ema(t_att)
    o_att_val = ema(o_att)
    
    t_tilt = t_att_val / (t_att_val + o_att_val) if (t_att_val + o_att_val) > 0 else 0.0
    o_tilt = o_att_val / (o_att_val + t_att_val) if (o_att_val + t_att_val) > 0 else 0.0
    delta['FieldTilt'] = (t_tilt, o_tilt)
    
    delta['HighPress'] = (ema(team_feats.get('HighPress', pd.Series([0.0]))), 
                         ema(opp_feats.get('HighPress', pd.Series([0.0]))))
    delta['AerialMismatch'] = (ema(team_feats.get('AerialWin%', pd.Series([0.0]))), 
                              ema(opp_feats.get('AerialWin%', pd.Series([0.0]))))
    delta['KeeperPSxGdiff'] = (ema(team_feats.get('PSxG', pd.Series([0.0]))), 
                              ema(opp_feats.get('PSxG', pd.Series([0.0]))))

    return delta

def compute_enhanced_weighted_score(delta_dict, use_ml_weights=False):
    """Enhanced scoring with improved z-score calculation and feature weighting."""
    z_team = {}
    z_opp = {}
    contribs = {}

    print(f"\n--- Enhanced Z-Score Calculation ---")
    print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Diff':<8} | {'Std':<8} | {'Z-Diff':<8} | {'Weight':<8} | {'Contrib':<8}")
    print("-" * 100)

    # Use ML weights if available
    weights_to_use = ML_WEIGHTS if use_ml_weights else WEIGHTS

    for feat, (t_ema, o_ema) in delta_dict.items():
        min_std = MIN_STD_VALUES.get(feat, max(0.1, abs(t_ema) * 0.1))
        combined = np.array([t_ema, o_ema])
        mean, std = np.nanmean(combined), np.nanstd(combined)
        
        # Enhanced robust_std calculation with feature-specific minimums
        robust_std = max(std, min_std, abs(mean) * 0.15, 0.3)
        
        zt = (t_ema - mean) / robust_std
        zo = (o_ema - mean) / robust_std
        
        # Adaptive clipping based on feature type
        if feat in ['WinStreak', 'UnbeatenStreak', 'LossStreak', 'RestDays']:
            # More tolerance for streak/contextual features
            zt = np.clip(zt, -3.0, 3.0)
            zo = np.clip(zo, -3.0, 3.0)
        else:
            # Standard clipping for performance features
            zt = np.clip(zt, -2.5, 2.5)
            zo = np.clip(zo, -2.5, 2.5)
        
        z_team[feat] = zt
        z_opp[feat] = zo
        
        weight = weights_to_use.get(feat, 0.0)
        
        # Enhanced contribution calculation with feature-specific scaling
        raw_contrib = weight * (zt - zo)
        
        # Feature-specific scaling factors
        if feat in ['WinRate5', 'WinRate10', 'PointsRate5', 'PointsRate10', 'RecentForm']:
            # Recent performance features get enhanced impact
            scaling_factor = 1.2
        elif feat in ['RestDays', 'SeasonProgress']:
            # Contextual features get reduced impact
            scaling_factor = 0.8
        else:
            scaling_factor = 1.0
        
        # Apply scaling and soft clipping
        contribs[feat] = raw_contrib * scaling_factor * np.tanh(abs(raw_contrib) / 4.0) / max(abs(raw_contrib), 1e-6)
        
        diff = t_ema - o_ema
        z_diff = zt - zo
        
        print(f"{feat:<20} | {t_ema:<10.3f} | {o_ema:<10.3f} | {diff:<+8.3f} | {robust_std:<8.3f} | {z_diff:<+8.3f} | {weight:<8.3f} | {contribs[feat]:<+8.3f}")

    # Special handling for certain features
    if 'KeeperPSxGdiff' in contribs:
        contribs['KeeperPSxGdiff'] = -contribs['KeeperPSxGdiff']

    weighted_diff = sum(contribs.values())
    
    # Enhanced scaling with adaptive range
    max_expected_diff = 15.0  # Increased to account for more features
    scaled_diff = max_expected_diff * np.tanh(weighted_diff / max_expected_diff)
    
    final = 50.0 + SCALE_TO_SCORE * scaled_diff
    
    # Dynamic capping based on confidence
    confidence = min(1.0, len([c for c in contribs.values() if abs(c) > 0.1]) / 10.0)
    cap_range = 30 + confidence * 25  # Range from 30-55 to 30-80 based on confidence
    
    final = np.clip(final, 30.0, 70.0 + cap_range)
    
    return final, scaled_diff, z_team, z_opp, contribs

# -------------------------
# Enhanced ML Functions
# -------------------------
def extract_match_result_enhanced(df):
    """Enhanced result extraction with better error handling."""
    result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
    
    if result_col is None:
        print("Geen resultaatkolom gevonden!")
        return None
    
    if len(df) == 0:
        return None
        
    result = df[result_col].iloc[-1]
    
    if result is None or pd.isna(result):
        return None
    
    # Get goals for context
    goals_col = find_column_flexible(df, [['gf_shooting'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    goals_for = series_to_numeric(df[goals_col]).iloc[-1] if goals_col and len(df) > 0 else None
    
    return extract_match_result_from_string(result, goals_for)

def prepare_enhanced_ml_features(home_feats, away_feats):
    """Enhanced ML feature preparation with all new features."""
    # Calculate EMA for all features
    home_ema = {f"home_{k}": ema(v) for k, v in home_feats.items()}
    away_ema = {f"away_{k}": ema(v) for k, v in away_feats.items()}
    
    # Combine features
    combined_features = {**home_ema, **away_ema}
    
    # Add difference and ratio features
    for key in home_feats.keys():
        if key in away_feats:
            home_val = home_ema.get(f"home_{key}", 0)
            away_val = away_ema.get(f"away_{key}", 0)
            combined_features[f"diff_{key}"] = home_val - away_val
            
            # Avoid division by zero for ratios
            if abs(away_val) > 1e-6:
                combined_features[f"ratio_{key}"] = home_val / away_val
            else:
                combined_features[f"ratio_{key}"] = home_val if abs(home_val) > 1e-6 else 1.0
    
    # Add interaction features for key metrics
    key_interactions = [
        ('WinRate5', 'RecentForm'),
        ('GoalDifference', 'AvgGoalsFor'),
        ('CleanSheetRate', 'SavePct_keeper'),
        ('RestDays', 'WinStreak')
    ]
    
    for feat1, feat2 in key_interactions:
        home_interaction = home_ema.get(f"home_{feat1}", 0) * home_ema.get(f"home_{feat2}", 0)
        away_interaction = away_ema.get(f"away_{feat1}", 0) * away_ema.get(f"away_{feat2}", 0)
        combined_features[f"interaction_home_{feat1}_{feat2}"] = home_interaction
        combined_features[f"interaction_away_{feat1}_{feat2}"] = away_interaction
        combined_features[f"interaction_diff_{feat1}_{feat2}"] = home_interaction - away_interaction
    
    return combined_features

def train_enhanced_model(X, y):
    """Enhanced model training with better hyperparameters."""
    if len(X) < 10:
        print(f"Onvoldoende data voor training: {len(X)} samples. Minimum 10 vereist.")
        return None, None
    
    # Split the data
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Enhanced Random Forest with better parameters
    model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,      # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Enhanced Model Nauwkeurigheid: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("\nClassificatie Rapport:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_names = X.columns.tolist()
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Belangrijkste Features:")
    print(feature_importance_df.head(15))
    
    # Update weights
    update_enhanced_weights_from_model(model, feature_names)
    
    # Save model and scaler
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Enhanced model opgeslagen als {MODEL_FILE}")
    print(f"Enhanced scaler opgeslagen als {SCALER_FILE}")
    
    return model, scaler

def update_enhanced_weights_from_model(model, feature_names):
    """Enhanced weight update with better feature mapping."""
    global ML_WEIGHTS
    
    if not hasattr(model, 'feature_importances_'):
        return
    
    feature_importance = model.feature_importances_
    
    # Create importance mapping
    importance_map = {}
    for i, feature_name in enumerate(feature_names):
        # Extract base feature name
        base_feature = None
        
        # Handle different feature name patterns
        for feat in ML_WEIGHTS.keys():
            patterns = [
                f"diff_{feat}",
                f"ratio_{feat}", 
                f"home_{feat}",
                f"away_{feat}",
                f"interaction_home_{feat}_",
                f"interaction_away_{feat}_",
                f"interaction_diff_{feat}_"
            ]
            
            if any(pattern in feature_name for pattern in patterns):
                base_feature = feat
                break
        
        if base_feature:
            if base_feature not in importance_map:
                importance_map[base_feature] = []
            importance_map[base_feature].append(feature_importance[i])
    
    # Update weights with combined importance
    for base_feature, importances in importance_map.items():
        if base_feature in ML_WEIGHTS:
            avg_importance = np.mean(importances)
            max_importance = max(feature_importance)
            
            if max_importance > 0:
                normalized_importance = avg_importance / max_importance
                
                # Combine with existing weight (60% ML, 40% expert knowledge)
                ML_WEIGHTS[base_feature] = (
                    0.6 * normalized_importance * 2.0 +  # Scale ML importance
                    0.4 * WEIGHTS[base_feature]
                )
    
    # Save updated weights
    with open(WEIGHTS_FILE, 'wb') as f:
        pickle.dump(ML_WEIGHTS, f)
    
    print("Enhanced gewichten bijgewerkt en opgeslagen")

def load_enhanced_model():
    """Enhanced model loading with better error handling."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
            
        # Try to load enhanced weights
        try:
            with open(WEIGHTS_FILE, 'rb') as f:
                global ML_WEIGHTS
                ML_WEIGHTS = pickle.load(f)
            print("Enhanced model, scaler en gewichten succesvol geladen")
        except FileNotFoundError:
            print("Enhanced model en scaler geladen, default gewichten gebruikt")
        
        return model, scaler
    except FileNotFoundError as e:
        print(f"Enhanced model niet gevonden: {e}")
        return None, None

def enhanced_combined_prediction(ml_prediction, ml_probabilities, statistical_score):
    """
    Enhanced prediction combining with better statistical score tempering
    and more sophisticated probability combination.
    """
    
    def enhanced_temper_score(score):
        """More sophisticated score tempering."""
        if score < 35:
            return 35 + (score - 35) * 0.2
        elif score > 75:
            return 75 + (score - 75) * 0.3
        elif score < 42:
            return 42 + (score - 42) * 0.5
        elif score > 68:
            return 68 + (score - 68) * 0.6
        else:
            return score
    
    tempered_score = enhanced_temper_score(statistical_score)
    
    # Convert to probabilities with more nuanced mapping
    if tempered_score > 50:
        stat_win_prob = min(0.75, max(0.15, (tempered_score - 45) / 20))
        stat_loss_prob = max(0.05, min(0.4, (55 - tempered_score) / 25))
    else:
        stat_win_prob = max(0.05, min(0.4, (tempered_score - 35) / 25))
        stat_loss_prob = min(0.75, max(0.15, (55 - tempered_score) / 20))
    
    stat_draw_prob = 1.0 - stat_win_prob - stat_loss_prob
    stat_draw_prob = max(0.15, stat_draw_prob)  # Minimum draw probability
    
    # Renormalize
    total = stat_win_prob + stat_draw_prob + stat_loss_prob
    if total > 0:
        stat_win_prob /= total
        stat_draw_prob /= total
        stat_loss_prob /= total
    
    # Dynamic weighting based on score reliability and ML confidence
    ml_confidence = max(ml_probabilities) - np.mean(ml_probabilities)
    score_extremity = abs(statistical_score - 50) / 25.0
    
    # More ML weight when ML is confident and score is not extreme
    if ml_confidence > 0.2 and score_extremity < 0.6:
        ml_weight = 0.7
        stat_weight = 0.3
    elif score_extremity > 0.8:  # Very extreme scores
        ml_weight = 0.8
        stat_weight = 0.2
    elif ml_confidence < 0.1:  # ML not confident
        ml_weight = 0.5
        stat_weight = 0.5
    else:
        ml_weight = 0.65
        stat_weight = 0.35
    
    # Combine probabilities
    combined_probs = (
        ml_weight * ml_probabilities[0] + stat_weight * stat_win_prob,
        ml_weight * ml_probabilities[1] + stat_weight * stat_draw_prob,  
        ml_weight * ml_probabilities[2] + stat_weight * stat_loss_prob
    )
    
    # Final prediction
    final_prediction = ['win', 'draw', 'loss'][np.argmax(combined_probs)]
    
    return final_prediction, combined_probs, tempered_score

# -------------------------
# Enhanced Main Function
# -------------------------
if __name__ == '__main__':
    print("Enhanced ML Voetbalwedstrijdvoorspeller v2.0")
    print("=" * 55)
    
    print("\nKies een optie:")
    print("1. Train een nieuw enhanced model")
    print("2. Voorspelling met enhanced model") 
    print("3. Traditionele score met enhanced features")
    print("4. Batch analyse van meerdere wedstrijden")
    
    choice = input("Jouw keuze (1-4): ").strip()
    
    if choice == "1":
        print("\nTraining Enhanced Model")
        print("Upload CSV-bestanden met uitgebreide historische data")
        
        training_files = []
        while True:
            file_path = choose_file('Upload CSV voor training (leeg = stoppen)')
            if not file_path:
                break
            training_files.append(file_path)
        
        if not training_files:
            print("Geen bestanden geselecteerd.")
            exit()
        
        # Collect training data
        X_data = []
        y_data = []
        
        print("\nVerwerken van trainingsdata...")
        for file_idx, file_path in enumerate(training_files):
            print(f"\nBestand {file_idx + 1}/{len(training_files)}: {file_path}")
            
            h_rows = detect_header_rows(file_path)
            df = pd.read_csv(file_path, header=list(range(h_rows)) if h_rows > 1 else 0, low_memory=False)
            
            print(f"  Geladen: {len(df)} wedstrijden")
            
            # Process each match with sufficient history
            for i in range(max(5, EMA_SPAN), len(df)):  # Need minimum history
                try:
                    # Create subsets
                    home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                    away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                    
                    if len(home_subset) >= 3 and len(away_subset) >= 3:  # Minimum matches
                        # Build enhanced features
                        home_feats = build_enhanced_feature_series(home_subset, "HOME")
                        away_feats = build_enhanced_feature_series(away_subset, "AWAY")
                        
                        # Prepare ML features
                        features = prepare_enhanced_ml_features(home_feats, away_feats)
                        
                        # Extract result
                        result = extract_match_result_enhanced(df.iloc[:i+1])
                        
                        if result and features:
                            X_data.append(features)
                            y_data.append(result)
                            
                except Exception as e:
                    print(f"  Fout bij wedstrijd {i}: {e}")
                    continue
        
        print(f"\nTotaal verzamelde samples: {len(X_data)}")
        
        if len(X_data) < 10:
            print("Onvoldoende trainingsdata verzameld.")
            exit()
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        print(f"Feature dimensies: {X_df.shape}")
        print(f"Label distributie:")
        print(y_series.value_counts())
        
        # Train enhanced model
        model, scaler = train_enhanced_model(X_df, y_series)
        
        if model is not None:
            print("\nEnhanced model training voltooid!")
        
    elif choice == "2":
        print("\nEnhanced Voorspelling")
        
        # Load enhanced model
        model, scaler = load_enhanced_model()
        if model is None:
            print("Geen enhanced model gevonden. Train eerst een model.")
            exit()
        
        # Get team data
        home_file = choose_file('Upload THUIS team CSV')
        away_file = choose_file('Upload UIT team CSV')

        # Load and process
        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)

        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home > 1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away > 1 else 0, low_memory=False)

        # Filter and build features
        home_df = apply_venue_filter(home_df, 'home')
        away_df = apply_venue_filter(away_df, 'away')

        home_feats = build_enhanced_feature_series(home_df, "HOME TEAM")
        away_feats = build_enhanced_feature_series(away_df, "AWAY TEAM")

        # ML prediction
        features = prepare_enhanced_ml_features(home_feats, away_feats)
        features_df = pd.DataFrame([features])
        features_scaled = scaler.transform(features_df)
        
        ml_prediction = model.predict(features_scaled)[0]
        ml_probabilities = model.predict_proba(features_scaled)[0]
        
        # Enhanced statistical score
        delta = make_enhanced_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs = compute_enhanced_weighted_score(delta, use_ml_weights=True)
        
        # Combined prediction
        final_prediction, combined_probs, tempered_score = enhanced_combined_prediction(
            ml_prediction, ml_probabilities, final_score)

        # Results
        print(f"\n=== ENHANCED VOORSPELLING ===")
        print(f"ML Voorspelling: {ml_prediction}")
        print(f"ML Kansen: Win={ml_probabilities[0]:.3f}, Draw={ml_probabilities[1]:.3f}, Loss={ml_probabilities[2]:.3f}")
        print(f"Statistische Score (raw): {final_score:.1f}")  
        print(f"Statistische Score (tempered): {tempered_score:.1f}")
        print(f"Gecombineerde Kansen: Win={combined_probs[0]:.3f}, Draw={combined_probs[1]:.3f}, Loss={combined_probs[2]:.3f}")
        print(f"FINALE VOORSPELLING: {final_prediction.upper()}")
        
        # Top contributing features
        print(f"\nTop Features (impact):")
        sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, contrib in sorted_contribs[:8]:
            print(f"  {feat}: {contrib:+.3f}")
            
    elif choice == "3":
        print("\nTraditionele Score met Enhanced Features")
        
        home_file = choose_file('Upload THUIS team CSV')
        away_file = choose_file('Upload UIT team CSV')

        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)

        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home > 1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away > 1 else 0, low_memory=False)

        home_df = apply_venue_filter(home_df, 'home')
        away_df = apply_venue_filter(away_df, 'away')

        home_feats = build_enhanced_feature_series(home_df, "HOME TEAM")
        away_feats = build_enhanced_feature_series(away_df, "AWAY TEAM")

        delta = make_enhanced_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs = compute_enhanced_weighted_score(delta)

        print(f'\n=== ENHANCED ANALYSE ===')
        print(f'Weighted difference = {weighted_diff:.3f}')
        print(f'Final score = {final_score:.1f}')
        
        if final_score > 52:
            print("THUIS TEAM voorkeur")
        elif final_score < 48:
            print("UIT TEAM voorkeur")
        else:
            print("Teams ongeveer gelijk")
    
    elif choice == "4":
        print("\nBatch Analyse (toekomstige uitbreiding)")
        print("Deze functie wordt in a volgende versie geïmplementeerd.")
    
    else:
        print("Ongeldige keuze. Kies 1, 2, 3 of 4.")
