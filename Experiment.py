#!/usr/bin/env python3
"""
enhanced_ml_soccer_predictor_cnn.py
Uitgebreide versie met CNN model integratie
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# -------------------------
# Enhanced Config with CNN
# -------------------------
EMA_SPAN = 7
SCALE_TO_SCORE = 4.0
MODEL_FILE = "enhanced_soccer_model.pkl"
XGBOOST_MODEL_FILE = "enhanced_xgboost_model.pkl"
CNN_MODEL_FILE = "enhanced_cnn_model.h5"
SCALER_FILE = "enhanced_scaler.pkl"
WEIGHTS_FILE = "enhanced_dynamic_weights.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# Uitgebreide minimum standard deviations
MIN_STD_VALUES = {
    'xG90': 0.3, 'Sh90': 1.0, 'SoT90': 0.5, 'ShotQual': 0.05, 'ConvRatio90': 0.1,
    'Goals': 0.5, 'Prog90': 5.0, 'PrgDist90': 100.0, 'Att3rd90': 10.0,
    'Possession': 0.05, 'FieldTilt': 0.05, 'HighPress': 1.0, 'AerialMismatch': 5.0,
    'KeeperPSxGdiff': 0.2, 'TkldPct_possession': 0.05, 'WonPct_misc': 0.05,
    'Att_3rd_defense': 1.0, 'SetPieces90': 0.8,
    # Nieuwe features
    'WinStreak': 0.5, 'UnbeatenStreak': 0.5, 'LossStreak': 0.5,
    'WinRate5': 0.1, 'WinRate10': 0.1, 'PointsRate5': 0.1, 'PointsRate10': 0.1,
    'RestDays': 1.0, 'RecentForm': 0.2,
    'HomeAdvantage': 0.1
}

# Uitgebreide initiele gewichten - aangepast voor gelijkspel-bias
WEIGHTS = {
    # Bestaande weights met aanpassingen
    'xG90': 1.1, 'Sh90': 1.6, 'SoT90': 0.8, 'ShotQual': 1.5, 'ConvRatio90': 1.8,
    'Goals': 0.8, 'Prog90': 0.35, 'PrgDist90': 0.25, 'Att3rd90': 0.6,
    'FieldTilt': 0.9, 'HighPress': 0.95, 'AerialMismatch': 0.6, 'Possession': 0.4,
    'KeeperPSxGdiff': -0.44, 'GoalsAgainst': -2.481, 'TkldPct_possession': 0.4,
    'WonPct_misc': 0.4, 'Att_3rd_defense': 0.8, 'SetPieces90': 0.8,
    # Nieuwe weights - aangepast om gelijkspel te verminderen
    'WinStreak': 1.4, 'UnbeatenStreak': 0.7, 'LossStreak': -1.2,
    'WinRate5': 1.7, 'WinRate10': 1.1, 'PointsRate5': 1.5, 'PointsRate10': 1.0,
    'RestDays': 0.4, 'RecentForm': 1.3,
    'HomeAdvantage': 0.6
}

ML_WEIGHTS = WEIGHTS.copy()

# XGBoost parameters - geoptimaliseerd tegen gelijkspel-bias
XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'random_state': 42,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 1.2
}

# CNN parameters
CNN_PARAMS = {
    'filters': [64, 128, 64],
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': [128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'patience': 15
}

# -------------------------
# Enhanced Helper Functions
# -------------------------
def choose_file(prompt):
    Tk().withdraw()
    path = askopenfilename(title=prompt)
    print(f"{prompt}: {path}")
    return path

def choose_files(prompt):
    """Selecteer meerdere bestanden tegelijk."""
    Tk().withdraw()
    paths = askopenfilename(title=prompt, multiple=True)
    if paths:
        print(f"{prompt}: {len(paths)} bestanden geselecteerd")
        for i, path in enumerate(paths, 1):
            print(f"  {i}. {path}")
    else:
        print("Geen bestanden geselecteerd")
    return paths

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
        score_match = re.match(r'(\d+)\s*[-â€":\u2013]\s*(\d+)', result_str)
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
            'RestDays': 7, 'RecentForm': 0
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
    
    return features

def calculate_seasonal_context_features(df):
    """Berekent seizoenscontext features."""
    features = {}
    n = len(df)
    
    if n == 0:
        return {
            'HomeAdvantage': pd.Series([0.0])
        }
    
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
    setpieces_col = find_column_flexible(df, [
        ['pass types_ck_passing_types'], 
        ['ck_passing_types'], 
        ['corner', 'kicks'], 
        ['corners'], 
        ['ck'],
        ['corner_kicks'],
        ['set_pieces'],
        ['setpieces']
    ])
    
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
        ['wonpct_misc'],  # DIRECTE MATCH met je CSV!
        ['aerial duels_won%_misc'], 
        ['won%_misc'], 
        ['aerial', 'won%'], 
        ['aerial', '%'], 
        ['duel', 'won%'],
        ['aerial', 'won', '%'],
        ['aerial', 'success', '%'],
        ['aerial', 'duels', '%'],
        ['air duels', '%'],
        ['heading', 'won', '%'],
        ['header', 'won', '%']
    ])

    # Debug voor aerial win
    if aerial_win_col:
        print(f"Aerial win kolom gevonden: {aerial_win_col}")
        print(f"Eerste paar waarden: {df[aerial_win_col].head().tolist()}")
    else:
        print("AERIAL WIN KOLOM NIET GEVONDEN!")
        print("Beschikbare kolommen met 'won' of 'aerial':")
        for col in df.columns:
            col_lower = str(col).lower()
            if 'won' in col_lower or 'aerial' in col_lower or 'duel' in col_lower:
                print(f"  Mogelijke kandidaat: {col}")
    
    # Defensive actions
    def3rd_col = find_column_flexible(df, [['tackles_def 3rd_defense'], ['def_3rd_defense'], ['defensive', '3rd']])
    int_col = find_column_flexible(df, [['performance_int_misc'], ['int_misc'], ['interceptions'], ['int']])
    
    # Tackled percentage
    tkldpct_col = find_column_flexible(df, [['tkldpct_possession'], ['tkld%_possession'], ['tackled', '%']])
    
    # Won percentage
    wonpct_col = find_column_flexible(df, [['won%_misc'], ['duels', 'won%'], ['won', '%']])
    
    # Attacking third defense
    att3rddef_col = find_column_flexible(df, [['tackles_att 3rd_defense'], ['att_3rd_defense'], ['attacking', '3rd', 'tackles']])
    
    # Debug voor set pieces
    if setpieces_col:
        print(f"Set pieces (corners) kolom gevonden: {setpieces_col}")
        print(f"Eerste paar waarden: {df[setpieces_col].head().tolist()}")
    else:
        print("SET PIECES KOLOM NIET GEVONDEN!")
        print("Beschikbare kolommen met 'ck' of 'corner':")
        for col in df.columns:
            col_lower = str(col).lower()
            if 'ck' in col_lower or 'corner' in col_lower:
                print(f"  Mogelijke kandidaat: {col}")
    
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
                mres = re.search(r'(\d+)\s*[-â€":\u2013]\s*(\d+)', res_str)
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
        'TkldPct_possession', 'WonPct_misc', 'Att_3rd_defense',
        # New historical features
        'WinStreak', 'UnbeatenStreak', 'LossStreak', 'WinRate5', 'WinRate10', 
        'PointsRate5', 'PointsRate10', 'RestDays', 'RecentForm',
        'HomeAdvantage'
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
    """Enhanced scoring met betere gelijkspel-bias correctie."""
    z_team = {}
    z_opp = {}
    contribs = {}

    print(f"\n--- Enhanced Z-Score Calculation (Anti-Draw Bias) ---")
    print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Diff':<8} | {'Std':<8} | {'Z-Diff':<8} | {'Weight':<8} | {'Contrib':<8}")
    print("-" * 100)

    # Use ML weights if available
    weights_to_use = ML_WEIGHTS if use_ml_weights else WEIGHTS

    for feat, (t_ema, o_ema) in delta_dict.items():
        min_std = MIN_STD_VALUES.get(feat, max(0.1, abs(t_ema) * 0.1))
        combined = np.array([t_ema, o_ema])
        mean, std = np.nanmean(combined), np.nanstd(combined)
        
        # Enhanced robust_std calculation met anti-draw bias
        robust_std = max(std, min_std, abs(mean) * 0.12, 0.25)  # Verlaagd voor meer spreiding
        
        zt = (t_ema - mean) / robust_std
        zo = (o_ema - mean) / robust_std
        
        # Meer agressieve clipping voor minder gelijkspellen
        if feat in ['WinStreak', 'UnbeatenStreak', 'LossStreak', 'RestDays']:
            zt = np.clip(zt, -3.5, 3.5)  # Meer extreme waarden toegestaan
            zo = np.clip(zo, -3.5, 3.5)
        else:
            zt = np.clip(zt, -3.0, 3.0)  # Verhoogd van -2.5, 2.5
            zo = np.clip(zo, -3.0, 3.0)
        
        z_team[feat] = zt
        z_opp[feat] = zo
        
        weight = weights_to_use.get(feat, 0.0)
        
        # Enhanced contribution met anti-draw bias
        raw_contrib = weight * (zt - zo)
        
        # Anti-draw scaling factors
        if feat in ['WinRate5', 'WinRate10', 'PointsRate5', 'PointsRate10', 'RecentForm']:
            scaling_factor = 1.4  # Verhoogd van 1.2
        elif feat in ['WinStreak', 'LossStreak']:
            scaling_factor = 1.3  # Nieuwe categorie voor extremere features
        elif feat in ['RestDays']:
            scaling_factor = 0.8
        else:
            scaling_factor = 1.1  # Verhoogd van 1.0
        
        # Minder agressieve soft clipping
        contribs[feat] = raw_contrib * scaling_factor * np.tanh(abs(raw_contrib) / 3.0) / max(abs(raw_contrib), 1e-6)  # Verlaagd van 4.0 naar 3.0
        
        diff = t_ema - o_ema
        z_diff = zt - zo
        
        print(f"{feat:<20} | {t_ema:<10.3f} | {o_ema:<10.3f} | {diff:<+8.3f} | {robust_std:<8.3f} | {z_diff:<+8.3f} | {weight:<8.3f} | {contribs[feat]:<+8.3f}")

    # Special handling for certain features
    if 'KeeperPSxGdiff' in contribs:
        contribs['KeeperPSxGdiff'] = -contribs['KeeperPSxGdiff']

    weighted_diff = sum(contribs.values())
    
    # Anti-draw bias scaling
    max_expected_diff = 12.0  # Verlaagd van 15.0 voor meer spreiding
    scaled_diff = max_expected_diff * np.tanh(weighted_diff / max_expected_diff)
    
    # Base score met meer extreme scaling
    final = 50.0 + SCALE_TO_SCORE * 1.2 * scaled_diff  # 20% meer extreme scores
    
    # Dynamic capping met bredere range
    confidence = min(1.0, len([c for c in contribs.values() if abs(c) > 0.1]) / 10.0)
    cap_range = 35 + confidence * 30  # Range van 35-65 naar 35-95
    
    final = np.clip(final, 25.0, 75.0 + cap_range)  # Bredere range: 25-100+
    
    return final, scaled_diff, z_team, z_opp, contribs

# -------------------------
# NEW: CNN Functions
# -------------------------
def create_cnn_model(input_shape, num_classes=3):
    """Creëer een 1D CNN model voor tabulaire data."""
    model = Sequential()
    
    # Eerste convolutie layer
    model.add(Conv1D(filters=CNN_PARAMS['filters'][0], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=CNN_PARAMS['pool_size']))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    # Tweede convolutie layer
    model.add(Conv1D(filters=CNN_PARAMS['filters'][1], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=CNN_PARAMS['pool_size']))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    # Derde convolutie layer
    model.add(Conv1D(filters=CNN_PARAMS['filters'][2], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=CNN_PARAMS['pool_size']))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    # Flatten en dense layers
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['dense_units'][0], activation='relu'))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    model.add(Dense(CNN_PARAMS['dense_units'][1], activation='relu'))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=CNN_PARAMS['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_cnn_model(X, y):
    """Train een CNN model met anti-draw bias."""
    if len(X) < 10:
        print(f"Onvoldoende data voor CNN training: {len(X)} samples. Minimum 10 vereist.")
        return None, None, None
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split the data
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, 
                                                        random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for CNN (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Create and train CNN model
    model = create_cnn_model((X_train_reshaped.shape[1], 1), num_classes=3)
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=CNN_PARAMS['patience'], restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=CNN_PARAMS['epochs'],
        batch_size=CNN_PARAMS['batch_size'],
        validation_data=(X_test_reshaped, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"\nCNN Model Performance:")
    print(f"Nauwkeurigheid: {test_acc:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Predictions
    y_pred_proba = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert back to original labels for report
    y_test_original = le.inverse_transform(np.argmax(y_test, axis=1))
    y_pred_original = le.inverse_transform(y_pred)
    
    print("\nCNN Classificatie Rapport:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_original, y_pred_original, labels=['W', 'D', 'L'])
    print(f"\nConfusion Matrix:")
    print(f"{'':>8} {'W':>8} {'D':>8} {'L':>8}")
    for i, true_label in enumerate(['W', 'D', 'L']):
        print(f"{true_label:>8} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y_encoded):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y_encoded[train_idx], y_encoded[val_idx]
        
        # Scale
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        # Reshape
        X_train_cv_reshaped = X_train_cv_scaled.reshape(X_train_cv_scaled.shape[0], X_train_cv_scaled.shape[1], 1)
        X_val_cv_reshaped = X_val_cv_scaled.reshape(X_val_cv_scaled.shape[0], X_val_cv_scaled.shape[1], 1)
        
        # Convert to categorical
        y_train_cv_cat = to_categorical(y_train_cv)
        y_val_cv_cat = to_categorical(y_val_cv)
        
        # Create and train model
        model_cv = create_cnn_model((X_train_cv_reshaped.shape[1], 1), num_classes=3)
        model_cv.fit(X_train_cv_reshaped, y_train_cv_cat, epochs=CNN_PARAMS['epochs'], 
                    batch_size=CNN_PARAMS['batch_size'], verbose=0)
        
        # Evaluate
        _, acc = model_cv.evaluate(X_val_cv_reshaped, y_val_cv_cat, verbose=0)
        cv_scores.append(acc)
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"CV Mean: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
    
    return model, scaler, le

# -------------------------
# Ensemble Functions
# -------------------------
def create_draw_penalty_weights(y):
    """Creëer aangepaste class weights om gelijkspel-bias te verminderen."""
    classes = np.unique(y)
    
    # Manual weights - penaliseer draws harder
    class_weights = {}
    for cls in classes:
        if cls == 'D':
            class_weights[cls] = 0.6  # Verminderd gewicht voor draws
        elif cls == 'W':
            class_weights[cls] = 1.2  # Verhoogd gewicht voor wins
        else:  # 'L'
            class_weights[cls] = 1.2  # Verhoogd gewicht voor losses
    
    return class_weights

def train_xgboost_model(X, y):
    """Train een XGBoost model met anti-draw bias."""
    if len(X) < 10:
        print(f"Onvoldoende data voor XGBoost training: {len(X)} samples. Minimum 10 vereist.")
        return None, None, None
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the data
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom class weights om draw bias te verminderen
    original_labels = le.inverse_transform(y_train)
    class_weights = create_draw_penalty_weights(original_labels)
    sample_weights = np.array([class_weights[le.inverse_transform([label])[0]] for label in y_train])
    
    # XGBoost model met aangepaste parameters
    xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    
    # Train met sample weights - aangepast voor nieuwere XGBoost versies
    try:
        # Probeer eerst met eval_set en callbacks (nieuwere versies)
        xgb_model.fit(
            X_train_scaled, 
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test_scaled, y_test)],
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
            verbose=False
        )
    except TypeError:
        # Fallback voor oudere versies of andere configuraties
        xgb_model.fit(
            X_train_scaled, 
            y_train,
            sample_weight=sample_weights,
            verbose=False
        )
    
    # Predictions
    y_pred = xgb_model.predict(X_test_scaled)
    y_prob = xgb_model.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nXGBoost Model Performance:")
    print(f"Nauwkeurigheid: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Convert back to original labels for report
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    print("\nXGBoost Classificatie Rapport:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_original, y_pred_original, labels=['W', 'D', 'L'])
    print(f"\nConfusion Matrix:")
    print(f"{'':>8} {'W':>8} {'D':>8} {'L':>8}")
    for i, true_label in enumerate(['W', 'D', 'L']):
        print(f"{true_label:>8} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    
    # Feature importance
    feature_names = X.columns.tolist()
    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 XGBoost Features:")
    print(feature_importance_df.head(15))
    
    # Cross-validation score
    cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=3, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return xgb_model, scaler, le

def ensemble_prediction(rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder, features):
    """Combineer Random Forest, XGBoost en CNN voorspellingen."""
    features_df = pd.DataFrame([features])
    
    # Random Forest prediction
    rf_features_scaled = rf_scaler.transform(features_df)
    rf_prediction = rf_model.predict(rf_features_scaled)[0]
    rf_probabilities = rf_model.predict_proba(rf_features_scaled)[0]
    
    # XGBoost prediction  
    xgb_features_scaled = xgb_scaler.transform(features_df)
    xgb_prediction_encoded = xgb_model.predict(xgb_features_scaled)[0]
    xgb_prediction = label_encoder.inverse_transform([xgb_prediction_encoded])[0]
    xgb_probabilities = xgb_model.predict_proba(xgb_features_scaled)[0]
    
    # CNN prediction
    cnn_features_scaled = cnn_scaler.transform(features_df)
    cnn_features_reshaped = cnn_features_scaled.reshape(cnn_features_scaled.shape[0], cnn_features_scaled.shape[1], 1)
    cnn_probabilities = cnn_model.predict(cnn_features_reshaped)[0]
    cnn_prediction_encoded = np.argmax(cnn_probabilities)
    cnn_prediction = label_encoder.inverse_transform([cnn_prediction_encoded])[0]
    
    # Convert XGB and CNN probabilities to same order as RF (W, D, L)
    xgb_classes = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    xgb_prob_dict = dict(zip(xgb_classes, xgb_probabilities))
    xgb_probs_ordered = [xgb_prob_dict.get(cls, 0) for cls in ['W', 'D', 'L']]
    
    cnn_classes = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    cnn_prob_dict = dict(zip(cnn_classes, cnn_probabilities))
    cnn_probs_ordered = [cnn_prob_dict.get(cls, 0) for cls in ['W', 'D', 'L']]
    
    # Ensemble weighting - meer gewicht naar CNN en XGBoost vanwege anti-draw bias
    rf_weight = 0.3
    xgb_weight = 0.35
    cnn_weight = 0.35
    
    ensemble_probs = [
        rf_weight * rf_probabilities[0] + xgb_weight * xgb_probs_ordered[0] + cnn_weight * cnn_probs_ordered[0],  # Win
        rf_weight * rf_probabilities[1] + xgb_weight * xgb_probs_ordered[1] + cnn_weight * cnn_probs_ordered[1],  # Draw
        rf_weight * rf_probabilities[2] + xgb_weight * xgb_probs_ordered[2] + cnn_weight * cnn_probs_ordered[2]   # Loss
    ]
    
    ensemble_prediction = ['W', 'D', 'L'][np.argmax(ensemble_probs)]
    
    return {
        'rf_prediction': rf_prediction,
        'rf_probabilities': rf_probabilities,
        'xgb_prediction': xgb_prediction,
        'xgb_probabilities': xgb_probs_ordered,
        'cnn_prediction': cnn_prediction,
        'cnn_probabilities': cnn_probs_ordered,
        'ensemble_prediction': ensemble_prediction,
        'ensemble_probabilities': ensemble_probs
    }

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
    
    # Add difference and ratio features - crucial voor model prestatie
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
    
    # Add interaction features voor sleutel metrics
    key_interactions = [
        ('WinRate5', 'RecentForm'),
        ('RestDays', 'WinStreak'),
        ('xG90', 'ShotQual'),
        ('Possession', 'FieldTilt')
    ]
    
    for feat1, feat2 in key_interactions:
        home_interaction = home_ema.get(f"home_{feat1}", 0) * home_ema.get(f"home_{feat2}", 0)
        away_interaction = away_ema.get(f"away_{feat1}", 0) * away_ema.get(f"away_{feat2}", 0)
        combined_features[f"interaction_home_{feat1}_{feat2}"] = home_interaction
        combined_features[f"interaction_away_{feat1}_{feat2}"] = away_interaction
        combined_features[f"interaction_diff_{feat1}_{feat2}"] = home_interaction - away_interaction
    
    return combined_features

def train_triple_models(X, y):
    """Train Random Forest, XGBoost en CNN modellen."""
    if len(X) < 10:
        print(f"Onvoldoende data voor training: {len(X)} samples. Minimum 10 vereist.")
        return None, None, None, None, None, None, None
    
    print("Training Random Forest model...")
    rf_model, rf_scaler = train_enhanced_rf_model(X, y)
    
    print("\nTraining XGBoost model...")
    xgb_model, xgb_scaler, label_encoder = train_xgboost_model(X, y)
    
    print("\nTraining CNN model...")
    cnn_model, cnn_scaler, _ = train_cnn_model(X, y)
    
    if rf_model is not None and xgb_model is not None and cnn_model is not None:
        # Save alle modellen
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(XGBOOST_MODEL_FILE, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        cnn_model.save(CNN_MODEL_FILE)
        
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump({'rf_scaler': rf_scaler, 'xgb_scaler': xgb_scaler, 'cnn_scaler': cnn_scaler}, f)
            
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"\nAlle modellen opgeslagen!")
        print(f"Random Forest: {MODEL_FILE}")
        print(f"XGBoost: {XGBOOST_MODEL_FILE}")
        print(f"CNN: {CNN_MODEL_FILE}")
        print(f"Scalers: {SCALER_FILE}")
        print(f"Label Encoder: {LABEL_ENCODER_FILE}")
    
    return rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder

def train_enhanced_rf_model(X, y):
    """Train Random Forest met aangepaste parameters."""
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom class weights voor draw penalty
    class_weights = create_draw_penalty_weights(y_train)
    
    # Enhanced Random Forest
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight=class_weights,
        max_features='sqrt'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Nauwkeurigheid: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("\nRandom Forest Classificatie Rapport:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def load_all_models():
    """Laad alle modellen voor ensemble voorspelling."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            rf_model = pickle.load(f)
        
        with open(XGBOOST_MODEL_FILE, 'rb') as f:
            xgb_model = pickle.load(f)
            
        cnn_model = tf.keras.models.load_model(CNN_MODEL_FILE)
            
        with open(SCALER_FILE, 'rb') as f:
            scalers = pickle.load(f)
            rf_scaler = scalers['rf_scaler']
            xgb_scaler = scalers['xgb_scaler']
            cnn_scaler = scalers['cnn_scaler']
        
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("Alle modellen succesvol geladen!")
        return rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder
        
    except FileNotFoundError as e:
        print(f"Model niet gevonden: {e}")
        return None, None, None, None, None, None, None

def ultimate_combined_prediction(ensemble_results, statistical_score):
    """
    Ultimate prediction die ensemble ML + statistical score combineert
    met betere anti-draw bias.
    """
    def anti_draw_temper_score(score):
        """Score tempering met anti-draw bias."""
        # Meer extreme scores om draws te vermijden
        if score < 40:
            return 35 + (score - 40) * 0.8
        elif score > 60:
            return 65 + (score - 60) * 0.8
        elif 45 <= score <= 55:
            # "Dead zone" - push away from draw territory
            center = 50
            distance = abs(score - center)
            if distance < 3:  # Very close to 50
                direction = 1 if score >= center else -1
                return center + direction * 6  # Push to 56 or 44
            else:
                return score + (3 if score > center else -3)
        else:
            return score
    
    tempered_score = anti_draw_temper_score(statistical_score)
    
    # Convert naar kansen met anti-draw bias
    if tempered_score > 52:
        stat_win_prob = min(0.80, max(0.20, (tempered_score - 40) / 25))
        stat_loss_prob = max(0.05, min(0.35, (60 - tempered_score) / 30))
    else:
        stat_win_prob = max(0.05, min(0.35, (tempered_score - 30) / 30))
        stat_loss_prob = min(0.80, max(0.20, (60 - tempered_score) / 25))
    
    # Zeer lage draw probability
    stat_draw_prob = max(0.08, 1.0 - stat_win_prob - stat_loss_prob)  # Maximum 8% draw
    
    # Renormalize
    total = stat_win_prob + stat_draw_prob + stat_loss_prob
    if total > 0:
        stat_win_prob /= total
        stat_draw_prob /= total
        stat_loss_prob /= total
    
    # Dynamic weighting gebaseerd op ensemble confidence
    ensemble_probs = ensemble_results['ensemble_probabilities']
    ml_confidence = max(ensemble_probs) - np.mean(ensemble_probs)
    score_extremity = abs(statistical_score - 50) / 25.0
    
    # Meer ML gewicht als het confident is EN minder draw voorspelt
    ml_draw_penalty = 1.0 - ensemble_probs[1]  # Minder draw = meer vertrouwen
    
    if ml_confidence > 0.25 and ml_draw_penalty > 0.7:
        ml_weight = 0.75  # Zeer confident ML
        stat_weight = 0.25
    elif ml_confidence > 0.15 and score_extremity > 0.6:
        ml_weight = 0.7   # Confident ML + extreme score
        stat_weight = 0.3
    elif ensemble_probs[1] < 0.25:  # ML voorspelt weinig draw
        ml_weight = 0.65
        stat_weight = 0.35
    else:
        ml_weight = 0.6   # Default
        stat_weight = 0.4
    
    # Final combination
    final_probs = [
        ml_weight * ensemble_probs[0] + stat_weight * stat_win_prob,  # Win
        ml_weight * ensemble_probs[1] + stat_weight * stat_draw_prob,  # Draw
        ml_weight * ensemble_probs[2] + stat_weight * stat_loss_prob   # Loss
    ]
    
    # Extra draw penalty
    draw_penalty = 0.85  # Reduce draw probability by 15%
    final_probs[1] *= draw_penalty
    
    # Redistribute draw probability to win/loss
    redistributed = final_probs[1] * (1 - draw_penalty)
    final_probs[0] += redistributed * 0.5
    final_probs[2] += redistributed * 0.5
    
    # Renormalize
    total = sum(final_probs)
    if total > 0:
        final_probs = [p / total for p in final_probs]
    
    final_prediction = ['W', 'D', 'L'][np.argmax(final_probs)]
    
    return final_prediction, final_probs, tempered_score

# -------------------------
# Enhanced Main Function
# -------------------------
if __name__ == '__main__':
    print("Enhanced CNN Voetbalwedstrijdvoorspeller v4.0")
    print("Met Anti-Draw Bias en Ensemble Learning (RF + XGBoost + CNN)")
    print("=" * 60)
    
    print("\nKies een optie:")
    print("1. Train nieuwe modellen (Random Forest + XGBoost + CNN) - meerdere CSV's mogelijk")
    print("2. Ensemble voorspelling (RF + XGBoost + CNN + Statistical)")
    print("3. Traditionele score met enhanced features")
    print("4. Model vergelijking en analyse")
    print("5. Batch analyse van meerdere wedstrijden")
    
    choice = input("Jouw keuze (1-5): ").strip()
    
    if choice == "1":
        print("\nTraining Ensemble Modellen")
        print("Upload CSV-bestanden met uitgebreide historische data")
        
        # Gebruik de nieuwe choose_files functie voor meerdere bestanden
        training_files = choose_files('Selecteer CSV bestanden voor training (meerdere mogelijk)')
        
        if not training_files:
            print("Geen bestanden geselecteerd.")
            exit()
        
        # Convert tuple to list if needed (tkinter returns tuple)
        if isinstance(training_files, tuple):
            training_files = list(training_files)
        
        # Collect training data
        X_data = []
        y_data = []
        
        print(f"\nVerwerken van {len(training_files)} trainingsbestanden...")
        for file_idx, file_path in enumerate(training_files, 1):
            print(f"\nBestand {file_idx}/{len(training_files)}: {file_path}")
            
            try:
                h_rows = detect_header_rows(file_path)
                df = pd.read_csv(file_path, header=list(range(h_rows)) if h_rows > 1 else 0, low_memory=False)
                
                print(f"  Geladen: {len(df)} wedstrijden")
                
                # Process each match met voldoende historie
                for i in range(max(7, EMA_SPAN), len(df)):  # Meer historie nodig
                    try:
                        # Create subsets
                        home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                        away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                        
                        if len(home_subset) >= 4 and len(away_subset) >= 4:  # Meer minimum matches
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
                        
            except Exception as e:
                print(f"  Fout bij laden bestand: {e}")
                continue
        
        print(f"\nTotaal verzamelde samples: {len(X_data)}")
        
        if len(X_data) < 15:
            print("Onvoldoende trainingsdata verzameld.")
            exit()
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        print(f"Feature dimensies: {X_df.shape}")
        print(f"Label distributie:")
        label_counts = y_series.value_counts()
        print(label_counts)
        print(f"Draw percentage: {label_counts.get('D', 0) / len(y_series) * 100:.1f}%")
        
        # Train ensemble models
        rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder = train_triple_models(X_df, y_series)
        
        if rf_model is not None and xgb_model is not None and cnn_model is not None:
            print("\nEnsemble model training voltooid!")
            print("Random Forest, XGBoost en CNN modellen zijn getraind met anti-draw bias.")
        
    elif choice == "2":
        print("\nEnsemble Voorspelling (RF + XGBoost + CNN + Statistical)")
        
        # Load alle modellen
        rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder = load_all_models()
        if rf_model is None or xgb_model is None or cnn_model is None:
            print("Modellen niet gevonden. Train eerst de modellen (optie 1).")
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

        # ML ensemble prediction
        features = prepare_enhanced_ml_features(home_feats, away_feats)
        ensemble_results = ensemble_prediction(
            rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder, features
        )
        
        # Enhanced statistical score
        delta = make_enhanced_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs = compute_enhanced_weighted_score(delta, use_ml_weights=True)
        
        # Ultimate combined prediction
        final_prediction, final_probs, tempered_score = ultimate_combined_prediction(
            ensemble_results, final_score
        )

        # Results
        print(f"\n=== ENSEMBLE VOORSPELLING (Anti-Draw Bias) ===")
        print(f"\nIndividuele Model Resultaten:")
        print(f"Random Forest: {ensemble_results['rf_prediction']} - "
              f"Win={ensemble_results['rf_probabilities'][0]:.3f}, "
              f"Draw={ensemble_results['rf_probabilities'][1]:.3f}, "
              f"Loss={ensemble_results['rf_probabilities'][2]:.3f}")
        
        print(f"XGBoost:       {ensemble_results['xgb_prediction']} - "
              f"Win={ensemble_results['xgb_probabilities'][0]:.3f}, "
              f"Draw={ensemble_results['xgb_probabilities'][1]:.3f}, "
              f"Loss={ensemble_results['xgb_probabilities'][2]:.3f}")
        
        print(f"CNN:           {ensemble_results['cnn_prediction']} - "
              f"Win={ensemble_results['cnn_probabilities'][0]:.3f}, "
              f"Draw={ensemble_results['cnn_probabilities'][1]:.3f}, "
              f"Loss={ensemble_results['cnn_probabilities'][2]:.3f}")
        
        print(f"ML Ensemble:   {ensemble_results['ensemble_prediction']} - "
              f"Win={ensemble_results['ensemble_probabilities'][0]:.3f}, "
              f"Draw={ensemble_results['ensemble_probabilities'][1]:.3f}, "
              f"Loss={ensemble_results['ensemble_probabilities'][2]:.3f}")
        
        print(f"\nStatistische Scores:")
        print(f"Raw Score: {final_score:.1f}")
        print(f"Anti-Draw Tempered: {tempered_score:.1f}")
        
        print(f"\n🏆 FINALE VOORSPELLING: {final_prediction}")
        print(f"Finale Kansen: Win={final_probs[0]:.3f}, Draw={final_probs[1]:.3f}, Loss={final_probs[2]:.3f}")
        
        # Confidence indicator
        max_prob = max(final_probs)
        if max_prob > 0.6:
            confidence = "HOOG"
        elif max_prob > 0.45:
            confidence = "GEMIDDELD"
        else:
            confidence = "LAAG"
        print(f"Voorspelling Vertrouwen: {confidence}")
        
        # Top contributing features
        print(f"\nTop Features (statistisch):")
        sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, contrib in sorted_contribs[:8]:
            print(f"  {feat}: {contrib:+.3f}")
            
    elif choice == "3":
        print("\nTraditionele Score met Enhanced Features (Anti-Draw Bias)")
        
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

        print(f'\n=== ENHANCED ANALYSE (Anti-Draw Bias) ===')
        print(f'Weighted difference = {weighted_diff:.3f}')
        print(f'Final score = {final_score:.1f}')
        
        if final_score > 55:
            print("STERKE THUIS TEAM voorkeur")
        elif final_score > 52:
            print("THUIS TEAM lichte voorkeur")
        elif final_score < 45:
            print("STERKE UIT TEAM voorkeur")
        elif final_score < 48:
            print("UIT TEAM lichte voorkeur")
        else:
            print("Teams relatief gelijk (mogelijk spannende wedstrijd)")
    
    elif choice == "4":
        print("\nModel Vergelijking en Analyse")
        
        # Load models
        rf_model, xgb_model, cnn_model, rf_scaler, xgb_scaler, cnn_scaler, label_encoder = load_all_models()
        if rf_model is None or xgb_model is None or cnn_model is None:
            print("Modellen niet gevonden. Train eerst de modellen (optie 1).")
            exit()
        
        print("Model vergelijking functie wordt geïmplementeerd in volgende versie.")
        print("Deze functie zal model prestaties vergelijken op test data.")
    
    elif choice == "5":
        print("\nBatch Analyse")
        print("Deze functie wordt geïmplementeerd voor analyse van meerdere wedstrijden tegelijk.")
    
    else:
        print("Ongeldige keuze. Kies 1, 2, 3, 4 of 5.")

print("\nProgramma beëindigd.")
