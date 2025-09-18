#!/usr/bin/env python3
"""
enhanced_ml_soccer_predictor_tactical_correlations.py
Versie 8.0 - Met Tactical Mismatches, Style Clustering en Correlation Analysis
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import mode, poisson
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# -------------------------
# Enhanced Config with Tactical Analysis
# -------------------------
EMA_SPAN = 7
SCALE_TO_SCORE = 4.0
MODEL_FILE = "enhanced_soccer_model.pkl"
XGBOOST_MODEL_FILE = "enhanced_xgboost_model.pkl"
LIGHTGBM_MODEL_FILE = "enhanced_lightgbm_model.pkl"
CNN_MODEL_FILE = "enhanced_cnn_model.h5"
TRANSFORMER_MODEL_FILE = "enhanced_transformer_model.h5"
STACKING_MODEL_FILE = "enhanced_stacking_model.pkl"
SCALER_FILE = "enhanced_scaler.pkl"
WEIGHTS_FILE = "enhanced_dynamic_weights.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# Poisson and Calibration model files
POISSON_HOME_MODEL_FILE = "poisson_home_model.pkl"
POISSON_AWAY_MODEL_FILE = "poisson_away_model.pkl"
CALIBRATED_RF_FILE = "calibrated_rf_model.pkl"
CALIBRATED_XGB_FILE = "calibrated_xgb_model.pkl"
CALIBRATED_LGB_FILE = "calibrated_lgb_model.pkl"

# New files for tactical analysis
STYLE_CLUSTERING_FILE = "style_clustering_model.pkl"
TACTICAL_CORRELATION_FILE = "tactical_correlation_weights.pkl"

# Poisson simulation parameters
POISSON_SIMULATIONS = 10000
MAX_GOALS = 7

# Playing style definitions
PLAYING_STYLES = {
    0: 'Possession-Based',
    1: 'Counter-Attack',
    2: 'Physical/Direct',
    3: 'High-Press',
    4: 'Balanced'
}

# Enhanced minimum standard deviations
MIN_STD_VALUES = {
    'xG90': 0.3, 'Sh90': 1.0, 'SoT90': 0.5, 'ShotQual': 0.05, 'ConvRatio90': 0.1,
    'Goals': 0.5, 'Prog90': 5.0, 'PrgDist90': 100.0, 'Att3rd90': 10.0,
    'Possession': 0.05, 'FieldTilt': 0.05, 'HighPress': 1.0, 
    'AerialMismatch': 3.0,
    'KeeperPSxGdiff': 0.2, 'TkldPct_possession': 0.05, 'WonPct_misc': 0.05,
    'Att_3rd_defense': 1.0, 'SetPieces90': 0.8,
    'WinStreak': 0.5, 'UnbeatenStreak': 0.5, 'LossStreak': 0.5,
    'WinRate5': 0.1, 'WinRate10': 0.1, 'PointsRate5': 0.1, 'PointsRate10': 0.1,
    'RestDays': 1.0, 'RecentForm': 0.2,
    'HomeAdvantage': 0.1
}

# Base weights - will be adjusted contextually
BASE_WEIGHTS = {
    'xG90': 1.0, 'Sh90': 1.4, 'SoT90': 0.8, 'ShotQual': 1.3, 'ConvRatio90': 1.5,
    'Goals': 0.8, 'Prog90': 0.35, 'PrgDist90': 0.25, 'Att3rd90': 0.6,
    'FieldTilt': 0.8, 'HighPress': 0.85, 
    'AerialMismatch': 1.8,
    'Possession': 0.4,
    'KeeperPSxGdiff': -0.44, 'GoalsAgainst': -2.2, 'TkldPct_possession': 0.4,
    'WonPct_misc': 0.4, 'Att_3rd_defense': 0.8, 'SetPieces90': 0.8,
    'WinStreak': 1.1, 'UnbeatenStreak': 0.6, 'LossStreak': -1.0,
    'WinRate5': 1.3, 'WinRate10': 1.0, 'PointsRate5': 1.2, 'PointsRate10': 0.9,
    'RestDays': 0.4, 'RecentForm': 1.1,
    'HomeAdvantage': 0.6
}

# Tactical mismatch weights
TACTICAL_WEIGHTS = {
    'press_vs_possession': 1.2,
    'high_line_vs_pace': 1.4,
    'aerial_vs_setpieces': 1.3,
    'defensive_vs_attacking': 1.1,
    'attack_vs_defense': 1.5,
    'possession_under_pressure': 1.0
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'learning_rate': 0.08,
    'n_estimators': 400,
    'random_state': 42,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'scale_pos_weight': 1.0,
    'gamma': 0.1,
    'max_delta_step': 1
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 25,
    'learning_rate': 0.08,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 25,
    'random_state': 42,
    'verbosity': -1,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'min_gain_to_split': 0.1,
    'max_delta_step': 1.0
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

# Transformer parameters
TRANSFORMER_PARAMS = {
    'num_heads': 4,
    'key_dim': 32,
    'dense_units': [128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'patience': 15
}

# -------------------------
# Helper Functions
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
        score_match = re.match(r'(\d+)\s*[-–":\u2013]\s*(\d+)', result_str)
        if score_match:
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))
            
            if goals_for is not None:
                if abs(goals_for - home_goals) < 1e-6:
                    our_goals, opp_goals = home_goals, away_goals
                else:
                    our_goals, opp_goals = away_goals, home_goals
            else:
                our_goals, opp_goals = home_goals, away_goals
            
            if our_goals > opp_goals:
                return 'W'
            elif our_goals == opp_goals:
                return 'D'
            else:
                return 'L'
    
    return None

def calculate_brier_score(y_true, y_prob):
    """Calculate multi-class Brier score."""
    brier_scores = []
    for i in range(y_prob.shape[1]):
        y_true_binary = (y_true == i).astype(int)
        brier_scores.append(brier_score_loss(y_true_binary, y_prob[:, i]))
    return np.mean(brier_scores)

def ema(series, span=EMA_SPAN):
    """Enhanced EMA with better handling of edge cases."""
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return series.iloc[0] if hasattr(series, 'iloc') else series[0]
    return pd.Series(series).ewm(span=span, adjust=False).mean().iloc[-1]

# -------------------------
# TACTICAL ANALYSIS FUNCTIONS
# -------------------------
def calculate_tactical_mismatches(home_feats, away_feats):
    """Calculate tactical style mismatches between teams."""
    mismatches = {}
    
    # High press vs possession-based teams
    home_press = ema(home_feats.get('HighPress', pd.Series([0])))
    away_possession = ema(away_feats.get('Possession', pd.Series([0.5])))
    mismatches['press_vs_possession'] = home_press * (1 - away_possession) * 2
    
    # Counter-attack vulnerability
    home_def_high_line = ema(home_feats.get('Att_3rd_defense', pd.Series([0])))
    away_pace = ema(away_feats.get('PrgDist90', pd.Series([0])))
    mismatches['high_line_vs_pace'] = (home_def_high_line * away_pace) / 1000
    
    # Set piece mismatch
    home_aerial = ema(home_feats.get('AerialWin%', pd.Series([0.5])))
    away_setpieces = ema(away_feats.get('SetPieces90', pd.Series([0])))
    mismatches['aerial_vs_setpieces'] = (0.5 - home_aerial) * away_setpieces
    
    # Defensive solidity vs attacking threat
    home_tackles = ema(home_feats.get('TkldPct_possession', pd.Series([0])))
    away_att3rd = ema(away_feats.get('Att3rd90', pd.Series([0])))
    mismatches['defensive_vs_attacking'] = (home_tackles * 100) - away_att3rd
    
    # Wide play vs narrow defense
    home_crosses = ema(home_feats.get('SetPieces90', pd.Series([0])))  # Proxy for wide play
    away_central_def = ema(away_feats.get('WonPct_misc', pd.Series([0.5])))
    mismatches['wide_vs_narrow'] = home_crosses * (1 - away_central_def)
    
    # Pressing resistance
    home_possession = ema(home_feats.get('Possession', pd.Series([0.5])))
    away_press = ema(away_feats.get('HighPress', pd.Series([0])))
    mismatches['press_resistance'] = home_possession - (away_press / 100)
    
    return mismatches

def identify_playing_styles(features_df):
    """Cluster teams into playing styles."""
    style_features = [
        col for col in features_df.columns 
        if any(key in col for key in ['Possession', 'HighPress', 'PrgDist', 'SetPieces', 'Aerial', 'FieldTilt'])
    ]
    
    if len(style_features) < 3:
        print("Insufficient style features for clustering")
        return None, None, PLAYING_STYLES
    
    X_style = features_df[style_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_style_scaled = scaler.fit_transform(X_style)
    
    # Cluster into 5 playing styles
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    styles = kmeans.fit_predict(X_style_scaled)
    
    return styles, kmeans, PLAYING_STYLES

def get_style_matchup_history(home_style, away_style, historical_data):
    """Get historical results for specific style matchups."""
    if 'home_style' not in historical_data.columns or 'away_style' not in historical_data.columns:
        return None
        
    matchups = historical_data[
        (historical_data['home_style'] == home_style) & 
        (historical_data['away_style'] == away_style)
    ]
    
    if len(matchups) > 10:
        win_rate = (matchups['result'] == 'W').mean()
        draw_rate = (matchups['result'] == 'D').mean()
        loss_rate = (matchups['result'] == 'L').mean()
        
        avg_goals_for = matchups['goals_for'].mean() if 'goals_for' in matchups else None
        avg_goals_against = matchups['goals_against'].mean() if 'goals_against' in matchups else None
        
        return {
            'W': win_rate, 
            'D': draw_rate, 
            'L': loss_rate,
            'avg_goals_for': avg_goals_for,
            'avg_goals_against': avg_goals_against,
            'sample_size': len(matchups)
        }
    return None

def create_interaction_features(home_feats, away_feats):
    """Create non-linear interaction features."""
    interactions = {}
    
    # Squared differences for key metrics
    key_metrics = ['xG90', 'Goals', 'Possession', 'HighPress', 'AerialWin%']
    for metric in key_metrics:
        h = ema(home_feats.get(metric, pd.Series([0])))
        a = ema(away_feats.get(metric, pd.Series([0])))
        
        # Squared difference
        interactions[f'{metric}_squared_diff'] = (h - a) ** 2
        
        # Log ratio (with safety for zero values)
        if h > 0 and a > 0:
            interactions[f'{metric}_ratio_log'] = np.log(h / a)
        else:
            interactions[f'{metric}_ratio_log'] = 0
    
    # Cross-team interactions
    home_attack = ema(home_feats.get('xG90', pd.Series([0])))
    away_defense = ema(away_feats.get('GoalsAgainst', pd.Series([0])))
    interactions['attack_vs_defense'] = home_attack * (1 + away_defense)
    
    home_midfield = ema(home_feats.get('Possession', pd.Series([0.5])))
    away_press = ema(away_feats.get('HighPress', pd.Series([0])))
    interactions['possession_under_pressure'] = home_midfield * (1 + away_press / 10)
    
    # Fatigue interaction
    home_rest = ema(home_feats.get('RestDays', pd.Series([7])))
    away_rest = ema(away_feats.get('RestDays', pd.Series([7])))
    interactions['fatigue_differential'] = (home_rest - away_rest) / 7
    
    # Form vs pressure
    home_form = ema(home_feats.get('RecentForm', pd.Series([0])))
    away_form = ema(away_feats.get('RecentForm', pd.Series([0])))
    interactions['form_momentum'] = (home_form - away_form) ** 2
    
    return interactions

def calculate_contextual_weights(home_feats, away_feats, base_weights):
    """Adjust feature weights based on opponent characteristics."""
    context_weights = base_weights.copy()
    
    # Als tegenstander weinig balbezit heeft, wordt counter-kwaliteit belangrijker
    away_poss = ema(away_feats.get('Possession', pd.Series([0.5])))
    if away_poss < 0.45:
        context_weights['PrgDist90'] *= 1.3
        context_weights['ConvRatio90'] *= 1.2
        print(f"Low possession opponent detected ({away_poss:.1%}) - boosting counter metrics")
    
    # Als tegenstander veel aerial duels wint
    away_aerial = ema(away_feats.get('AerialWin%', pd.Series([0.5])))
    if away_aerial > 0.55:
        context_weights['AerialMismatch'] *= 1.4
        context_weights['SetPieces90'] *= 0.8
        print(f"Strong aerial opponent ({away_aerial:.1%}) - adjusting aerial weights")
    
    # Bij hoge press tegenstanders
    away_press = ema(away_feats.get('HighPress', pd.Series([0])))
    if away_press > 1.5:  # Assuming this is a high value
        context_weights['Possession'] *= 1.2
        context_weights['TkldPct_possession'] *= 1.3
        print(f"High pressing opponent - boosting possession retention metrics")
    
    # Home advantage adjustment based on recent home form
    home_advantage = ema(home_feats.get('HomeAdvantage', pd.Series([0])))
    if home_advantage > 0.2:
        context_weights['HomeAdvantage'] *= 1.2
        print(f"Strong home advantage detected - boosting home weight")
    
    return context_weights

def calculate_psychological_factors(home_df, away_df):
    """Calculate psychological and momentum factors."""
    factors = {}
    
    # Pressure situations - check recent big games
    def calculate_big_game_performance(df):
        # Approximation: games with high xG are "big games"
        if 'xG90' not in df.columns:
            return 0
        high_xg_games = df[df['xG90'] > df['xG90'].quantile(0.75)]
        if len(high_xg_games) > 0:
            wins = sum(1 for _, row in high_xg_games.iterrows() 
                      if extract_match_result_enhanced(row) == 'W')
            return wins / len(high_xg_games)
        return 0.5
    
    home_big_games = calculate_big_game_performance(home_df.tail(10))
    away_big_games = calculate_big_game_performance(away_df.tail(10))
    factors['big_game_differential'] = home_big_games - away_big_games
    
    # Comeback ability - games won from losing positions
    def count_comebacks(df):
        comebacks = 0
        for _, row in df.iterrows():
            # This is simplified - you'd need halftime scores ideally
            result = extract_match_result_enhanced(row)
            goals_against = row.get('GoalsAgainst', 0)
            if result == 'W' and goals_against > 0:
                comebacks += 1
        return comebacks
    
    home_comebacks = count_comebacks(home_df.tail(20))
    away_comebacks = count_comebacks(away_df.tail(20))
    factors['comeback_factor'] = (home_comebacks - away_comebacks) / 20
    
    # Consistency factor
    if 'Goals' in home_df.columns and 'Goals' in away_df.columns:
        home_goals_std = home_df['Goals'].std()
        away_goals_std = away_df['Goals'].std()
        factors['consistency_diff'] = 1 / (1 + home_goals_std) - 1 / (1 + away_goals_std)
    else:
        factors['consistency_diff'] = 0
    
    # Momentum (recent trajectory)
    if 'WinRate5' in home_df.columns and 'WinRate10' in home_df.columns:
        home_momentum = home_df['WinRate5'].iloc[-1] - home_df['WinRate10'].iloc[-1]
        away_momentum = away_df['WinRate5'].iloc[-1] - away_df['WinRate10'].iloc[-1]
        factors['momentum_diff'] = home_momentum - away_momentum
    else:
        factors['momentum_diff'] = 0
    
    return factors

def enhanced_correlation_prediction(home_feats, away_feats, base_weights):
    """Complete correlation-based prediction system."""
    
    # 1. Calculate tactical mismatches
    mismatches = calculate_tactical_mismatches(home_feats, away_feats)
    
    # 2. Get contextual weights
    context_weights = calculate_contextual_weights(home_feats, away_feats, base_weights)
    
    # 3. Create interaction features
    interactions = create_interaction_features(home_feats, away_feats)
    
    # 4. Calculate correlation score
    correlation_score = 0
    
    # Add mismatch contributions
    for mismatch_key, value in mismatches.items():
        weight = TACTICAL_WEIGHTS.get(mismatch_key, 1.0)
        correlation_score += value * weight
    
    # Add interaction contributions
    for interaction_key, value in interactions.items():
        correlation_score += value * 0.5
    
    # 5. Convert to probability adjustments
    prob_adjustment = np.tanh(correlation_score / 10)
    
    # 6. Identify key factors
    all_factors = {**mismatches, **interactions}
    key_factors = sorted(all_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return {
        'correlation_score': correlation_score,
        'prob_adjustment': prob_adjustment,
        'key_mismatches': sorted(mismatches.items(), key=lambda x: abs(x[1]), reverse=True)[:3],
        'key_interactions': sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True)[:3],
        'context_weights': context_weights,
        'all_factors': all_factors
    }

# -------------------------
# Enhanced Feature Engineering
# -------------------------
def calculate_historical_features(df):
    """Berekent historische prestatie-features."""
    features = {}
    n = len(df)
    
    if n == 0:
        default_features = {
            'WinStreak': 0, 'UnbeatenStreak': 0, 'LossStreak': 0,
            'WinRate5': 0, 'WinRate10': 0, 'PointsRate5': 0, 'PointsRate10': 0,
            'RestDays': 7, 'RecentForm': 0
        }
        return {k: pd.Series([v]) for k, v in default_features.items()}
    
    date_col = find_column(df, ['date'])
    if date_col:
        df = df.copy()
        df['parsed_date'] = df[date_col].apply(parse_date)
        df = df.sort_values('parsed_date').reset_index(drop=True)
    
    result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
    goals_col = find_column_flexible(df, [['gf', 'shooting'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    ga_col = find_column_flexible(df, [['ga', 'shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_']])
    
    results = []
    goals_for = []
    goals_against = []
    
    for i in range(n):
        if result_col:
            result_str = df[result_col].iloc[i]
            goals_for_val = series_to_numeric(df[goals_col]).iloc[i] if goals_col else None
            result = extract_match_result_from_string(result_str, goals_for_val)
            results.append(result)
        else:
            results.append(None)
        
        goals_for.append(series_to_numeric(df[goals_col]).iloc[i] if goals_col else 0)
        goals_against.append(series_to_numeric(df[ga_col]).iloc[i] if ga_col else 0)
    
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
        current_streak_w = 0
        current_streak_u = 0
        current_streak_l = 0
        
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
                break
        
        win_streaks.append(current_streak_w)
        unbeaten_streaks.append(current_streak_u)
        loss_streaks.append(current_streak_l)
        
        last_5_results = results[max(0, i-4):i+1] if i >= 0 else []
        last_10_results = results[max(0, i-9):i+1] if i >= 0 else []
        
        wins_5 = sum(1 for r in last_5_results if r == 'W')
        wins_10 = sum(1 for r in last_10_results if r == 'W')
        
        win_rates_5.append(wins_5 / max(1, len(last_5_results)))
        win_rates_10.append(wins_10 / max(1, len(last_10_results)))
        
        points_5 = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in last_5_results)
        points_10 = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in last_10_results)
        
        points_rates_5.append(points_5 / max(1, len(last_5_results) * 3))
        points_rates_10.append(points_10 / max(1, len(last_10_results) * 3))
        
        weights = [0.4, 0.3, 0.2, 0.1]
        form_score = 0
        for j, r in enumerate(reversed(last_5_results[-4:])):
            if j < len(weights):
                if r == 'W':
                    form_score += 3 * weights[j]
                elif r == 'D':
                    form_score += 1 * weights[j]
        recent_forms.append(form_score)
        
        if date_col and i > 0:
            current_date = df['parsed_date'].iloc[i]
            previous_date = df['parsed_date'].iloc[i-1]
            if pd.notna(current_date) and pd.notna(previous_date):
                rest_days.append((current_date - previous_date).days)
            else:
                rest_days.append(7)
        else:
            rest_days.append(7)
    
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
    
    venue_col = find_column(df, ['venue'])
    if venue_col:
        home_results = []
        for i in range(n):
            venue = str(df[venue_col].iloc[i]).lower()
            is_home = 'home' in venue
            
            home_points = 0
            away_points = 0
            home_games = 0
            away_games = 0
            
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
            home_advantage = (home_avg - away_avg) / 3.0
            
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
    """Enhanced feature building with tactical analysis support."""
    n = len(df)
    feats = {}
    minutes = pd.Series([90.0] * n)

    # Enhanced column detection
    xg_col = find_column_flexible(df, [['expected_xg_shooting'], ['xg_shooting'], ['xg'], ['npxg_shooting'], ['npxg']])
    sh_col = find_column_flexible(df, [['standard_sh_shooting'], ['sh_shooting'], ['sh'], ['shots'], ['total', 'shots']])
    sot_col = find_column_flexible(df, [['standard_sot_shooting'], ['sot_shooting'], ['sot'], ['shots on target'], ['on target']])
    goals_col = find_column_flexible(df, [['standard_gls_shooting'], ['gls_shooting'], ['gf_shooting'], ['gf'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    
    prgp_col = find_column_flexible(df, [['prgp_passing'], ['prgr_possession'], ['prgp'], ['progressive', 'passes']])
    prgdist_col = find_column_flexible(df, [['total_prgdist_passing'], ['prgdist_possession'], ['prgdist'], ['progressive', 'distance'], ['pass', 'prgdist']])
    
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
    
    att3rd_col = find_column_flexible(df, [['touches_att 3rd_possession'], ['touches_att_3rd_possession'], ['att_3rd_possession'], ['att_3rd'], ['att', '3rd']])
    poss_col = find_column_flexible(df, [['poss_possession'], ['possession'], ['poss', '%'], ['possession', '%']])
    
    sota_col = find_column_flexible(df, [['performance_sota_keeper'], ['sota_keeper'], ['sota'], ['shots on target against']])
    saves_col = find_column_flexible(df, [['performance_saves_keeper'], ['saves_keeper'], ['saves']])
    psxg_col = find_column_flexible(df, [['performance_psxg_keeper'], ['psxg_keeper'], ['psxg']])
    
    # ENHANCED AERIAL DETECTION
    aerial_win_col = find_column_flexible(df, [
        ['wonpct_misc'],
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
        ['header', 'won', '%'],
        ['aerial_duels_won'],
        ['aerial_success'],
        ['duels_aerial_won']
    ])
    
    # Defensive actions
    def3rd_col = find_column_flexible(df, [['tackles_def 3rd_defense'], ['def_3rd_defense'], ['defensive', '3rd']])
    int_col = find_column_flexible(df, [['performance_int_misc'], ['int_misc'], ['interceptions'], ['int']])
    tkldpct_col = find_column_flexible(df, [['tkldpct_possession'], ['tkld%_possession'], ['tackled', '%']])
    wonpct_col = find_column_flexible(df, [['won%_misc'], ['duels', 'won%'], ['won', '%']])
    att3rddef_col = find_column_flexible(df, [['tackles_att 3rd_defense'], ['att_3rd_defense'], ['attacking', '3rd', 'tackles']])
    
    # Basic features per 90 minutes
    def per90(col):
        return series_to_numeric(df[col]) / minutes * 90.0 if col else pd.Series([0.0] * n)
    
    feats['xG90'] = per90(xg_col)
    feats['Sh90'] = per90(sh_col)
    feats['SoT90'] = per90(sot_col)

    # Shot quality calculation
    sh_safe = series_to_numeric(df[sh_col].replace(0, np.nan)) if sh_col else pd.Series([1.0] * n)
    feats['ShotQual'] = 0.6 * (series_to_numeric(df[xg_col]) / sh_safe).fillna(0.0) + 0.4 * (series_to_numeric(df[sot_col]) / sh_safe).fillna(0.0) if xg_col and sh_col else pd.Series([0.0] * n)
    feats['Goals'] = series_to_numeric(df[goals_col]).fillna(0.0) if goals_col else pd.Series([0.0] * n)

    # Goals against calculation
    GA_WEIGHT = 0.25
    ga_col = find_column_flexible(df, [['standard_ga_shooting'], ['ga_shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_'], ['ga']])
    if ga_col:
        feats['GoalsAgainst'] = series_to_numeric(df[ga_col]).fillna(0.0)
    else:
        res_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
        parsed_ga = []
        if res_col:
            goals_series = series_to_numeric(df[goals_col]).fillna(np.nan) if goals_col else pd.Series([np.nan] * n, index=df.index)
            for idx_row, row in df.iterrows():
                res_val = row.get(res_col, '') if isinstance(row, dict) or isinstance(row, pd.Series) else ''
                if pd.isna(res_val): 
                    res_val = ''
                res_str = str(res_val)
                mres = re.search(r'(\d+)\s*[-–":\u2013]\s*(\d+)', res_str)
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
    if poss_raw.max() > 1.5:
        feats['Possession'] = poss_raw / 100.0
    else:
        feats['Possession'] = poss_raw

    feats['SoTA90'] = per90(sota_col)
    feats['SaveRate'] = (series_to_numeric(df[saves_col]) / series_to_numeric(df[sota_col])).fillna(0.0) if saves_col and sota_col else pd.Series([0.0] * n)
    feats['PSxG'] = series_to_numeric(df[psxg_col]).fillna(0.0) if psxg_col else pd.Series([0.0] * n)

    # ENHANCED AERIAL WIN PERCENTAGE
    if aerial_win_col:
        aerial_raw = series_to_numeric(df[aerial_win_col])
        if aerial_raw.max() > 1.5:
            feats['AerialWin%'] = aerial_raw / 100.0
        else:
            feats['AerialWin%'] = aerial_raw
        feats['AerialWin%'] = feats['AerialWin%'].clip(0.0, 1.0)
    else:
        if wonpct_col:
            won_raw = series_to_numeric(df[wonpct_col])
            if won_raw.max() > 1.5:
                feats['AerialWin%'] = won_raw / 100.0
            else:
                feats['AerialWin%'] = won_raw
        else:
            feats['AerialWin%'] = pd.Series([0.5] * n)

    # High press using defensive actions in attacking third
    feats['HighPress'] = per90(att3rddef_col) if att3rddef_col else pd.Series([0.0] * n)

    # Additional features with percentage handling
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

def make_enhanced_delta(team_feats, opp_feats):
    """Enhanced delta calculation with tactical correlations."""
    delta = {}
    
    # Basic features
    keys = [
        'xG90', 'Sh90', 'SoT90', 'ShotQual', 'ConvRatio90', 'Goals', 'GoalsAgainst',
        'Prog90', 'PrgDist90', 'SetPieces90', 'Att3rd90', 'Possession',
        'TkldPct_possession', 'WonPct_misc', 'Att_3rd_defense',
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
    
    # ENHANCED AERIAL MISMATCH
    team_aerial = ema(team_feats.get('AerialWin%', pd.Series([0.5])))
    opp_aerial = ema(opp_feats.get('AerialWin%', pd.Series([0.5])))
    
    delta['AerialMismatch'] = (team_aerial, opp_aerial)
    
    delta['KeeperPSxGdiff'] = (ema(team_feats.get('PSxG', pd.Series([0.0]))), 
                              ema(opp_feats.get('PSxG', pd.Series([0.0]))))

    return delta

def compute_enhanced_weighted_score_with_correlations(delta_dict, home_feats, away_feats, use_ml_weights=False):
    """Enhanced scoring with tactical correlations and contextual adjustments."""
    
    # Get contextual weights based on opponent characteristics
    weights_to_use = calculate_contextual_weights(home_feats, away_feats, BASE_WEIGHTS)
    
    z_team = {}
    z_opp = {}
    contribs = {}

    print(f"\n--- ENHANCED Z-Score Calculation with Tactical Correlations ---")
    print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Diff':<8} | {'Std':<8} | {'Z-Diff':<8} | {'Weight':<8} | {'Contrib':<8}")
    print("-" * 100)

    for feat, (t_ema, o_ema) in delta_dict.items():
        min_std = MIN_STD_VALUES.get(feat, max(0.1, abs(t_ema) * 0.1))
        combined = np.array([t_ema, o_ema])
        mean, std = np.nanmean(combined), np.nanstd(combined)
        
        if feat == 'AerialMismatch':
            robust_std = max(std, min_std, abs(mean) * 0.08, 0.15)
        else:
            robust_std = max(std, min_std, abs(mean) * 0.15, 0.2)
        
        zt = (t_ema - mean) / robust_std
        zo = (o_ema - mean) / robust_std
        
        if feat in ['RestDays']:
            zt = np.clip(zt, -2.0, 2.0)
            zo = np.clip(zo, -2.0, 2.0)
        elif feat == 'AerialMismatch':
            zt = np.clip(zt, -3.5, 3.5)
            zo = np.clip(zo, -3.5, 3.5)
        else:
            zt = np.clip(zt, -2.5, 2.5)
            zo = np.clip(zo, -2.5, 2.5)
        
        z_team[feat] = zt
        z_opp[feat] = zo
        
        weight = weights_to_use.get(feat, 0.0)
        
        raw_contrib = weight * (zt - zo)
        
        if feat in ['WinRate5', 'WinRate10', 'PointsRate5', 'PointsRate10', 'RecentForm']:
            scaling_factor = 1.15
        elif feat == 'AerialMismatch':
            scaling_factor = 1.5
        elif feat in ['WinStreak', 'LossStreak']:
            scaling_factor = 1.1
        elif feat in ['RestDays']:
            scaling_factor = 0.9
        else:
            scaling_factor = 1.0
        
        clipping_factor = 5.0
        contribs[feat] = raw_contrib * scaling_factor * np.tanh(abs(raw_contrib) / clipping_factor) / max(abs(raw_contrib), 1e-6)
        
        diff = t_ema - o_ema
        z_diff = zt - zo
        
        print(f"{feat:<20} | {t_ema:<10.3f} | {o_ema:<10.3f} | {diff:<+8.3f} | {robust_std:<8.3f} | {z_diff:<+8.3f} | {weight:<8.3f} | {contribs[feat]:<+8.3f}")

    if 'KeeperPSxGdiff' in contribs:
        contribs['KeeperPSxGdiff'] = -contribs['KeeperPSxGdiff']

    # Calculate base weighted difference
    base_weighted_diff = sum(contribs.values())
    
    # Add tactical correlation adjustments
    correlation_analysis = enhanced_correlation_prediction(home_feats, away_feats, weights_to_use)
    correlation_adjustment = correlation_analysis['prob_adjustment'] * 2  # Scale the impact
    
    # Final weighted difference including correlations
    weighted_diff = base_weighted_diff + correlation_adjustment
    
    max_expected_diff = 10.0
    scaled_diff = max_expected_diff * np.tanh(weighted_diff / max_expected_diff)
    
    final = 50.0 + SCALE_TO_SCORE * scaled_diff
    
    confidence = min(1.0, len([c for c in contribs.values() if abs(c) > 0.05]) / 12.0)
    cap_range = 20 + confidence * 25
    
    final = np.clip(final, 30.0, 70.0)
    
    print(f"\nBase weighted difference: {base_weighted_diff:.3f}")
    print(f"Tactical correlation adjustment: {correlation_adjustment:.3f}")
    print(f"Final weighted difference: {weighted_diff:.3f}")
    print(f"Final score: {final:.1f}")
    
    # Print key tactical mismatches
    print(f"\n--- Key Tactical Mismatches ---")
    for mismatch, value in correlation_analysis['key_mismatches']:
        impact = "Favorable" if value > 0 else "Unfavorable"
        print(f"{mismatch}: {value:+.3f} ({impact})")
    
    return final, scaled_diff, z_team, z_opp, contribs, correlation_analysis

# -------------------------
# POISSON MODEL FUNCTIONS
# -------------------------
def simulate_match_probabilities(lambda_home, lambda_away, n_simulations=POISSON_SIMULATIONS):
    """Simulate match outcomes using Poisson distributions."""
    lambda_home = np.atleast_1d(lambda_home)
    lambda_away = np.atleast_1d(lambda_away)
    
    n_matches = len(lambda_home)
    probabilities = np.zeros((n_matches, 3))  # W, D, L
    
    for i in range(n_matches):
        lh = max(0.1, min(10, lambda_home[i]))
        la = max(0.1, min(10, lambda_away[i]))
        
        prob_matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                prob_matrix[h, a] = poisson.pmf(h, lh) * poisson.pmf(a, la)
        
        win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        loss_prob = np.sum(np.triu(prob_matrix, 1))
        
        total = win_prob + draw_prob + loss_prob
        if total > 0:
            probabilities[i] = [win_prob/total, draw_prob/total, loss_prob/total]
        else:
            probabilities[i] = [1/3, 1/3, 1/3]
    
    return probabilities

def predict_with_poisson(poisson_home, poisson_away, scaler, features):
    """Make predictions using Poisson models."""
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    
    lambda_home = np.clip(poisson_home.predict(features_scaled)[0], 0.1, 10)
    lambda_away = np.clip(poisson_away.predict(features_scaled)[0], 0.1, 10)
    
    probabilities = simulate_match_probabilities(
        np.array([lambda_home]), 
        np.array([lambda_away])
    )[0]
    
    prediction = ['W', 'D', 'L'][np.argmax(probabilities)]
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'expected_home_goals': lambda_home,
        'expected_away_goals': lambda_away
    }

def train_poisson_models(X, y_goals_home, y_goals_away):
    """Train Poisson regression models for goal prediction."""
    print("\n=== Training Poisson Regression Models ===")
    
    if len(X) < 30:
        print("Onvoldoende data voor Poisson regressie (minimum 30 samples)")
        return None, None, None
    
    # Clean the data
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], 0)
    X_clean = X_clean.fillna(0)
    
    percentile_99 = X_clean.quantile(0.99)
    percentile_1 = X_clean.quantile(0.01)
    X_clean = X_clean.clip(lower=percentile_1, upper=percentile_99, axis=1)
    
    y_goals_home = np.clip(np.array(y_goals_home), 0, 10)
    y_goals_away = np.clip(np.array(y_goals_away), 0, 10)
    
    valid_mask = (
        ~np.isnan(y_goals_home) & 
        ~np.isnan(y_goals_away) & 
        ~np.isinf(y_goals_home) & 
        ~np.isinf(y_goals_away) &
        (y_goals_home >= 0) & 
        (y_goals_away >= 0) &
        (y_goals_home <= 10) &
        (y_goals_away <= 10)
    )
    
    X_clean = X_clean[valid_mask]
    y_goals_home = y_goals_home[valid_mask]
    y_goals_away = y_goals_away[valid_mask]
    
    print(f"Valid samples after cleaning: {len(X_clean)}/{len(X)}")
    print(f"Goal statistics after cleaning:")
    print(f"  Home: Mean={y_goals_home.mean():.2f}, Std={y_goals_home.std():.2f}")
    print(f"  Away: Mean={y_goals_away.mean():.2f}, Std={y_goals_away.std():.2f}")
    
    if len(X_clean) < 30:
        print("Te weinig valid samples na cleaning")
        return None, None, None
    
    test_size = min(0.3, max(0.1, len(X_clean) * 0.2 / len(X_clean)))
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X_clean, y_goals_home, y_goals_away, test_size=test_size, random_state=42
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Poisson models
    print("\nTraining Poisson model for HOME goals...")
    poisson_home = PoissonRegressor(alpha=1.0, max_iter=500, tol=1e-3)
    poisson_home.fit(X_train_scaled, y_home_train)
    
    home_pred = np.clip(poisson_home.predict(X_test_scaled), 0, 10)
    home_mae = np.mean(np.abs(home_pred - y_home_test))
    print(f"Home goals MAE: {home_mae:.3f}")
    print(f"Home goals mean predicted: {home_pred.mean():.2f}, actual: {y_home_test.mean():.2f}")
    
    print("\nTraining Poisson model for AWAY goals...")
    poisson_away = PoissonRegressor(alpha=1.0, max_iter=500, tol=1e-3)
    poisson_away.fit(X_train_scaled, y_away_train)
    
    away_pred = np.clip(poisson_away.predict(X_test_scaled), 0, 10)
    away_mae = np.mean(np.abs(away_pred - y_away_test))
    print(f"Away goals MAE: {away_mae:.3f}")
    print(f"Away goals mean predicted: {away_pred.mean():.2f}, actual: {y_away_test.mean():.2f}")
    
    # Validate with probabilities
    test_probs = simulate_match_probabilities(home_pred, away_pred)
    
    test_results = []
    for h, a in zip(y_home_test, y_away_test):
        if h > a:
            test_results.append('W')
        elif h == a:
            test_results.append('D')
        else:
            test_results.append('L')
    
    le = LabelEncoder()
    le.fit(['W', 'D', 'L'])
    y_test_encoded = le.transform(test_results)
    
    brier = calculate_brier_score(y_test_encoded, test_probs)
    print(f"\nPoisson Model Brier Score: {brier:.4f}")
    print(f"Probability distribution - W: {test_probs[:, 0].mean():.3f}, D: {test_probs[:, 1].mean():.3f}, L: {test_probs[:, 2].mean():.3f}")
    
    return poisson_home, poisson_away, scaler

# -------------------------
# CALIBRATED CLASSIFIER FUNCTIONS
# -------------------------
def train_calibrated_classifiers(X, y):
    """Train calibrated versions of classifiers with tactical features."""
    print("\n=== Training Calibrated Classifiers ===")
    
    if len(X) < 30:
        print("Onvoldoende data voor calibrated classifiers (minimum 30 samples)")
        return None, None, None, None
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    
    calibrated_models = {}
    brier_scores = {}
    
    # Random Forest with Calibration
    print("\nTraining Calibrated Random Forest...")
    rf_base = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        max_features='sqrt'
    )
    rf_base.fit(X_train_scaled, y_train)
    
    rf_calibrated = CalibratedClassifierCV(
        rf_base, method='isotonic', cv='prefit'
    )
    rf_calibrated.fit(X_cal_scaled, y_cal)
    calibrated_models['rf'] = rf_calibrated
    
    # Evaluate calibration
    rf_probs = rf_calibrated.predict_proba(X_cal_scaled)
    rf_pred = rf_calibrated.predict(X_cal_scaled)
    rf_brier = calculate_brier_score(y_cal, rf_probs)
    brier_scores['rf'] = rf_brier
    
    print(f"Calibrated RF Brier Score: {rf_brier:.4f}")
    
    # Classification report
    y_cal_labels = le.inverse_transform(y_cal)
    rf_pred_labels = le.inverse_transform(rf_pred)
    print("\nRF Classification Report:")
    print(classification_report(y_cal_labels, rf_pred_labels, digits=3))
    
    # XGBoost with Calibration
    print("\n" + "="*50)
    print("Training Calibrated XGBoost...")
    xgb_base = xgb.XGBClassifier(**XGBOOST_PARAMS)
    xgb_base.fit(X_train_scaled, y_train)
    
    xgb_calibrated = CalibratedClassifierCV(
        xgb_base, method='isotonic', cv='prefit'
    )
    xgb_calibrated.fit(X_cal_scaled, y_cal)
    calibrated_models['xgb'] = xgb_calibrated
    
    # Evaluate calibration
    xgb_probs = xgb_calibrated.predict_proba(X_cal_scaled)
    xgb_pred = xgb_calibrated.predict(X_cal_scaled)
    xgb_brier = calculate_brier_score(y_cal, xgb_probs)
    brier_scores['xgb'] = xgb_brier
    
    print(f"Calibrated XGBoost Brier Score: {xgb_brier:.4f}")
    
    # Classification report
    xgb_pred_labels = le.inverse_transform(xgb_pred)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_cal_labels, xgb_pred_labels, digits=3))
    
    # LightGBM with Calibration
    print("\n" + "="*50)
    print("Training Calibrated LightGBM...")
    
    lgb_base = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=300,
        num_leaves=25,
        learning_rate=0.08,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_child_samples=25,
        random_state=42,
        verbosity=-1,
        reg_alpha=0.05,
        reg_lambda=0.05,
        min_gain_to_split=0.1,
        force_col_wise=True
    )
    
    lgb_base.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_cal_scaled, y_cal)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    lgb_calibrated = CalibratedClassifierCV(
        lgb_base, method='isotonic', cv='prefit'
    )
    lgb_calibrated.fit(X_cal_scaled, y_cal)
    calibrated_models['lgb'] = lgb_calibrated
    
    lgb_probs = lgb_calibrated.predict_proba(X_cal_scaled)
    lgb_pred = lgb_calibrated.predict(X_cal_scaled)
    lgb_brier = calculate_brier_score(y_cal, lgb_probs)
    brier_scores['lgb'] = lgb_brier
    
    print(f"Calibrated LightGBM Brier Score: {lgb_brier:.4f}")
    
    lgb_pred_labels = le.inverse_transform(lgb_pred)
    print("\nLightGBM Classification Report:")
    print(classification_report(y_cal_labels, lgb_pred_labels, digits=3))
    
    # Summary
    print("\n" + "="*50)
    print("=== CALIBRATION SUMMARY ===")
    print("="*50)
    
    avg_brier = np.mean([rf_brier, xgb_brier, lgb_brier])
    print(f"\nBrier Scores (lower is better):")
    print(f"  Random Forest: {rf_brier:.4f}")
    print(f"  XGBoost:       {xgb_brier:.4f}")
    print(f"  LightGBM:      {lgb_brier:.4f}")
    print(f"  Average:       {avg_brier:.4f}")
    
    return calibrated_models, scaler, le, brier_scores

# -------------------------

# -------------------------
# Enhanced ML Feature Preparation with Tactical Correlations
# -------------------------
def prepare_enhanced_ml_features_with_tactical(home_feats, away_feats):
    """Enhanced ML feature preparation with tactical analysis."""
    # Basic features
    home_ema = {f"home_{k}": ema(v) for k, v in home_feats.items()}
    away_ema = {f"away_{k}": ema(v) for k, v in away_feats.items()}
    
    combined_features = {**home_ema, **away_ema}
    
    # Differences and ratios
    for key in home_feats.keys():
        if key in away_feats:
            home_val = home_ema.get(f"home_{key}", 0)
            away_val = away_ema.get(f"away_{key}", 0)
            combined_features[f"diff_{key}"] = home_val - away_val
            
            if abs(away_val) > 1e-6:
                combined_features[f"ratio_{key}"] = home_val / away_val
            else:
                combined_features[f"ratio_{key}"] = home_val if abs(home_val) > 1e-6 else 1.0
    
    # Basic interactions
    key_interactions = [
        ('WinRate5', 'RecentForm'),
        ('RestDays', 'WinStreak'),
        ('xG90', 'ShotQual'),
        ('Possession', 'FieldTilt'),
        ('AerialWin%', 'SetPieces90'),
        ('AerialWin%', 'HighPress'),
    ]
    
    for feat1, feat2 in key_interactions:
        home_interaction = home_ema.get(f"home_{feat1}", 0) * home_ema.get(f"home_{feat2}", 0)
        away_interaction = away_ema.get(f"away_{feat1}", 0) * away_ema.get(f"away_{feat2}", 0)
        combined_features[f"interaction_home_{feat1}_{feat2}"] = home_interaction
        combined_features[f"interaction_away_{feat1}_{feat2}"] = away_interaction
        combined_features[f"interaction_diff_{feat1}_{feat2}"] = home_interaction - away_interaction
    
    # Add tactical mismatches
    mismatches = calculate_tactical_mismatches(home_feats, away_feats)
    for key, value in mismatches.items():
        combined_features[f"tactical_{key}"] = value
    
    # Add non-linear interactions
    interactions = create_interaction_features(home_feats, away_feats)
    for key, value in interactions.items():
        combined_features[f"nonlinear_{key}"] = value
    
    return combined_features

# -------------------------
# CNN and Transformer Model Functions
# -------------------------
def create_cnn_model(input_shape, num_classes=3):
    """Create 1D CNN model for tabular data."""
    model = Sequential()
    
    model.add(Conv1D(filters=CNN_PARAMS['filters'][0], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=CNN_PARAMS['pool_size']))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    model.add(Conv1D(filters=CNN_PARAMS['filters'][1], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=CNN_PARAMS['pool_size']))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    model.add(Conv1D(filters=CNN_PARAMS['filters'][2], 
                     kernel_size=CNN_PARAMS['kernel_size'], 
                     activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    model.add(Dense(CNN_PARAMS['dense_units'][0], activation='relu'))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    model.add(Dense(CNN_PARAMS['dense_units'][1], activation='relu'))
    model.add(Dropout(CNN_PARAMS['dropout_rate']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=CNN_PARAMS['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_transformer_model(input_shape, num_classes=3):
    """Create Transformer/Attention model for tabular data."""
    inputs = Input(shape=input_shape)
    
    x = Dense(64, activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = tf.keras.layers.Reshape((8, 8))(x)
    
    attention = MultiHeadAttention(
        num_heads=2,
        key_dim=16,
        dropout=0.2
    )(x, x)
    
    x = tf.keras.layers.Add()([x, attention])
    x = LayerNormalization()(x)
    
    attention2 = MultiHeadAttention(
        num_heads=2,
        key_dim=16,
        dropout=0.2
    )(x, x)
    
    x = tf.keras.layers.Add()([x, attention2])
    x = LayerNormalization()(x)
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax', 
                   kernel_initializer='glorot_uniform')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_cnn_model(X, y):
    """Train CNN model."""
    if len(X) < 10:
        print(f"Onvoldoende data voor CNN training: {len(X)} samples.")
        return None, None, None
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, 
                                                        random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    model = create_cnn_model((X_train_reshaped.shape[1], 1), num_classes=3)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=CNN_PARAMS['patience'], restore_best_weights=True)
    
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=CNN_PARAMS['epochs'],
        batch_size=CNN_PARAMS['batch_size'],
        validation_data=(X_test_reshaped, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"\nCNN Model Performance:")
    print(f"Nauwkeurigheid: {test_acc:.3f}")
    
    return model, scaler, le

def train_transformer_model(X, y):
    """Train Transformer model."""
    if len(X) < 10:
        print(f"Onvoldoende data voor Transformer training: {len(X)} samples.")
        return None, None, None
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, 
                                                        random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = create_transformer_model((X_train_scaled.shape[1],), num_classes=3)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=TRANSFORMER_PARAMS['patience'], restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=TRANSFORMER_PARAMS['epochs'],
        batch_size=TRANSFORMER_PARAMS['batch_size'],
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTransformer Model Performance:")
    print(f"Nauwkeurigheid: {test_acc:.3f}")
    
    return model, scaler, le

# -------------------------
# TACTICAL STYLE CLUSTERING
# -------------------------
def train_style_clustering_model(X):
    """Train a clustering model to identify team playing styles."""
    print("\n=== Training Style Clustering Model ===")
    
    style_features = [
        col for col in X.columns 
        if any(key in col.lower() for key in ['possession', 'highpress', 'prgdist', 'setpieces', 'aerial', 'fieldtilt'])
    ]
    
    if len(style_features) < 3:
        print("Insufficient features for style clustering")
        return None, None
    
    X_style = X[style_features].fillna(0)
    
    scaler = StandardScaler()
    X_style_scaled = scaler.fit_transform(X_style)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    K_range = range(3, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_style_scaled)
        inertias.append(kmeans.inertia_)
    
    # Use 5 clusters as default
    final_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    style_labels = final_kmeans.fit_predict(X_style_scaled)
    
    # Analyze cluster characteristics
    print("\n--- Playing Style Clusters ---")
    for i in range(5):
        cluster_data = X_style[style_labels == i]
        print(f"\nStyle {i} ({PLAYING_STYLES.get(i, 'Unknown')}):")
        print(f"  Samples: {len(cluster_data)}")
        if len(cluster_data) > 0:
            print(f"  Key characteristics:")
            means = cluster_data.mean()
            top_features = means.nlargest(3)
            for feat, val in top_features.items():
                print(f"    {feat}: {val:.2f}")
    
    return final_kmeans, scaler

# -------------------------
# ENHANCED TRAINING WITH TACTICAL ANALYSIS
# -------------------------
def train_all_enhanced_models_with_tactical(X, y, y_goals_home=None, y_goals_away=None):
    """Train all models including Poisson, calibrated, and tactical analysis."""
    if len(X) < 30:
        print(f"Onvoldoende data voor enhanced training: {len(X)} samples. Minimum 30 vereist.")
        return None
    
    models = {}
    scalers = {}
    
    # Train Style Clustering
    print("\n" + "="*60)
    print("TRAINING TACTICAL STYLE CLUSTERING")
    print("="*60)
    style_model, style_scaler = train_style_clustering_model(X)
    if style_model is not None:
        models['style_clustering'] = style_model
        scalers['style_scaler'] = style_scaler
    
    # Train Poisson models if goal data available
    if y_goals_home is not None and y_goals_away is not None:
        print("\n" + "="*60)
        print("TRAINING POISSON GOAL PREDICTION MODELS")
        print("="*60)
        poisson_home, poisson_away, poisson_scaler = train_poisson_models(X, y_goals_home, y_goals_away)
        if poisson_home is not None:
            models['poisson_home'] = poisson_home
            models['poisson_away'] = poisson_away
            scalers['poisson_scaler'] = poisson_scaler
    
    # Train calibrated classifiers
    print("\n" + "="*60)
    print("TRAINING CALIBRATED PROBABILISTIC CLASSIFIERS")
    print("="*60)
    calibrated_models, cal_scaler, label_encoder, brier_scores = train_calibrated_classifiers(X, y)
    if calibrated_models is not None:
        models.update({f'calibrated_{k}': v for k, v in calibrated_models.items()})
        scalers['calibrated_scaler'] = cal_scaler
        models['brier_scores'] = brier_scores
    
    # Train neural networks
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK MODELS")
    print("="*60)
    
    print("\nTraining CNN model...")
    cnn_model, cnn_scaler, _ = train_cnn_model(X, y)
    if cnn_model is not None:
        models['cnn'] = cnn_model
        scalers['cnn_scaler'] = cnn_scaler
    
    print("\nTraining Transformer model...")
    transformer_model, transformer_scaler, _ = train_transformer_model(X, y)
    if transformer_model is not None:
        models['transformer'] = transformer_model
        scalers['transformer_scaler'] = transformer_scaler
    
    # Save all models
    if len(models) > 0:
        try:
            # Save tactical models
            if 'style_clustering' in models:
                with open(STYLE_CLUSTERING_FILE, 'wb') as f:
                    pickle.dump(models['style_clustering'], f)
            
            # Save Poisson models
            if 'poisson_home' in models:
                with open(POISSON_HOME_MODEL_FILE, 'wb') as f:
                    pickle.dump(models['poisson_home'], f)
                with open(POISSON_AWAY_MODEL_FILE, 'wb') as f:
                    pickle.dump(models['poisson_away'], f)
            
            # Save calibrated models
            for key in ['calibrated_rf', 'calibrated_xgb', 'calibrated_lgb']:
                if key in models:
                    filename = f"{key.replace('calibrated_', 'calibrated_')}_model.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(models[key], f)
            
            # Save neural networks
            if 'cnn' in models:
                models['cnn'].save(CNN_MODEL_FILE)
            
            if 'transformer' in models:
                models['transformer'].save(TRANSFORMER_MODEL_FILE)
            
            # Save scalers and label encoder
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(scalers, f)
            
            with open(LABEL_ENCODER_FILE, 'wb') as f:
                pickle.dump(label_encoder, f)
            
            # Save tactical correlation weights
            with open(TACTICAL_CORRELATION_FILE, 'wb') as f:
                pickle.dump(TACTICAL_WEIGHTS, f)
            
            print(f"\nAlle enhanced modellen met tactical analysis opgeslagen!")
            print(f"Getrainde modellen: {list(models.keys())}")
            
        except Exception as e:
            print(f"Fout bij opslaan modellen: {e}")
    
    return models, scalers, label_encoder

def load_all_enhanced_models_with_tactical():
    """Load all enhanced models including tactical analysis."""
    models = {}
    scalers = {}
    
    try:
        # Load style clustering
        try:
            with open(STYLE_CLUSTERING_FILE, 'rb') as f:
                models['style_clustering'] = pickle.load(f)
            print("Style clustering model geladen.")
        except FileNotFoundError:
            print("Style clustering model niet gevonden.")
        
        # Load Poisson models
        try:
            with open(POISSON_HOME_MODEL_FILE, 'rb') as f:
                models['poisson_home'] = pickle.load(f)
            with open(POISSON_AWAY_MODEL_FILE, 'rb') as f:
                models['poisson_away'] = pickle.load(f)
            print("Poisson models geladen.")
        except FileNotFoundError:
            print("Poisson models niet gevonden.")
        
        # Load calibrated models
        for model_type in ['rf', 'xgb', 'lgb']:
            filename = f"calibrated_{model_type}_model.pkl"
            try:
                with open(filename, 'rb') as f:
                    models[f'calibrated_{model_type}'] = pickle.load(f)
                print(f"Calibrated {model_type.upper()} model geladen.")
            except FileNotFoundError:
                print(f"Calibrated {model_type.upper()} model niet gevonden.")
        
        # Load neural networks
        try:
            models['cnn'] = tf.keras.models.load_model(CNN_MODEL_FILE)
            print("CNN model geladen.")
        except:
            print("CNN model niet gevonden.")
        
        try:
            models['transformer'] = tf.keras.models.load_model(TRANSFORMER_MODEL_FILE)
            print("Transformer model geladen.")
        except:
            print("Transformer model niet gevonden.")
        
        # Load scalers
        try:
            with open(SCALER_FILE, 'rb') as f:
                scalers = pickle.load(f)
            print("Scalers geladen.")
        except FileNotFoundError:
            print("Scalers niet gevonden.")
        
        # Load label encoder
        try:
            with open(LABEL_ENCODER_FILE, 'rb') as f:
                label_encoder = pickle.load(f)
            print("Label encoder geladen.")
        except FileNotFoundError:
            print("Label encoder niet gevonden.")
            label_encoder = None
        
        # Load tactical weights
        try:
            with open(TACTICAL_CORRELATION_FILE, 'rb') as f:
                tactical_weights = pickle.load(f)
            print("Tactical correlation weights geladen.")
        except FileNotFoundError:
            tactical_weights = TACTICAL_WEIGHTS
            print("Using default tactical weights.")
        
        if len(models) > 0:
            print(f"\nTotaal {len(models)} modellen succesvol geladen!")
            return models, scalers, label_encoder, tactical_weights
        else:
            print("Geen modellen gevonden.")
            return None, None, None, None
        
    except Exception as e:
        print(f"Fout bij laden modellen: {e}")
        return None, None, None, None

# -------------------------
# ENHANCED ENSEMBLE PREDICTION WITH TACTICAL
# -------------------------
def enhanced_ensemble_prediction_with_tactical(models, scalers, label_encoder, features, home_feats, away_feats):
    """Enhanced ensemble prediction including tactical correlations."""
    features_df = pd.DataFrame([features])
    predictions = {}
    
    # Get tactical correlation analysis
    correlation_analysis = enhanced_correlation_prediction(home_feats, away_feats, BASE_WEIGHTS)
    
    print("\n" + "="*60)
    print("TACTICAL ANALYSIS")
    print("="*60)
    print(f"Correlation Score: {correlation_analysis['correlation_score']:.3f}")
    print(f"Probability Adjustment: {correlation_analysis['prob_adjustment']:+.3f}")
    
    print("\nKey Tactical Mismatches:")
    for mismatch, value in correlation_analysis['key_mismatches']:
        impact = "Favorable" if value > 0 else "Unfavorable"
        print(f"  {mismatch}: {value:+.3f} ({impact})")
    
    print("\nKey Interactions:")
    for interaction, value in correlation_analysis['key_interactions']:
        print(f"  {interaction}: {value:+.3f}")
    
    # Style clustering prediction if available
    if 'style_clustering' in models and 'style_scaler' in scalers:
        style_features = [col for col in features_df.columns 
                         if any(key in col.lower() for key in ['possession', 'highpress', 'prgdist', 'setpieces', 'aerial', 'fieldtilt'])]
        if len(style_features) > 2:
            X_style = features_df[style_features].fillna(0)
            X_style_scaled = scalers['style_scaler'].transform(X_style)
            style_pred = models['style_clustering'].predict(X_style_scaled)[0]
            print(f"\nPredicted Playing Style: {PLAYING_STYLES.get(style_pred, 'Unknown')}")
    
    # Poisson prediction
    if 'poisson_home' in models and 'poisson_away' in models:
        poisson_result = predict_with_poisson(
            models['poisson_home'],
            models['poisson_away'],
            scalers.get('poisson_scaler'),
            features_df.values[0]
        )
        predictions['poisson'] = poisson_result
        print(f"\nPoisson Model:")
        print(f"  Expected goals: Home={poisson_result['expected_home_goals']:.2f}, "
              f"Away={poisson_result['expected_away_goals']:.2f}")
        print(f"  Probabilities: W={poisson_result['probabilities'][0]:.3f}, "
              f"D={poisson_result['probabilities'][1]:.3f}, "
              f"L={poisson_result['probabilities'][2]:.3f}")
    
    # Calibrated model predictions
    calibrated_probs = []
    for model_name in ['calibrated_rf', 'calibrated_xgb', 'calibrated_lgb']:
        if model_name in models:
            cal_features = scalers['calibrated_scaler'].transform(features_df)
            probs = models[model_name].predict_proba(cal_features)[0]
            
            if len(label_encoder.classes_) == 3:
                prob_dict = dict(zip(label_encoder.classes_, probs))
                ordered_probs = [prob_dict.get('W', 0), prob_dict.get('D', 0), prob_dict.get('L', 0)]
            else:
                ordered_probs = probs
                
            predictions[model_name] = {
                'prediction': ['W', 'D', 'L'][np.argmax(ordered_probs)],
                'probabilities': ordered_probs
            }
            calibrated_probs.append(ordered_probs)
            
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Prediction: {predictions[model_name]['prediction']}")
            print(f"  Probabilities: W={ordered_probs[0]:.3f}, "
                  f"D={ordered_probs[1]:.3f}, L={ordered_probs[2]:.3f}")
    
    # Neural network predictions
    for model_name in ['cnn', 'transformer']:
        if model_name in models:
            nn_scaler = scalers.get(f'{model_name}_scaler')
            if nn_scaler:
                nn_features = nn_scaler.transform(features_df)
                
                if model_name == 'cnn':
                    nn_features = nn_features.reshape(nn_features.shape[0], nn_features.shape[1], 1)
                
                probs = models[model_name].predict(nn_features)[0]
                prob_dict = dict(zip(label_encoder.classes_, probs))
                ordered_probs = [prob_dict.get('W', 0), prob_dict.get('D', 0), prob_dict.get('L', 0)]
                
                predictions[model_name] = {
                    'prediction': ['W', 'D', 'L'][np.argmax(ordered_probs)],
                    'probabilities': ordered_probs
                }
    
    # Combined ensemble with tactical adjustment
    all_probs = []
    weights = []
    
    if 'poisson' in predictions:
        all_probs.append(predictions['poisson']['probabilities'])
        weights.append(0.35)
    
    for cal_probs in calibrated_probs:
        all_probs.append(cal_probs)
        weights.append(0.20)
    
    for model_name in ['cnn', 'transformer']:
        if model_name in predictions:
            all_probs.append(predictions[model_name]['probabilities'])
            weights.append(0.075)
    
    # Calculate base ensemble
    if len(weights) > 0:
        weights = np.array(weights) / np.sum(weights)
        ensemble_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Apply tactical correlation adjustment
        tactical_adj = correlation_analysis['prob_adjustment'] * 0.15  # Scale the impact
        ensemble_probs[0] += tactical_adj  # Adjust home win
        ensemble_probs[2] -= tactical_adj  # Adjust away win
        
        # Renormalize
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs)
        
        predictions['ensemble'] = {
            'prediction': ['W', 'D', 'L'][np.argmax(ensemble_probs)],
            'probabilities': ensemble_probs
        }
        
        print(f"\n=== FINAL TACTICAL ENSEMBLE ===")
        print(f"Prediction: {predictions['ensemble']['prediction']}")
        print(f"Probabilities: W={ensemble_probs[0]:.3f}, "
              f"D={ensemble_probs[1]:.3f}, L={ensemble_probs[2]:.3f}")
        print(f"Tactical Adjustment Applied: {tactical_adj:+.3f}")
        
        # Calculate confidence
        max_prob = max(ensemble_probs)
        if max_prob > 0.6:
            confidence = "HIGH"
        elif max_prob > 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        print(f"Confidence: {confidence}")
    
    return predictions, correlation_analysis

def extract_match_result_enhanced(df):
    """Enhanced result extraction."""
    if isinstance(df, pd.DataFrame):
        if len(df) == 0:
            return None
        row = df.iloc[-1]
    else:
        row = df
    
    result_col = find_column_flexible(pd.DataFrame([row]), [['result'], ['score'], ['result_shooting']])
    
    if result_col is None:
        return None
    
    result = row.get(result_col)
    
    if result is None or pd.isna(result):
        return None
    
    goals_col = find_column_flexible(pd.DataFrame([row]), [['gf_shooting'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    goals_for = series_to_numeric(pd.Series([row.get(goals_col)])).iloc[0] if goals_col else None
    
    return extract_match_result_from_string(result, goals_for)

def extract_goals_from_matches(df):
    """Extract home and away goals from historical matches."""
    goals_col = find_column_flexible(df, [['gf_shooting'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    ga_col = find_column_flexible(df, [['ga_shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_']])
    venue_col = find_column(df, ['venue'])
    
    if goals_col is None or ga_col is None:
        return None, None
    
    goals_for = series_to_numeric(df[goals_col])
    goals_against = series_to_numeric(df[ga_col])
    
    if venue_col:
        venues = df[venue_col].str.lower()
        home_mask = venues.str.contains('home')
        
        home_goals = np.where(home_mask, goals_for, goals_against)
        away_goals = np.where(home_mask, goals_against, goals_for)
    else:
        home_goals = goals_for
        away_goals = goals_against
    
    return home_goals, away_goals

# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == '__main__':
    print("="*70)
    print("ENHANCED SOCCER PREDICTOR v8.0")
    print("WITH TACTICAL CORRELATIONS & STYLE ANALYSIS")
    print("="*70)
    
    print("\nKies een optie:")
    print("1. Train COMPLETE models (Poisson + Calibrated + Neural + Tactical)")
    print("2. Voorspelling met tactical correlation analysis")
    print("3. Team style analysis en matchup history")
    print("4. Tactical mismatch rapport")
    print("5. Vergelijk voorspellingen met en zonder tactical correlations")
    
    choice = input("\nJouw keuze (1-5): ").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("TRAINING COMPLETE TACTICAL MODELS")
        print("="*60)
        
        training_files = choose_files('Selecteer CSV bestanden voor training (meerdere mogelijk)')
        
        if not training_files:
            print("Geen bestanden geselecteerd.")
            exit()
        
        if isinstance(training_files, tuple):
            training_files = list(training_files)
        
        # Collect training data
        X_data = []
        y_data = []
        y_goals_home = []
        y_goals_away = []
        
        print(f"\nVerwerken van {len(training_files)} trainingsbestanden...")
        for file_idx, file_path in enumerate(training_files, 1):
            print(f"\nBestand {file_idx}/{len(training_files)}: {file_path}")
            
            try:
                h_rows = detect_header_rows(file_path)
                df = pd.read_csv(file_path, header=list(range(h_rows)) if h_rows > 1 else 0, low_memory=False)
                
                print(f"  Geladen: {len(df)} wedstrijden")
                
                # Extract goals for Poisson model
                goals_home, goals_away = extract_goals_from_matches(df)
                
                for i in range(max(7, EMA_SPAN), len(df)):
                    try:
                        home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                        away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                        
                        if len(home_subset) >= 4 and len(away_subset) >= 4:
                            home_feats = build_enhanced_feature_series(home_subset, "HOME")
                            away_feats = build_enhanced_feature_series(away_subset, "AWAY")
                            
                            # Use enhanced features with tactical analysis
                            features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
                            
                            result = extract_match_result_enhanced(df.iloc[:i+1])
                            
                            if result and features:
                                X_data.append(features)
                                y_data.append(result)
                                
                                # Add goal data if available
                                if goals_home is not None and goals_away is not None:
                                    y_goals_home.append(goals_home[i])
                                    y_goals_away.append(goals_away[i])
                                
                    except Exception as e:
                        print(f"  Fout bij wedstrijd {i}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Fout bij laden bestand: {e}")
                continue
        
        print(f"\nTotaal verzamelde samples: {len(X_data)}")
        
        if len(X_data) < 30:
            print("Onvoldoende trainingsdata verzameld (minimum 30 samples vereist).")
            exit()
        
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        print(f"\nFeature dimensies: {X_df.shape}")
        print(f"Label distributie:")
        label_counts = y_series.value_counts()
        print(label_counts)
        print(f"Draw percentage: {label_counts.get('D', 0) / len(y_series) * 100:.1f}%")
        
        # Check for tactical features
        tactical_features = [col for col in X_df.columns if 'tactical_' in col or 'nonlinear_' in col]
        print(f"\nTactical features: {len(tactical_features)} gevonden")
        if tactical_features:
            print(f"Voorbeelden: {tactical_features[:5]}")
        
        # Prepare goal data
        if len(y_goals_home) == len(X_data):
            y_goals_home = np.array(y_goals_home)
            y_goals_away = np.array(y_goals_away)
            print(f"\nGoal statistics:")
            print(f"  Home goals - Mean: {y_goals_home.mean():.2f}, Std: {y_goals_home.std():.2f}")
            print(f"  Away goals - Mean: {y_goals_away.mean():.2f}, Std: {y_goals_away.std():.2f}")
        else:
            print("\nGoal data niet volledig beschikbaar voor Poisson training")
            y_goals_home = None
            y_goals_away = None
        
        models, scalers, label_encoder = train_all_enhanced_models_with_tactical(
            X_df, y_series, y_goals_home, y_goals_away
        )
        
        if models and len(models) > 0:
            print("\n" + "="*60)
            print("TRAINING COMPLETE WITH TACTICAL ANALYSIS!")
            print("="*60)
            print(f"Succesvol getrainde modellen: {list(models.keys())}")
            
            if 'brier_scores' in models:
                print("\nModel Calibration Quality (lower is better):")
                for model, score in models['brier_scores'].items():
                    print(f"  {model.upper()}: {score:.4f}")
        
    elif choice == "2":
        print("\n" + "="*60)
        print("ENHANCED PREDICTION WITH TACTICAL CORRELATIONS")
        print("="*60)
        
        models, scalers, label_encoder, tactical_weights = load_all_enhanced_models_with_tactical()
        if models is None or len(models) == 0:
            print("Geen modellen gevonden. Train eerst de modellen (optie 1).")
            exit()
        
        print(f"\nBeschikbare modellen: {list(models.keys())}")
        
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

        # ML predictions with tactical analysis
        features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
        
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL PREDICTIONS")
        print("="*60)
        
        predictions, correlation_analysis = enhanced_ensemble_prediction_with_tactical(
            models, scalers, label_encoder, features, home_feats, away_feats
        )
        
        # Statistical analysis with tactical correlations
        delta = make_enhanced_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs, corr_analysis = compute_enhanced_weighted_score_with_correlations(
            delta, home_feats, away_feats, use_ml_weights=True
        )
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS WITH TACTICAL CORRELATIONS")
        print("="*60)
        print(f"Base Statistical Score: {final_score:.1f}")
        print(f"Weighted Difference: {weighted_diff:.3f}")
        
        # Tactical impact analysis
        print("\n" + "="*60)
        print("TACTICAL IMPACT ANALYSIS")
        print("="*60)
        
        # Sort all factors by impact
        all_tactical_factors = correlation_analysis['all_factors']
        sorted_factors = sorted(all_tactical_factors.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 10 Tactical Factors:")
        for i, (factor, value) in enumerate(sorted_factors[:10], 1):
            impact = "+" if value > 0 else ""
            print(f"{i:2}. {factor:<35}: {impact}{value:.3f}")
        
        # Calculate psychological factors
        psych_factors = calculate_psychological_factors(home_df, away_df)
        print("\nPsychological Factors:")
        for factor, value in psych_factors.items():
            print(f"  {factor}: {value:+.3f}")
        
        # Final summary
        if 'ensemble' in predictions:
            ensemble_probs = predictions['ensemble']['probabilities']
            print("\n" + "="*60)
            print("FINAL TACTICAL PREDICTION SUMMARY")
            print("="*60)
            print(f"Outcome: {predictions['ensemble']['prediction']}")
            print(f"Win Probability: {ensemble_probs[0]:.1%}")
            print(f"Draw Probability: {ensemble_probs[1]:.1%}")
            print(f"Loss Probability: {ensemble_probs[2]:.1%}")
            
            # Tactical assessment
            tactical_score = correlation_analysis['correlation_score']
            if abs(tactical_score) > 2:
                tactical_assessment = "STRONG TACTICAL ADVANTAGE"
            elif abs(tactical_score) > 1:
                tactical_assessment = "MODERATE TACTICAL EDGE"
            else:
                tactical_assessment = "BALANCED TACTICAL MATCHUP"
            
            if tactical_score > 0:
                tactical_assessment += " (HOME)"
            elif tactical_score < 0:
                tactical_assessment += " (AWAY)"
                
            print(f"\nTactical Assessment: {tactical_assessment}")
            print(f"Tactical Score: {tactical_score:+.2f}")
            
    elif choice == "3":
        print("\n" + "="*60)
        print("TEAM STYLE ANALYSIS & MATCHUP HISTORY")
        print("="*60)
        
        models, scalers, label_encoder, tactical_weights = load_all_enhanced_models_with_tactical()
        if models is None:
            print("Geen modellen gevonden.")
            exit()
        
        if 'style_clustering' not in models:
            print("Style clustering model niet beschikbaar. Train eerst met optie 1.")
            exit()
        
        home_file = choose_file('Upload THUIS team CSV')
        away_file = choose_file('Upload UIT team CSV')

        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)

        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home > 1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away > 1 else 0, low_memory=False)

        home_feats = build_enhanced_feature_series(home_df, "HOME TEAM")
        away_feats = build_enhanced_feature_series(away_df, "AWAY TEAM")
        
        # Determine playing styles
        features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
        features_df = pd.DataFrame([features])
        
        style_features = [col for col in features_df.columns 
                         if any(key in col.lower() for key in ['possession', 'highpress', 'prgdist', 'setpieces', 'aerial', 'fieldtilt'])]
        
        if len(style_features) > 2:
            X_style = features_df[style_features].fillna(0)
            X_style_scaled = scalers['style_scaler'].transform(X_style)
            
            # Get styles
            home_style_features = [col for col in style_features if 'home' in col]
            away_style_features = [col for col in style_features if 'away' in col]
            
            print("\n--- TEAM STYLE PROFILES ---")
            
            print("\nHome Team Style Characteristics:")
            for feat in home_style_features[:5]:
                value = features_df[feat].iloc[0]
                print(f"  {feat.replace('home_', '')}: {value:.2f}")
            
            print("\nAway Team Style Characteristics:")
            for feat in away_style_features[:5]:
                value = features_df[feat].iloc[0]
                print(f"  {feat.replace('away_', '')}: {value:.2f}")
            
            # Calculate style matchup
            print("\n--- STYLE MATCHUP ANALYSIS ---")
            mismatches = calculate_tactical_mismatches(home_feats, away_feats)
            
            for key, value in mismatches.items():
                if abs(value) > 0.5:
                    advantage = "HOME advantage" if value > 0 else "AWAY advantage"
                    print(f"{key}: {value:+.3f} ({advantage})")
        
    elif choice == "4":
        print("\n" + "="*60)
        print("TACTICAL MISMATCH RAPPORT")
        print("="*60)
        
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
        
        # Complete tactical analysis
        print("\n--- COMPREHENSIVE TACTICAL ANALYSIS ---\n")
        
        # 1. Tactical Mismatches
        print("1. TACTICAL MISMATCHES")
        print("-" * 30)
        mismatches = calculate_tactical_mismatches(home_feats, away_feats)
        for key, value in sorted(mismatches.items(), key=lambda x: abs(x[1]), reverse=True):
            impact = "HOME+" if value > 0 else "AWAY+"
            print(f"{key:<25}: {value:+7.3f} ({impact})")
        
        # 2. Non-linear Interactions
        print("\n2. NON-LINEAR INTERACTIONS")
        print("-" * 30)
        interactions = create_interaction_features(home_feats, away_feats)
        for key, value in sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"{key:<25}: {value:+7.3f}")
        
        # 3. Psychological Factors
        print("\n3. PSYCHOLOGICAL FACTORS")
        print("-" * 30)
        psych = calculate_psychological_factors(home_df, away_df)
        for key, value in psych.items():
            impact = "HOME+" if value > 0 else "AWAY+"
            print(f"{key:<25}: {value:+7.3f} ({impact})")
        
        # 4. Contextual Weights
        print("\n4. CONTEXTUAL WEIGHT ADJUSTMENTS")
        print("-" * 30)
        context_weights = calculate_contextual_weights(home_feats, away_feats, BASE_WEIGHTS)
        
        # Show only changed weights
        for feat, weight in context_weights.items():
            base = BASE_WEIGHTS.get(feat, 1.0)
            if abs(weight - base) > 0.01:
                change = (weight - base) / base * 100
                print(f"{feat:<20}: {base:.2f} -> {weight:.2f} ({change:+.1f}%)")
        
        # 5. Overall Assessment
        print("\n5. OVERALL TACTICAL ASSESSMENT")
        print("-" * 30)
        
        total_mismatch = sum(mismatches.values())
        total_interaction = sum(interactions.values())
        total_psych = sum(psych.values())
        
        print(f"Total Tactical Score: {total_mismatch:+.3f}")
        print(f"Total Interaction Score: {total_interaction:+.3f}")
        print(f"Total Psychological Score: {total_psych:+.3f}")
        
        overall = total_mismatch + total_interaction * 0.5 + total_psych * 0.3
        print(f"\nOVERALL TACTICAL ADVANTAGE: {overall:+.3f}")
        
        if overall > 1:
            print("Strong HOME tactical advantage")
        elif overall < -1:
            print("Strong AWAY tactical advantage")
        else:
            print("Tactically balanced matchup")
        
    elif choice == "5":
        print("\n" + "="*60)
        print("COMPARISON: WITH vs WITHOUT TACTICAL CORRELATIONS")
        print("="*60)
        
        models, scalers, label_encoder, tactical_weights = load_all_enhanced_models_with_tactical()
        if models is None:
            print("Geen modellen gevonden.")
            exit()
        
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
        
        # Prediction WITHOUT tactical correlations
        print("\n--- WITHOUT TACTICAL CORRELATIONS ---")
        features_basic = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
        features_df = pd.DataFrame([features_basic])
        
        # Remove tactical features
        non_tactical_cols = [col for col in features_df.columns 
                            if not ('tactical_' in col or 'nonlinear_' in col)]
        features_without_tactical = features_df[non_tactical_cols]
        
        # Get base predictions (simplified for comparison)
        if 'calibrated_rf' in models and 'calibrated_scaler' in scalers:
            cal_features = scalers['calibrated_scaler'].transform(features_without_tactical)
            base_probs = models['calibrated_rf'].predict_proba(cal_features)[0]
            
            if len(label_encoder.classes_) == 3:
                prob_dict = dict(zip(label_encoder.classes_, base_probs))
                base_ordered = [prob_dict.get('W', 0), prob_dict.get('D', 0), prob_dict.get('L', 0)]
            else:
                base_ordered = base_probs
            
            print(f"Base Prediction: {['W', 'D', 'L'][np.argmax(base_ordered)]}")
            print(f"Base Probabilities: W={base_ordered[0]:.3f}, D={base_ordered[1]:.3f}, L={base_ordered[2]:.3f}")
        
        # Prediction WITH tactical correlations
        print("\n--- WITH TACTICAL CORRELATIONS ---")
        predictions, correlation_analysis = enhanced_ensemble_prediction_with_tactical(
            models, scalers, label_encoder, features_basic, home_feats, away_feats
        )
        
        if 'ensemble' in predictions:
            tactical_probs = predictions['ensemble']['probabilities']
            print(f"Tactical Prediction: {predictions['ensemble']['prediction']}")
            print(f"Tactical Probabilities: W={tactical_probs[0]:.3f}, D={tactical_probs[1]:.3f}, L={tactical_probs[2]:.3f}")
            
            # Show difference
            print("\n--- IMPACT OF TACTICAL ANALYSIS ---")
            print(f"Win prob change: {(tactical_probs[0] - base_ordered[0]):+.3f}")
            print(f"Draw prob change: {(tactical_probs[1] - base_ordered[1]):+.3f}")
            print(f"Loss prob change: {(tactical_probs[2] - base_ordered[2]):+.3f}")
            
            print(f"\nTactical Correlation Score: {correlation_analysis['correlation_score']:+.3f}")
            print(f"Probability Adjustment: {correlation_analysis['prob_adjustment']:+.3f}")
        
    else:
        print("Ongeldige keuze. Kies 1-5.")

print("\n" + "="*60)
print("Programma beëindigd")
print("="*60)
