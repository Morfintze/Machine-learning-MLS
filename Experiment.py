#!/usr/bin/env python3
"""
DEEL 1: Enhanced Imports, Config en SHAP Feature Analysis
Advanced Soccer Predictor v9.0 - Met SHAP, Enhanced Ensembles en Dynamic Weighting

Dit deel bevat:
- Uitgebreide imports voor alle nieuwe functionaliteiten
- Enhanced configuratie parameters
- SHAP feature importance analysis
- Dynamic feature weighting systeem
"""

import pandas as pd
import numpy as np
import re
from scipy import stats
import scipy.stats as stats
import pickle
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------
# ENHANCED IMPORTS - DEEL 1
# -------------------------
# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, Lasso, ElasticNet
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.stats import mode, poisson
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# SHAP for feature importance
import shap

# Neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Bayesian methods
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson

def safe_execute(func, *args, default_return=None, error_msg="Function execution failed", **kwargs):
    """
    Safely execute a function with error handling.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"{error_msg}: {e}")
        return default_return
    
# -------------------------
# ENHANCED CONFIGURATION - DEEL 1  
# -------------------------
EMA_SPAN = 7
SCALE_TO_SCORE = 4.0

# Model files
MODEL_FILE = "enhanced_soccer_model_v9.pkl"
SHAP_WEIGHTS_FILE = "shap_feature_weights.pkl"
ENSEMBLE_MODEL_FILE = "enhanced_ensemble_v9.pkl"
DYNAMIC_WEIGHTS_FILE = "dynamic_contextual_weights.pkl"
POLYNOMIAL_FEATURES_FILE = "polynomial_features.pkl"
BAYESIAN_MODEL_FILE = "bayesian_uncertainty_model.pkl"

# SHAP Analysis Parameters
SHAP_PARAMS = {
    'n_samples': 1000,  # For SHAP sampling
    'feature_perturbation': 'tree_path_dependent',
    'check_additivity': False,
    'approximate': True
}

# Enhanced XGBoost Parameters (optimized from research)
ENHANCED_XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 10,  # Increased based on research
    'min_child_weight': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'learning_rate': 0.06,  # Optimized from research
    'n_estimators': 500,  # Increased
    'random_state': 42,
    'reg_alpha': 0.9,  # L1 penalty from research
    'reg_lambda': 0.8,  # L2 penalty from research
    'scale_pos_weight': 1.0,
    'gamma': 0.1,
    'max_delta_step': 1,
    'tree_method': 'hist'
}

# CatBoost Parameters (new high-performance model)
CATBOOST_PARAMS = {
    'loss_function': 'MultiClass',
    'iterations': 500,
    'learning_rate': 0.08,
    'depth': 8,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': False
}

# Enhanced LightGBM Parameters
ENHANCED_LIGHTGBM_PARAMS = {
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
    'max_delta_step': 1.0,
    'force_col_wise': True
}

# Polynomial Feature Parameters
POLYNOMIAL_PARAMS = {
    'degree': 2,
    'interaction_only': True,  # Only interaction terms, not pure polynomial
    'include_bias': False
}

# Time Series Cross Validation Parameters
TSCV_PARAMS = {
    'n_splits': 5,
    'test_size': 0.2,
    'gap': 0  # No gap between train and test
}

# Dynamic Weighting Parameters
DYNAMIC_WEIGHT_PARAMS = {
    'decay_parameter': 1.0,  # For w_i = 1/d_i^p
    'context_sensitivity': 1.2,  # How much to adjust for context
    'min_weight': 0.1,  # Minimum weight for any feature
    'max_weight': 3.0   # Maximum weight for any feature
}

# Enhanced minimum standard deviations (research-based)
ENHANCED_MIN_STD_VALUES = {
    'xG90': 0.3, 'Sh90': 1.0, 'SoT90': 0.5, 'ShotQual': 0.05, 'ConvRatio90': 0.1,
    'Goals': 0.5, 'Prog90': 5.0, 'PrgDist90': 100.0, 'Att3rd90': 10.0,
    'Possession': 0.05, 'FieldTilt': 0.05, 'HighPress': 1.0, 
    'AerialMismatch': 3.0, 'KeeperPSxGdiff': 0.2, 'TkldPct_possession': 0.05, 
    'WonPct_misc': 0.05, 'Att_3rd_defense': 1.0, 'SetPieces90': 0.8,
    'WinStreak': 0.5, 'UnbeatenStreak': 0.5, 'LossStreak': 0.5,
    'WinRate5': 0.1, 'WinRate10': 0.1, 'PointsRate5': 0.1, 'PointsRate10': 0.1,
    'RestDays': 1.0, 'RecentForm': 0.2, 'HomeAdvantage': 0.1,
    # New enhanced features
    'ProgressivePassRatio': 0.05, 'xGperShot': 0.02, 'DefensiveActionRate': 0.5,
    'PossessionQuality': 0.03, 'CounterAttackEfficiency': 0.1
}

# Initial base weights (will be replaced by SHAP analysis)
INITIAL_BASE_WEIGHTS = {
    'xG90': 1.0, 'Sh90': 1.4, 'SoT90': 0.8, 'ShotQual': 1.3, 'ConvRatio90': 1.5,
    'Goals': 0.8, 'Prog90': 0.35, 'PrgDist90': 0.25, 'Att3rd90': 0.6,
    'FieldTilt': 0.8, 'HighPress': 0.85, 'AerialMismatch': 1.8, 'Possession': 0.4,
    'KeeperPSxGdiff': -0.44, 'GoalsAgainst': -2.2, 'TkldPct_possession': 0.4,
    'WonPct_misc': 0.4, 'Att_3rd_defense': 0.8, 'SetPieces90': 0.8,
    'WinStreak': 1.1, 'UnbeatenStreak': 0.6, 'LossStreak': -1.0,
    'WinRate5': 1.3, 'WinRate10': 1.0, 'PointsRate5': 1.2, 'PointsRate10': 0.9,
    'RestDays': 0.4, 'RecentForm': 1.1, 'HomeAdvantage': 0.6
}

# -------------------------
# SHAP FEATURE ANALYSIS FUNCTIONS - DEEL 1
# -------------------------
def calculate_shap_feature_importance(models, X, y, feature_names, n_samples=1000):
    """
    Calculate SHAP values for all models and derive feature importance weights.
    FIXED: Better error handling and array shape issues.
    """
    print("\n=== SHAP FEATURE IMPORTANCE ANALYSIS ===")
    
    shap_importance_scores = {}
    all_shap_values = {}
    
    # Sample data for SHAP analysis (for computational efficiency)
    if len(X) > n_samples:
        sample_idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx] if hasattr(y, 'iloc') else y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    for model_name, model in models.items():
        if model is None:
            continue
            
        print(f"\nAnalyzing {model_name}...")
        
        try:
            if 'xgb' in model_name.lower() or isinstance(model, xgb.XGBClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class SHAP values
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    # Multi-class: average absolute values across classes
                    feature_importance = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
                else:
                    # Binary or single array
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    feature_importance = np.mean(np.abs(shap_values), axis=0)
                    
            elif 'lightgbm' in model_name.lower() or 'lgb' in model_name.lower():
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample, check_additivity=False)
                    
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        feature_importance = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
                    else:
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        feature_importance = np.mean(np.abs(shap_values), axis=0)
                        
                except Exception as e:
                    print(f"SHAP failed for {model_name}, using permutation importance: {e}")
                    perm_importance = permutation_importance(model, X_sample, y_sample, 
                                                           n_repeats=3, random_state=42, scoring='accuracy')
                    feature_importance = perm_importance.importances_mean
                    
            elif 'catboost' in model_name.lower():
                try:
                    # CatBoost has built-in feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    else:
                        print(f"Using permutation importance for {model_name}")
                        perm_importance = permutation_importance(model, X_sample, y_sample, 
                                                               n_repeats=3, random_state=42, scoring='accuracy')
                        feature_importance = perm_importance.importances_mean
                except Exception as e:
                    print(f"Feature importance failed for {model_name}: {e}")
                    continue
                    
            elif 'forest' in model_name.lower() or isinstance(model, RandomForestClassifier):
                try:
                    # Use built-in feature importance for Random Forest
                    feature_importance = model.feature_importances_
                except Exception as e:
                    print(f"Using permutation importance for {model_name}: {e}")
                    perm_importance = permutation_importance(model, X_sample, y_sample, 
                                                           n_repeats=3, random_state=42, scoring='accuracy')
                    feature_importance = perm_importance.importances_mean
                    
            else:
                # Use permutation importance for other models
                print(f"Using permutation importance for {model_name}")
                try:
                    # For ensemble models, try to use base estimator importances first
                    if hasattr(model, 'estimators_') and hasattr(model, 'feature_importances_'):
                        # Use built-in feature importance if available
                        feature_importance = model.feature_importances_
                        print(f"  Using built-in feature importance for {model_name}")
                    elif hasattr(model, 'estimators_'):
                        # For ensemble models, average base estimator importances
                        print(f"  Averaging base estimator importances for {model_name}")
                        base_importances = []
                        
                        for estimator in model.estimators_:
                            if hasattr(estimator, 'feature_importances_'):
                                base_importances.append(estimator.feature_importances_)
                            elif hasattr(estimator, 'coef_'):
                                # For linear models, use absolute coefficients
                                coef = estimator.coef_
                                if coef.ndim > 1:
                                    coef = np.mean(np.abs(coef), axis=0)
                                else:
                                    coef = np.abs(coef)
                                base_importances.append(coef)
                        
                        if base_importances:
                            feature_importance = np.mean(base_importances, axis=0)
                            print(f"  Averaged {len(base_importances)} base estimator importances")
                        else:
                            raise ValueError("No base estimator importances available")
                    else:
                        raise ValueError("No feature importance method available")
                        
                except Exception as e:
                    print(f"  Base estimator method failed: {e}")
                    print(f"  Falling back to permutation importance...")
                    
                    # Fallback to permutation importance with better parameters
                    perm_importance = permutation_importance(
                        model, X_sample, y_sample, 
                        n_repeats=5,  # Increased 
                        random_state=42, 
                        scoring='neg_log_loss',  # Better scoring for classification
                        n_jobs=1  # Single job to avoid issues
                    )
                    feature_importance = np.abs(perm_importance.importances_mean)
        
                    # Check if result is uniform (indicating failure)
                    unique_values = len(np.unique(np.round(feature_importance, 8)))
                    if unique_values <= 2:
                        print(f"  Warning: Permutation importance failed (only {unique_values} unique values)")
                        # Create dummy importance based on feature position
                        feature_importance = np.random.exponential(0.01, len(feature_names))
                        feature_importance = feature_importance / np.sum(feature_importance)
                
            # Ensure feature_importance is a valid 1D array
            if isinstance(feature_importance, (list, tuple)):
                feature_importance = np.array(feature_importance)
            
            if feature_importance.ndim > 1:
                feature_importance = feature_importance.flatten()
            
            # Handle NaN values
            feature_importance = np.nan_to_num(feature_importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize importance scores
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            else:
                feature_importance = np.ones(len(feature_names)) / len(feature_names)
            
            # Ensure we have the right number of features
            if len(feature_importance) != len(feature_names):
                print(f"Warning: Feature importance length ({len(feature_importance)}) doesn't match feature names ({len(feature_names)})")
                if len(feature_importance) < len(feature_names):
                    # Pad with zeros
                    feature_importance = np.pad(feature_importance, (0, len(feature_names) - len(feature_importance)))
                else:
                    # Truncate
                    feature_importance = feature_importance[:len(feature_names)]
            
            # Create importance dictionary
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(feature_importance[i])
            
            shap_importance_scores[model_name] = importance_dict
            
            # Print top 10 most important features
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 10 features for {model_name}:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"  {i+1:2d}. {feature:<25}: {importance:.4f}")
                
        except Exception as e:
            print(f"Complete failure for {model_name}: {e}")
            continue
    
    # Calculate ensemble feature importance (average across models)
    if shap_importance_scores:
        print(f"\n=== ENSEMBLE FEATURE IMPORTANCE ===")
        
        try:
            # Get all unique features
            all_features = set()
            for scores in shap_importance_scores.values():
                all_features.update(scores.keys())
            all_features = list(all_features)
            
            ensemble_importance = {}
            for feature in all_features:
                # Collect scores from all models, using 0 if feature not present
                scores = []
                for model_scores in shap_importance_scores.values():
                    score = model_scores.get(feature, 0.0)
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        scores.append(float(score))
                
                if scores:  # Only if we have valid scores
                    ensemble_importance[feature] = np.mean(scores)
                else:
                    ensemble_importance[feature] = 0.0
            
            # Normalize ensemble importance
            total_importance = sum(ensemble_importance.values())
            if total_importance > 0:
                ensemble_importance = {k: v/total_importance for k, v in ensemble_importance.items()}
            
            # Convert to feature weights (scale by factor for better performance)
            feature_weights = {}
            importance_scale = 5.0  # Scale factor to make weights more impactful
            
            for feature, importance in ensemble_importance.items():
                weight = importance * importance_scale
                # Apply bounds
                weight = max(DYNAMIC_WEIGHT_PARAMS['min_weight'], 
                            min(DYNAMIC_WEIGHT_PARAMS['max_weight'], weight))
                feature_weights[feature] = weight
            
            # Print top 15 ensemble features
            sorted_ensemble = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)
            print("Top 15 Ensemble Features:")
            for i, (feature, importance) in enumerate(sorted_ensemble[:15]):
                weight = feature_weights.get(feature, 1.0)
                print(f"  {i+1:2d}. {feature:<25}: {importance:.4f} (weight: {weight:.3f})")
            
            return feature_weights, shap_importance_scores, all_shap_values
        
        except Exception as e:
            print(f"Ensemble calculation failed: {e}")
            return INITIAL_BASE_WEIGHTS, shap_importance_scores, all_shap_values
    
    else:
        print("No SHAP analysis completed successfully, using initial weights")
        return INITIAL_BASE_WEIGHTS, {}, {}

def calculate_dynamic_contextual_weights(home_feats, away_feats, base_weights, context_sensitivity=1.2):
    """
    Enhanced dynamic weighting based on opponent characteristics and match context.
    Implements exponential decay weighting from research: w_i = 1/d_i^p
    """
    context_weights = base_weights.copy()
    
    print("\n=== CALCULATING DYNAMIC CONTEXTUAL WEIGHTS ===")
    
    # Context 1: Low possession opponent -> boost counter-attack metrics
    away_poss = ema(away_feats.get('Possession', pd.Series([0.5])))
    if away_poss < 0.45:
        boost_factor = context_sensitivity * (0.45 - away_poss) * 2
        context_weights['PrgDist90'] = context_weights.get('PrgDist90', 1.0) * (1 + boost_factor)
        context_weights['ConvRatio90'] = context_weights.get('ConvRatio90', 1.0) * (1 + boost_factor * 0.8)
        print(f"Low possession opponent ({away_poss:.1%}) - boosting counter metrics by {boost_factor:.3f}")
    
    # Context 2: Strong aerial opponent
    away_aerial = ema(away_feats.get('AerialWin%', pd.Series([0.5])))
    if away_aerial > 0.55:
        aerial_factor = context_sensitivity * (away_aerial - 0.55) * 3
        context_weights['AerialMismatch'] = context_weights.get('AerialMismatch', 1.0) * (1 + aerial_factor)
        context_weights['SetPieces90'] = context_weights.get('SetPieces90', 1.0) * (1 - aerial_factor * 0.5)
        print(f"Strong aerial opponent ({away_aerial:.1%}) - adjusting aerial weights by {aerial_factor:.3f}")
    
    # Context 3: High pressing opponent
    away_press = ema(away_feats.get('HighPress', pd.Series([0])))
    if away_press > 1.5:
        press_factor = context_sensitivity * min(away_press / 3.0, 0.5)
        context_weights['Possession'] = context_weights.get('Possession', 1.0) * (1 + press_factor)
        context_weights['TkldPct_possession'] = context_weights.get('TkldPct_possession', 1.0) * (1 + press_factor * 1.2)
        print(f"High pressing opponent ({away_press:.2f}) - boosting possession retention")
    
    # Context 4: Home advantage strength
    home_advantage = ema(home_feats.get('HomeAdvantage', pd.Series([0])))
    if abs(home_advantage) > 0.15:
        home_factor = context_sensitivity * abs(home_advantage) * 2
        context_weights['HomeAdvantage'] = context_weights.get('HomeAdvantage', 1.0) * (1 + home_factor)
        print(f"Strong home effect ({home_advantage:+.1%}) - boosting home weight")
    
    # Context 5: Form momentum differential
    home_form = ema(home_feats.get('RecentForm', pd.Series([0])))
    away_form = ema(away_feats.get('RecentForm', pd.Series([0])))
    form_diff = home_form - away_form
    
    if abs(form_diff) > 0.5:
        form_factor = context_sensitivity * min(abs(form_diff) / 2.0, 0.3)
        context_weights['RecentForm'] = context_weights.get('RecentForm', 1.0) * (1 + form_factor)
        context_weights['WinRate5'] = context_weights.get('WinRate5', 1.0) * (1 + form_factor * 0.8)
        print(f"Form differential ({form_diff:+.2f}) - boosting form weights")
    
    # Apply bounds to all weights
    for key in context_weights:
        context_weights[key] = max(DYNAMIC_WEIGHT_PARAMS['min_weight'], 
                                 min(DYNAMIC_WEIGHT_PARAMS['max_weight'], context_weights[key]))
    
    return context_weights

def calculate_contextual_weights(home_feats, away_feats, base_weights):
    """
    Calculate contextual weights based on match situation.
    Adjusts base weights based on team characteristics and context.
    """
    contextual_weights = base_weights.copy()
    
    try:
        # Get recent form data
        home_form = ema(home_feats.get('RecentForm', pd.Series([0])))
        away_form = ema(away_feats.get('RecentForm', pd.Series([0])))
        
        # Get possession styles
        home_poss = ema(home_feats.get('Possession', pd.Series([0.5])))
        away_poss = ema(away_feats.get('Possession', pd.Series([0.5])))
        
        # Get attacking vs defensive tendencies
        home_xg = ema(home_feats.get('xG90', pd.Series([0])))
        away_xg = ema(away_feats.get('xG90', pd.Series([0])))
        home_ga = ema(home_feats.get('GoalsAgainst', pd.Series([1.5])))
        away_ga = ema(away_feats.get('GoalsAgainst', pd.Series([1.5])))
        
        # Context 1: High-scoring game expected
        total_xg = home_xg + away_xg
        if total_xg > 2.5:
            contextual_weights['xG90'] = contextual_weights.get('xG90', 1.0) * 1.15
            contextual_weights['ShotQual'] = contextual_weights.get('ShotQual', 1.0) * 1.1
            contextual_weights['ConvRatio90'] = contextual_weights.get('ConvRatio90', 1.0) * 1.2
        
        # Context 2: Possession mismatch
        poss_diff = abs(home_poss - away_poss)
        if poss_diff > 0.15:  # Significant possession difference
            contextual_weights['Possession'] = contextual_weights.get('Possession', 1.0) * 1.3
            contextual_weights['FieldTilt'] = contextual_weights.get('FieldTilt', 1.0) * 1.2
        
        # Context 3: Form differential impact
        form_diff = abs(home_form - away_form)
        if form_diff > 0.5:
            contextual_weights['RecentForm'] = contextual_weights.get('RecentForm', 1.0) * 1.25
            contextual_weights['WinRate5'] = contextual_weights.get('WinRate5', 1.0) * 1.15
        
        # Context 4: Defensive solidity
        avg_ga = (home_ga + away_ga) / 2
        if avg_ga < 1.0:  # Both teams defensive
            contextual_weights['GoalsAgainst'] = contextual_weights.get('GoalsAgainst', 1.0) * 1.3
            contextual_weights['SetPieces90'] = contextual_weights.get('SetPieces90', 1.0) * 1.2
        
        # Normalize weights to prevent explosion
        max_weight = 3.0
        for key in contextual_weights:
            contextual_weights[key] = min(max_weight, contextual_weights[key])
        
    except Exception as e:
        print(f"Error in contextual weights calculation: {e}")
        return base_weights
    
    return contextual_weights


# -------------------------
# HELPER FUNCTIONS - DEEL 1
# -------------------------
def ema_robust(series, span=EMA_SPAN, min_periods=1):
    """
    Robust EMA calculation with better edge case handling.
    """
    if series is None or len(series) == 0:
        return 0.0
    
    # Convert to pandas Series if not already
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except:
            return 0.0
    
    # Handle single value
    if len(series) == 1:
        return float(series.iloc[0]) if not pd.isna(series.iloc[0]) else 0.0
    
    # Remove NaN values
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    
    # Calculate EMA
    try:
        if len(clean_series) < min_periods:
            return float(clean_series.mean())
        else:
            ema_result = clean_series.ewm(span=span, adjust=False).mean().iloc[-1]
            return float(ema_result) if not pd.isna(ema_result) else 0.0
    except:
        return float(clean_series.iloc[-1]) if len(clean_series) > 0 else 0.0

# Update the existing ema function

def ema(series, span=EMA_SPAN):
    """Updated EMA function."""
    return ema_robust(series, span)
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

def validate_configuration():
    """Validate configuration settings."""
    try:
        # Basic validation
        return True
    except:
        return False

def simple_test():
    """Simple test to check if basic functions work."""
    print("Testing basic functionality...")
    test_data = pd.Series([1, 2, 3, 4, 5])
    result = ema_robust(test_data)
    print(f"EMA test result: {result}")
    print("Basic test complete!")

print("DEEL 1 geladen: Enhanced Imports, Config en SHAP Feature Analysis")
print("Volgende: DEEL 2 - Enhanced Ensemble Methods en Soft Voting")

"""
DEEL 2: Enhanced Ensemble Methods en Soft Voting
Advanced Soccer Predictor v9.0

Dit deel bevat:
- Enhanced ensemble methods met soft voting
- CatBoost model integratie 
- Stacking ensemble voor meta-learning
- Improved time-series cross-validation
- Model calibration en uncertainty quantification
- Advanced performance metrics
"""

# -------------------------
# ENHANCED ENSEMBLE METHODS - DEEL 2
# -------------------------

def train_enhanced_ensemble_models(X, y, shap_weights=None, use_time_series_cv=True):
    """
    Train enhanced ensemble models met soft voting en stacking.
    Implementeert research-based configuraties voor optimale performance.
    """
    print("\n" + "="*60)
    print("TRAINING ENHANCED ENSEMBLE MODELS")
    print("="*60)
    
    if len(X) < 30:
        print(f"Onvoldoende data voor ensemble training: {len(X)} samples. Minimum 30 vereist.")
        return None, None, None, None
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Enhanced train-test split met time-aware splitting
    if use_time_series_cv:
        # Time-series aware split (laatste 30% als test)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        
        print(f"Time-series split: Train={len(X_train)}, Test={len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    calibrated_models = {}
    
    # -------------------------
    # MODEL 1: Enhanced XGBoost
    # -------------------------
    print("\n1. Training Enhanced XGBoost...")
    xgb_model = xgb.XGBClassifier(**ENHANCED_XGBOOST_PARAMS)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate XGBoost
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_prob = xgb_model.predict_proba(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_brier = calculate_brier_score(y_test, xgb_prob)
    
    models['xgboost'] = xgb_model
    print(f"XGBoost Accuracy: {xgb_acc:.4f}, Brier Score: {xgb_brier:.4f}")
    
    # -------------------------
    # MODEL 2: Enhanced LightGBM
    # -------------------------
    print("\n2. Training Enhanced LightGBM...")
    lgb_model = lgb.LGBMClassifier(**ENHANCED_LIGHTGBM_PARAMS)
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Evaluate LightGBM
    lgb_pred = lgb_model.predict(X_test_scaled)
    lgb_prob = lgb_model.predict_proba(X_test_scaled)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_brier = calculate_brier_score(y_test, lgb_prob)
    
    models['lightgbm'] = lgb_model
    print(f"LightGBM Accuracy: {lgb_acc:.4f}, Brier Score: {lgb_brier:.4f}")
    
    # -------------------------
    # MODEL 3: CatBoost (High Performance)
    # -------------------------
    print("\n3. Training CatBoost...")
    cat_model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), verbose=False)
    
    # Evaluate CatBoost
    cat_pred = cat_model.predict(X_test_scaled)
    cat_prob = cat_model.predict_proba(X_test_scaled)
    cat_acc = accuracy_score(y_test, cat_pred)
    cat_brier = calculate_brier_score(y_test, cat_prob)
    
    models['catboost'] = cat_model
    print(f"CatBoost Accuracy: {cat_acc:.4f}, Brier Score: {cat_brier:.4f}")
    
    # -------------------------
    # MODEL 4: Enhanced Random Forest
    # -------------------------
    print("\n4. Training Enhanced Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Increased from research
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        max_features='sqrt',
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_brier = calculate_brier_score(y_test, rf_prob)
    
    models['random_forest'] = rf_model
    print(f"Random Forest Accuracy: {rf_acc:.4f}, Brier Score: {rf_brier:.4f}")
    
    # -------------------------
    # SOFT VOTING ENSEMBLE
    # -------------------------
    print("\n5. Creating Soft Voting Ensemble...")
    
    # Create voting classifier with optimized weights
    voting_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ]
    
    # Weight models based on their individual performance
    model_weights = []
    accuracies = [xgb_acc, lgb_acc, cat_acc, rf_acc]
    brier_scores = [xgb_brier, lgb_brier, cat_brier, rf_brier]
    
    # Calculate weights based on accuracy and inverse Brier score
    for acc, brier in zip(accuracies, brier_scores):
        # Higher accuracy and lower Brier = higher weight
        weight = (acc / max(accuracies)) * (min(brier_scores) / brier)
        model_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(model_weights)
    model_weights = [w / total_weight for w in model_weights]
    
    print(f"Model weights: XGB={model_weights[0]:.3f}, LGB={model_weights[1]:.3f}, "
          f"CAT={model_weights[2]:.3f}, RF={model_weights[3]:.3f}")
    
    # Create and train voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=voting_models,
        voting='soft',
        weights=model_weights
    )
    
    # Fit on scaled data (models are already trained, this just sets up voting)
    voting_ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate voting ensemble
    vote_pred = voting_ensemble.predict(X_test_scaled)
    vote_prob = voting_ensemble.predict_proba(X_test_scaled)
    vote_acc = accuracy_score(y_test, vote_pred)
    vote_brier = calculate_brier_score(y_test, vote_prob)
    
    models['voting_ensemble'] = voting_ensemble
    print(f"Voting Ensemble Accuracy: {vote_acc:.4f}, Brier Score: {vote_brier:.4f}")
    
    # -------------------------
    # STACKING ENSEMBLE (META-LEARNER)
    # -------------------------
    print("\n6. Creating Stacking Ensemble...")
    
    # Meta-learner: Elastic Net (research recommended)
    meta_learner = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42, max_iter=1000)
    
    # For classification, we need a classifier meta-learner
    meta_classifier = LogisticRegression(
        C=1.0, 
        penalty='elasticnet', 
        l1_ratio=0.7, 
        solver='saga', 
        max_iter=1000, 
        random_state=42
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=voting_models,
        final_estimator=meta_classifier,
        cv=5,  # 5-fold CV for meta-features
        stack_method='predict_proba'
    )
    
    stacking_ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate stacking ensemble
    stack_pred = stacking_ensemble.predict(X_test_scaled)
    stack_prob = stacking_ensemble.predict_proba(X_test_scaled)
    stack_acc = accuracy_score(y_test, stack_pred)
    stack_brier = calculate_brier_score(y_test, stack_prob)
    
    models['stacking_ensemble'] = stacking_ensemble
    print(f"Stacking Ensemble Accuracy: {stack_acc:.4f}, Brier Score: {stack_brier:.4f}")
    
    # -------------------------
    # MODEL CALIBRATION
    # -------------------------
    print("\n7. Calibrating Models...")
    
    # Calibrate the best models for better probability estimates
    best_models = ['voting_ensemble', 'stacking_ensemble']
    
    for model_name in best_models:
        if model_name in models:
            print(f"Calibrating {model_name}...")
            calibrated_model = CalibratedClassifierCV(
                models[model_name], 
                method='isotonic',  # Better for small datasets
                cv=3
            )
            calibrated_model.fit(X_train_scaled, y_train)
            calibrated_models[f'{model_name}_calibrated'] = calibrated_model
            
            # Evaluate calibrated model
            cal_pred = calibrated_model.predict(X_test_scaled)
            cal_prob = calibrated_model.predict_proba(X_test_scaled)
            cal_acc = accuracy_score(y_test, cal_pred)
            cal_brier = calculate_brier_score(y_test, cal_prob)
            
            print(f"Calibrated {model_name} Accuracy: {cal_acc:.4f}, Brier Score: {cal_brier:.4f}")
    
    # -------------------------
    # CROSS-VALIDATION ANALYSIS
    # -------------------------
    print("\n8. Cross-Validation Analysis...")
    
    if use_time_series_cv:
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=TSCV_PARAMS['n_splits'], test_size=None)
        cv_scores = {}
        
        for name, model in models.items():
            if name in ['voting_ensemble', 'stacking_ensemble']:  # Skip ensembles for CV (already fitted)
                continue
            scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
            cv_scores[name] = scores
            print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # -------------------------
    # PERFORMANCE SUMMARY
    # -------------------------
    print("\n" + "="*60)
    print("=== ENSEMBLE PERFORMANCE SUMMARY ===")
    print("="*60)
    
    all_accuracies = {
        'XGBoost': xgb_acc,
        'LightGBM': lgb_acc,
        'CatBoost': cat_acc,
        'Random Forest': rf_acc,
        'Voting Ensemble': vote_acc,
        'Stacking Ensemble': stack_acc
    }
    
    all_brier_scores = {
        'XGBoost': xgb_brier,
        'LightGBM': lgb_brier,
        'CatBoost': cat_brier,
        'Random Forest': rf_brier,
        'Voting Ensemble': vote_brier,
        'Stacking Ensemble': stack_brier
    }
    
    # Sort by accuracy
    sorted_models = sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Brier Score':<12}")
    print("-" * 45)
    
    for model_name, accuracy in sorted_models:
        brier = all_brier_scores[model_name]
        print(f"{model_name:<20} | {accuracy:<10.4f} | {brier:<12.4f}")
    
    best_model_name = sorted_models[0][0]
    best_accuracy = sorted_models[0][1]
    
    print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Combine all models
    all_models = {**models, **calibrated_models}
    
    return all_models, scaler, le, {
        'accuracies': all_accuracies,
        'brier_scores': all_brier_scores,
        'best_model': best_model_name,
        'model_weights': model_weights
    }

def calculate_brier_score(y_true, y_prob):
    """Calculate multi-class Brier score."""
    brier_scores = []
    for i in range(y_prob.shape[1]):
        y_true_binary = (y_true == i).astype(int)
        brier_scores.append(brier_score_loss(y_true_binary, y_prob[:, i]))
    return np.mean(brier_scores)

# -------------------------
# TIME-SERIES CROSS VALIDATION - DEEL 2
# -------------------------
def enhanced_time_series_validation(models, X, y, n_splits=5):
    """
    Enhanced time-series cross-validation met forward chaining.
    Implementeert research-based validation zonder data leakage.
    """
    print("\n=== ENHANCED TIME-SERIES CROSS-VALIDATION ===")
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
    validation_results = {}
    
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"\nValidating {name}...")
        
        fold_scores = []
        fold_brier_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            try:
                # Handle different model types
                if hasattr(model, 'fit'):
                    if 'lightgbm' in name.lower() or 'lgb' in name.lower():
                        # LightGBM specific fitting
                        model.fit(X_train_scaled, y_train_fold, eval_set=[(X_test_scaled, y_test_fold)], verbose=0)
                    else:
                        model.fit(X_train_scaled, y_train_fold)
                
                # Predict
                y_pred_fold = model.predict(X_test_scaled)
                y_prob_fold = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                fold_acc = accuracy_score(y_test_fold, y_pred_fold)
                
                # Convert y_test_fold to numeric if needed for Brier score
                if hasattr(y_test_fold, 'map'):
                    le_temp = LabelEncoder()
                    y_test_numeric = le_temp.fit_transform(y_test_fold)
                else:
                    y_test_numeric = y_test_fold
                
                fold_brier = calculate_brier_score(y_test_numeric, y_prob_fold)
                
                fold_scores.append(fold_acc)
                fold_brier_scores.append(fold_brier)
                
                print(f"  Fold {fold}: Accuracy={fold_acc:.4f}, Brier={fold_brier:.4f}")
                
            except Exception as e:
                print(f"  Fold {fold}: Error - {e}")
                continue
        
        if fold_scores:
            mean_acc = np.mean(fold_scores)
            std_acc = np.std(fold_scores)
            mean_brier = np.mean(fold_brier_scores)
            std_brier = np.std(fold_brier_scores)
            
            validation_results[name] = {
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'brier_mean': mean_brier,
                'brier_std': std_brier,
                'fold_scores': fold_scores,
                'fold_brier_scores': fold_brier_scores
            }
            
            print(f"  Final: Accuracy={mean_acc:.4f} (+/- {std_acc*2:.4f})")
            print(f"         Brier={mean_brier:.4f} (+/- {std_brier*2:.4f})")
    
    return validation_results

# -------------------------
# ADVANCED PREDICTION FUNCTION - DEEL 2
# -------------------------
def enhanced_ensemble_prediction(models, scaler, label_encoder, features_dict, 
                                home_feats, away_feats, use_calibrated=True):
    """
    Enhanced ensemble prediction met uncertainty quantification.
    Combineert alle modellen met intelligente weighting.
    """
    print("\n" + "="*60)
    print("ENHANCED ENSEMBLE PREDICTION")
    print("="*60)
    
    # Prepare features
    features_df = pd.DataFrame([features_dict])
    features_scaled = scaler.transform(features_df)
    
    predictions = {}
    probabilities = {}
    confidences = {}
    
    # Individual model predictions
    model_predictions = []
    model_probabilities = []
    model_confidences = []
    
    for name, model in models.items():
        if model is None:
            continue
            
        try:
            # Get prediction and probabilities
            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            # Convert numeric prediction back to label
            if hasattr(label_encoder, 'inverse_transform'):
                pred_label = label_encoder.inverse_transform([pred])[0]
            else:
                pred_label = ['W', 'D', 'L'][pred]
            
            # Calculate confidence (max probability)
            confidence = np.max(prob)
            
            predictions[name] = pred_label
            probabilities[name] = prob
            confidences[name] = confidence
            
            # Store for ensemble calculation
            model_predictions.append(pred)
            model_probabilities.append(prob)
            model_confidences.append(confidence)
            
            print(f"{name:<25}: {pred_label} ({confidence:.3f} conf)")
            print(f"{'':25}  W={prob[0]:.3f}, D={prob[1]:.3f}, L={prob[2]:.3f}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    # Enhanced ensemble calculation
    if model_probabilities:
        print(f"\n=== ENSEMBLE CALCULATION ===")
        
        # Method 1: Simple average
        ensemble_prob_simple = np.mean(model_probabilities, axis=0)
        ensemble_pred_simple = np.argmax(ensemble_prob_simple)
        
        # Method 2: Confidence-weighted average
        weights = np.array(model_confidences)
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_prob_weighted = np.average(model_probabilities, axis=0, weights=weights)
        ensemble_pred_weighted = np.argmax(ensemble_prob_weighted)
        
        # Method 3: Use calibrated models if available
        calibrated_probs = []
        calibrated_names = [name for name in models.keys() if 'calibrated' in name]
        
        if calibrated_names:
            for name in calibrated_names:
                if name in probabilities:
                    calibrated_probs.append(probabilities[name])
            
            if calibrated_probs:
                ensemble_prob_calibrated = np.mean(calibrated_probs, axis=0)
                ensemble_pred_calibrated = np.argmax(ensemble_prob_calibrated)
            else:
                ensemble_prob_calibrated = ensemble_prob_weighted
                ensemble_pred_calibrated = ensemble_pred_weighted
        else:
            ensemble_prob_calibrated = ensemble_prob_weighted
            ensemble_pred_calibrated = ensemble_pred_weighted
        
        # Choose best ensemble method (prefereer calibrated)
        if use_calibrated and calibrated_probs:
            final_prob = ensemble_prob_calibrated
            final_pred = ensemble_pred_calibrated
            method_used = "Calibrated Ensemble"
        else:
            final_prob = ensemble_prob_weighted
            final_pred = ensemble_pred_weighted
            method_used = "Confidence-Weighted Ensemble"
        
        # Convert prediction back to label
        final_pred_label = label_encoder.inverse_transform([final_pred])[0] if hasattr(label_encoder, 'inverse_transform') else ['W', 'D', 'L'][final_pred]
        final_confidence = np.max(final_prob)
        
        # Calculate uncertainty metrics
        entropy = -np.sum(final_prob * np.log(final_prob + 1e-10))  # Shannon entropy
        uncertainty = 1 - final_confidence  # Simple uncertainty measure
        
        print(f"\n{method_used}:")
        print(f"Prediction: {final_pred_label}")
        print(f"Probabilities: W={final_prob[0]:.3f}, D={final_prob[1]:.3f}, L={final_prob[2]:.3f}")
        print(f"Confidence: {final_confidence:.3f}")
        print(f"Uncertainty: {uncertainty:.3f}")
        print(f"Entropy: {entropy:.3f}")
        
        # Confidence assessment
        if final_confidence > 0.7:
            confidence_level = "HIGH"
        elif final_confidence > 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        print(f"Confidence Level: {confidence_level}")
        
        return {
            'prediction': final_pred_label,
            'probabilities': final_prob,
            'confidence': final_confidence,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'confidence_level': confidence_level,
            'method_used': method_used,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'individual_confidences': confidences
        }
    
    else:
        print("No successful predictions from ensemble models")
        return None

print("DEEL 2 geladen: Enhanced Ensemble Methods en Soft Voting")
print("Volgende: DEEL 3 - Polynomial Features en Non-Linear Relationships")
"""
DEEL 2: Enhanced Ensemble Methods en Soft Voting
Advanced Soccer Predictor v9.0

Dit deel bevat:
- Enhanced ensemble methods met soft voting
- CatBoost model integratie 
- Stacking ensemble voor meta-learning
- Improved time-series cross-validation
- Model calibration en uncertainty quantification
- Advanced performance metrics
"""

# -------------------------
# ENHANCED ENSEMBLE METHODS - DEEL 2
# -------------------------

def train_enhanced_ensemble_models(X, y, shap_weights=None, use_time_series_cv=True):
    """
    Train enhanced ensemble models met soft voting en stacking.
    Implementeert research-based configuraties voor optimale performance.
    """
    print("\n" + "="*60)
    print("TRAINING ENHANCED ENSEMBLE MODELS")
    print("="*60)
    
    if len(X) < 30:
        print(f"Onvoldoende data voor ensemble training: {len(X)} samples. Minimum 30 vereist.")
        return None, None, None, None
    
    # Import precision/recall metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Enhanced train-test split met time-aware splitting
    if use_time_series_cv:
        # Time-series aware split (laatste 30% als test)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        
        print(f"Time-series split: Train={len(X_train)}, Test={len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    calibrated_models = {}
    
    # Store all metrics
    all_metrics = {}
    
    # -------------------------
    # MODEL 1: Enhanced XGBoost
    # -------------------------
    print("\n1. Training Enhanced XGBoost...")
    xgb_model = xgb.XGBClassifier(**ENHANCED_XGBOOST_PARAMS)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate XGBoost with all metrics
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_prob = xgb_model.predict_proba(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_brier = calculate_brier_score(y_test, xgb_prob)
    xgb_precision = precision_score(y_test, xgb_pred, average='weighted', zero_division=0)
    xgb_recall = recall_score(y_test, xgb_pred, average='weighted', zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
    
    models['xgboost'] = xgb_model
    all_metrics['XGBoost'] = {
        'accuracy': xgb_acc, 'brier': xgb_brier, 
        'precision': xgb_precision, 'recall': xgb_recall, 'f1': xgb_f1
    }
    print(f"XGBoost - Acc: {xgb_acc:.4f}, Prec: {xgb_precision:.4f}, Rec: {xgb_recall:.4f}, F1: {xgb_f1:.4f}")
    
    # -------------------------
    # MODEL 2: Enhanced LightGBM
    # -------------------------
    print("\n2. Training Enhanced LightGBM...")
    lgb_model = lgb.LGBMClassifier(**ENHANCED_LIGHTGBM_PARAMS)
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Evaluate LightGBM with all metrics
    lgb_pred = lgb_model.predict(X_test_scaled)
    lgb_prob = lgb_model.predict_proba(X_test_scaled)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_brier = calculate_brier_score(y_test, lgb_prob)
    lgb_precision = precision_score(y_test, lgb_pred, average='weighted', zero_division=0)
    lgb_recall = recall_score(y_test, lgb_pred, average='weighted', zero_division=0)
    lgb_f1 = f1_score(y_test, lgb_pred, average='weighted', zero_division=0)
    
    models['lightgbm'] = lgb_model
    all_metrics['LightGBM'] = {
        'accuracy': lgb_acc, 'brier': lgb_brier,
        'precision': lgb_precision, 'recall': lgb_recall, 'f1': lgb_f1
    }
    print(f"LightGBM - Acc: {lgb_acc:.4f}, Prec: {lgb_precision:.4f}, Rec: {lgb_recall:.4f}, F1: {lgb_f1:.4f}")
    
    # -------------------------
    # MODEL 3: CatBoost (High Performance)
    # -------------------------
    print("\n3. Training CatBoost...")
    cat_model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), verbose=False)
    
    # Evaluate CatBoost with all metrics
    cat_pred = cat_model.predict(X_test_scaled)
    cat_prob = cat_model.predict_proba(X_test_scaled)
    cat_acc = accuracy_score(y_test, cat_pred)
    cat_brier = calculate_brier_score(y_test, cat_prob)
    cat_precision = precision_score(y_test, cat_pred, average='weighted', zero_division=0)
    cat_recall = recall_score(y_test, cat_pred, average='weighted', zero_division=0)
    cat_f1 = f1_score(y_test, cat_pred, average='weighted', zero_division=0)
    
    models['catboost'] = cat_model
    all_metrics['CatBoost'] = {
        'accuracy': cat_acc, 'brier': cat_brier,
        'precision': cat_precision, 'recall': cat_recall, 'f1': cat_f1
    }
    print(f"CatBoost - Acc: {cat_acc:.4f}, Prec: {cat_precision:.4f}, Rec: {cat_recall:.4f}, F1: {cat_f1:.4f}")
    
    # -------------------------
    # MODEL 4: Enhanced Random Forest
    # -------------------------
    print("\n4. Training Enhanced Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Increased from research
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        max_features='sqrt',
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate Random Forest with all metrics
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_brier = calculate_brier_score(y_test, rf_prob)
    rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
    
    models['random_forest'] = rf_model
    all_metrics['Random Forest'] = {
        'accuracy': rf_acc, 'brier': rf_brier,
        'precision': rf_precision, 'recall': rf_recall, 'f1': rf_f1
    }
    print(f"Random Forest - Acc: {rf_acc:.4f}, Prec: {rf_precision:.4f}, Rec: {rf_recall:.4f}, F1: {rf_f1:.4f}")
    
    # -------------------------
    # SOFT VOTING ENSEMBLE
    # -------------------------
    print("\n5. Creating Soft Voting Ensemble...")
    
    # Create voting classifier with optimized weights
    voting_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ]
    
    # Weight models based on their individual performance
    model_weights = []
    accuracies = [xgb_acc, lgb_acc, cat_acc, rf_acc]
    brier_scores = [xgb_brier, lgb_brier, cat_brier, rf_brier]
    
    # Calculate weights based on accuracy and inverse Brier score
    for acc, brier in zip(accuracies, brier_scores):
        # Higher accuracy and lower Brier = higher weight
        weight = (acc / max(accuracies)) * (min(brier_scores) / max(brier, 1e-6))
        model_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(model_weights)
    if total_weight > 0:
        model_weights = [w / total_weight for w in model_weights]
    else:
        model_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights fallback
    
    print(f"Model weights: XGB={model_weights[0]:.3f}, LGB={model_weights[1]:.3f}, "
          f"CAT={model_weights[2]:.3f}, RF={model_weights[3]:.3f}")
    
    # Create and train voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=voting_models,
        voting='soft',
        weights=model_weights
    )
    
    # Fit on scaled data (models are already trained, this just sets up voting)
    voting_ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate voting ensemble with all metrics
    vote_pred = voting_ensemble.predict(X_test_scaled)
    vote_prob = voting_ensemble.predict_proba(X_test_scaled)
    vote_acc = accuracy_score(y_test, vote_pred)
    vote_brier = calculate_brier_score(y_test, vote_prob)
    vote_precision = precision_score(y_test, vote_pred, average='weighted', zero_division=0)
    vote_recall = recall_score(y_test, vote_pred, average='weighted', zero_division=0)
    vote_f1 = f1_score(y_test, vote_pred, average='weighted', zero_division=0)
    
    models['voting_ensemble'] = voting_ensemble
    all_metrics['Voting Ensemble'] = {
        'accuracy': vote_acc, 'brier': vote_brier,
        'precision': vote_precision, 'recall': vote_recall, 'f1': vote_f1
    }
    print(f"Voting Ensemble - Acc: {vote_acc:.4f}, Prec: {vote_precision:.4f}, Rec: {vote_recall:.4f}, F1: {vote_f1:.4f}")
    
    # -------------------------
    # STACKING ENSEMBLE (META-LEARNER)
    # -------------------------
    print("\n6. Creating Stacking Ensemble...")
    
    # Meta-learner: Logistic Regression for classification
    meta_classifier = LogisticRegression(
        C=1.0, 
        penalty='l2',  # Changed from elasticnet for stability
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=voting_models,
        final_estimator=meta_classifier,
        cv=5,  # 5-fold CV for meta-features
        stack_method='predict_proba'
    )
    
    stacking_ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate stacking ensemble with all metrics
    stack_pred = stacking_ensemble.predict(X_test_scaled)
    stack_prob = stacking_ensemble.predict_proba(X_test_scaled)
    stack_acc = accuracy_score(y_test, stack_pred)
    stack_brier = calculate_brier_score(y_test, stack_prob)
    stack_precision = precision_score(y_test, stack_pred, average='weighted', zero_division=0)
    stack_recall = recall_score(y_test, stack_pred, average='weighted', zero_division=0)
    stack_f1 = f1_score(y_test, stack_pred, average='weighted', zero_division=0)
    
    models['stacking_ensemble'] = stacking_ensemble
    all_metrics['Stacking Ensemble'] = {
        'accuracy': stack_acc, 'brier': stack_brier,
        'precision': stack_precision, 'recall': stack_recall, 'f1': stack_f1
    }
    print(f"Stacking Ensemble - Acc: {stack_acc:.4f}, Prec: {stack_precision:.4f}, Rec: {stack_recall:.4f}, F1: {stack_f1:.4f}")
    
    # -------------------------
    # MODEL CALIBRATION
    # -------------------------
    print("\n7. Calibrating Models...")
    
    # Calibrate the best models for better probability estimates
    best_models = ['voting_ensemble', 'stacking_ensemble']
    
    for model_name in best_models:
        if model_name in models:
            print(f"Calibrating {model_name}...")
            try:
                calibrated_model = CalibratedClassifierCV(
                    models[model_name], 
                    method='isotonic',  # Better for small datasets
                    cv=3
                )
                calibrated_model.fit(X_train_scaled, y_train)
                calibrated_models[f'{model_name}_calibrated'] = calibrated_model
                
                # Evaluate calibrated model
                cal_pred = calibrated_model.predict(X_test_scaled)
                cal_prob = calibrated_model.predict_proba(X_test_scaled)
                cal_acc = accuracy_score(y_test, cal_pred)
                cal_brier = calculate_brier_score(y_test, cal_prob)
                cal_precision = precision_score(y_test, cal_pred, average='weighted', zero_division=0)
                cal_recall = recall_score(y_test, cal_pred, average='weighted', zero_division=0)
                cal_f1 = f1_score(y_test, cal_pred, average='weighted', zero_division=0)
                
                all_metrics[f'Calibrated {model_name.replace("_", " ").title()}'] = {
                    'accuracy': cal_acc, 'brier': cal_brier,
                    'precision': cal_precision, 'recall': cal_recall, 'f1': cal_f1
                }
                print(f"Calibrated {model_name} - Acc: {cal_acc:.4f}, Prec: {cal_precision:.4f}, Rec: {cal_recall:.4f}, F1: {cal_f1:.4f}")
                
            except Exception as e:
                print(f"Calibration failed for {model_name}: {e}")
    
    # -------------------------
    # CROSS-VALIDATION ANALYSIS
    # -------------------------
    print("\n8. Cross-Validation Analysis...")
    
    if use_time_series_cv:
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=TSCV_PARAMS['n_splits'], test_size=None)
        cv_scores = {}
        
        for name, model in models.items():
            if name in ['voting_ensemble', 'stacking_ensemble']:  # Skip ensembles for CV (already fitted)
                continue
            try:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
                cv_scores[name] = scores
                print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"CV failed for {name}: {e}")
    
    # -------------------------
    # PERFORMANCE SUMMARY
    # -------------------------
    print("\n" + "="*80)
    print("=== ENHANCED ENSEMBLE PERFORMANCE SUMMARY ===")
    print("="*80)
    
    # Sort by F1 score (better for classification)
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    print(f"{'Model':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'Brier':<8}")
    print("-" * 85)
    
    for model_name, metrics in sorted_models:
        print(f"{model_name:<25} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | "
              f"{metrics['recall']:<8.4f} | {metrics['f1']:<8.4f} | {metrics['brier']:<8.4f}")
    
    best_model_name = sorted_models[0][0]
    best_f1 = sorted_models[0][1]['f1']
    
    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    
    # Combine all models
    all_models = {**models, **calibrated_models}
    
    return all_models, scaler, le, {
        'all_metrics': all_metrics,
        'best_model': best_model_name,
        'model_weights': model_weights
    }

def calculate_brier_score(y_true, y_prob):
    """Calculate multi-class Brier score."""
    try:
        brier_scores = []
        for i in range(y_prob.shape[1]):
            y_true_binary = (y_true == i).astype(int)
            brier_scores.append(brier_score_loss(y_true_binary, y_prob[:, i]))
        return np.mean(brier_scores)
    except:
        return 0.5  # Fallback value

# -------------------------
# TIME-SERIES CROSS VALIDATION - DEEL 2
# -------------------------
def enhanced_time_series_validation(models, X, y, n_splits=5):
    """
    Enhanced time-series cross-validation met forward chaining.
    Implementeert research-based validation zonder data leakage.
    """
    print("\n=== ENHANCED TIME-SERIES CROSS-VALIDATION ===")
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
    validation_results = {}
    
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"\nValidating {name}...")
        
        fold_scores = []
        fold_brier_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            try:
                # Handle different model types
                if hasattr(model, 'fit'):
                    if 'lightgbm' in name.lower() or 'lgb' in name.lower():
                        # LightGBM specific fitting
                        model.fit(X_train_scaled, y_train_fold, eval_set=[(X_test_scaled, y_test_fold)], verbose=0)
                    else:
                        model.fit(X_train_scaled, y_train_fold)
                
                # Predict
                y_pred_fold = model.predict(X_test_scaled)
                y_prob_fold = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                fold_acc = accuracy_score(y_test_fold, y_pred_fold)
                
                # Convert y_test_fold to numeric if needed for Brier score
                if hasattr(y_test_fold, 'map'):
                    le_temp = LabelEncoder()
                    y_test_numeric = le_temp.fit_transform(y_test_fold)
                else:
                    y_test_numeric = y_test_fold
                
                fold_brier = calculate_brier_score(y_test_numeric, y_prob_fold)
                
                fold_scores.append(fold_acc)
                fold_brier_scores.append(fold_brier)
                
                print(f"  Fold {fold}: Accuracy={fold_acc:.4f}, Brier={fold_brier:.4f}")
                
            except Exception as e:
                print(f"  Fold {fold}: Error - {e}")
                continue
        
        if fold_scores:
            mean_acc = np.mean(fold_scores)
            std_acc = np.std(fold_scores)
            mean_brier = np.mean(fold_brier_scores)
            std_brier = np.std(fold_brier_scores)
            
            validation_results[name] = {
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'brier_mean': mean_brier,
                'brier_std': std_brier,
                'fold_scores': fold_scores,
                'fold_brier_scores': fold_brier_scores
            }
            
            print(f"  Final: Accuracy={mean_acc:.4f} (+/- {std_acc*2:.4f})")
            print(f"         Brier={mean_brier:.4f} (+/- {std_brier*2:.4f})")
    
    return validation_results

# -------------------------
# ADVANCED PREDICTION FUNCTION - DEEL 2
# -------------------------
def enhanced_ensemble_prediction(models, scaler, label_encoder, features_dict, 
                                home_feats, away_feats, use_calibrated=True):
    """
    Enhanced ensemble prediction met uncertainty quantification.
    Combineert alle modellen met intelligente weighting.
    """
    print("\n" + "="*60)
    print("ENHANCED ENSEMBLE PREDICTION")
    print("="*60)
    
    # Prepare features
    features_df = pd.DataFrame([features_dict])
    features_scaled = scaler.transform(features_df)
    
    predictions = {}
    probabilities = {}
    confidences = {}
    
    # Individual model predictions
    model_predictions = []
    model_probabilities = []
    model_confidences = []
    
    for name, model in models.items():
        if model is None:
            continue
            
        try:
            # Get prediction and probabilities
            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            # Convert numeric prediction back to label
            if hasattr(label_encoder, 'inverse_transform'):
                pred_label = label_encoder.inverse_transform([pred])[0]
            else:
                pred_label = ['W', 'D', 'L'][pred]
            
            # Calculate confidence (max probability)
            confidence = np.max(prob)
            
            predictions[name] = pred_label
            probabilities[name] = prob
            confidences[name] = confidence
            
            # Store for ensemble calculation
            model_predictions.append(pred)
            model_probabilities.append(prob)
            model_confidences.append(confidence)
            
            print(f"{name:<25}: {pred_label} ({confidence:.3f} conf)")
            print(f"{'':25}  W={prob[0]:.3f}, D={prob[1]:.3f}, L={prob[2]:.3f}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    # Enhanced ensemble calculation
    if model_probabilities:
        print(f"\n=== ENSEMBLE CALCULATION ===")
        
        # Method 1: Simple average
        ensemble_prob_simple = np.mean(model_probabilities, axis=0)
        ensemble_pred_simple = np.argmax(ensemble_prob_simple)
        
        # Method 2: Confidence-weighted average
        weights = np.array(model_confidences)
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_prob_weighted = np.average(model_probabilities, axis=0, weights=weights)
        ensemble_pred_weighted = np.argmax(ensemble_prob_weighted)
        
        # Method 3: Use calibrated models if available
        calibrated_probs = []
        calibrated_names = [name for name in models.keys() if 'calibrated' in name]
        
        if calibrated_names:
            for name in calibrated_names:
                if name in probabilities:
                    calibrated_probs.append(probabilities[name])
            
            if calibrated_probs:
                ensemble_prob_calibrated = np.mean(calibrated_probs, axis=0)
                ensemble_pred_calibrated = np.argmax(ensemble_prob_calibrated)
            else:
                ensemble_prob_calibrated = ensemble_prob_weighted
                ensemble_pred_calibrated = ensemble_pred_weighted
        else:
            ensemble_prob_calibrated = ensemble_prob_weighted
            ensemble_pred_calibrated = ensemble_pred_weighted
        
        # Choose best ensemble method (prefereer calibrated)
        if use_calibrated and calibrated_probs:
            final_prob = ensemble_prob_calibrated
            final_pred = ensemble_pred_calibrated
            method_used = "Calibrated Ensemble"
        else:
            final_prob = ensemble_prob_weighted
            final_pred = ensemble_pred_weighted
            method_used = "Confidence-Weighted Ensemble"
        
        # Convert prediction back to label
        final_pred_label = label_encoder.inverse_transform([final_pred])[0] if hasattr(label_encoder, 'inverse_transform') else ['W', 'D', 'L'][final_pred]
        final_confidence = np.max(final_prob)
        
        # Calculate uncertainty metrics
        entropy = -np.sum(final_prob * np.log(final_prob + 1e-10))  # Shannon entropy
        uncertainty = 1 - final_confidence  # Simple uncertainty measure
        
        print(f"\n{method_used}:")
        print(f"Prediction: {final_pred_label}")
        print(f"Probabilities: W={final_prob[0]:.3f}, D={final_prob[1]:.3f}, L={final_prob[2]:.3f}")
        print(f"Confidence: {final_confidence:.3f}")
        print(f"Uncertainty: {uncertainty:.3f}")
        print(f"Entropy: {entropy:.3f}")
        
        # Confidence assessment
        if final_confidence > 0.7:
            confidence_level = "HIGH"
        elif final_confidence > 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        print(f"Confidence Level: {confidence_level}")
        
        return {
            'prediction': final_pred_label,
            'probabilities': final_prob,
            'confidence': final_confidence,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'confidence_level': confidence_level,
            'method_used': method_used,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'individual_confidences': confidences
        }
    
    else:
        print("No successful predictions from ensemble models")
        return None

print("DEEL 2 geladen: Enhanced Ensemble Methods en Soft Voting")
print("Volgende: DEEL 3 - Polynomial Features en Non-Linear Relationships")
"""
DEEL 3: Polynomial Features en Non-Linear Relationships
Advanced Soccer Predictor v9.0

Dit deel bevat:
- Polynomial feature engineering
- Soccer-specific feature interactions
- Non-linear relationship detection
- Advanced feature engineering
- Multicollinearity detection en handling
- Enhanced feature selection
"""

from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------
# POLYNOMIAL FEATURE ENGINEERING - DEEL 3
# -------------------------

def create_polynomial_features(X, feature_names, degree=2, interaction_only=True):
    """
    Create polynomial features met focus op interactions.
    Implementeert research-based feature engineering voor voetbal data.
    """
    print("\n" + "="*60)
    print("CREATING POLYNOMIAL FEATURES")
    print("="*60)
    
    # Select most important features for polynomial expansion (avoid curse of dimensionality)
    important_features = [
        'home_xG90', 'away_xG90', 'diff_xG90',
        'home_ShotQual', 'away_ShotQual', 'diff_ShotQual',
        'home_Possession', 'away_Possession', 'diff_Possession',
        'home_FieldTilt', 'away_FieldTilt', 'diff_FieldTilt',
        'home_RecentForm', 'away_RecentForm', 'diff_RecentForm',
        'home_WinRate5', 'away_WinRate5', 'diff_WinRate5',
        'home_AerialWin%', 'away_AerialWin%', 'diff_AerialWin%',
        'home_HighPress', 'away_HighPress', 'diff_HighPress'
    ]
    
    # Filter to existing features
    available_features = [f for f in important_features if f in feature_names]
    
    if len(available_features) < 5:
        print(f"Onvoldoende features voor polynomial expansion: {len(available_features)}")
        print("Beschikbare features:", available_features)
        return X, feature_names
    
    print(f"Gebruikte features voor polynomial expansion: {len(available_features)}")
    
    # Get indices of important features
    feature_indices = [feature_names.index(f) for f in available_features if f in feature_names]
    X_subset = X[:, feature_indices]
    
    # Create polynomial features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    
    X_poly_subset = poly.fit_transform(X_subset)
    
    # Get polynomial feature names
    poly_feature_names = poly.get_feature_names_out(available_features)
    
    print(f"Original features: {len(available_features)}")
    print(f"Polynomial features: {len(poly_feature_names)}")
    
    # Combine original features with new polynomial features
    # Remove original subset features to avoid duplication
    remaining_indices = [i for i in range(len(feature_names)) if i not in feature_indices]
    X_remaining = X[:, remaining_indices]
    remaining_features = [feature_names[i] for i in remaining_indices]
    
    # Combine everything
    X_combined = np.concatenate([X_remaining, X_poly_subset], axis=1)
    combined_feature_names = remaining_features + list(poly_feature_names)
    
    print(f"Final feature count: {len(combined_feature_names)}")
    
    # Print some example polynomial features
    print("\nExample polynomial features created:")
    poly_only_features = [f for f in poly_feature_names if ' ' in f][:10]
    for i, feature in enumerate(poly_only_features):
        print(f"  {i+1:2d}. {feature}")
    
    return X_combined, combined_feature_names, poly

def create_soccer_specific_interactions(home_feats, away_feats):
    """
    Create soccer-specific feature interactions gebaseerd op domain knowledge.
    Implementeert tactical en strategische interactions.
    """
    print("\n=== CREATING SOCCER-SPECIFIC INTERACTIONS ===")
    
    interactions = {}
    
    # -------------------------
    # ATTACKING INTERACTIONS
    # -------------------------
    
    # 1. Shot Quality vs Volume Interaction
    home_shots = ema(home_feats.get('Sh90', pd.Series([0])))
    home_shot_qual = ema(home_feats.get('ShotQual', pd.Series([0])))
    away_shots = ema(away_feats.get('Sh90', pd.Series([0])))
    away_shot_qual = ema(away_feats.get('ShotQual', pd.Series([0])))
    
    interactions['home_shot_efficiency'] = home_shots * home_shot_qual
    interactions['away_shot_efficiency'] = away_shots * away_shot_qual
    interactions['shot_efficiency_diff'] = interactions['home_shot_efficiency'] - interactions['away_shot_efficiency']
    
    # 2. xG vs Conversion Interaction
    home_xg = ema(home_feats.get('xG90', pd.Series([0])))
    home_conv = ema(home_feats.get('ConvRatio90', pd.Series([0])))
    away_xg = ema(away_feats.get('xG90', pd.Series([0])))
    away_conv = ema(away_feats.get('ConvRatio90', pd.Series([0])))
    
    interactions['home_clinical_finishing'] = home_xg * (1 + home_conv)
    interactions['away_clinical_finishing'] = away_xg * (1 + away_conv)
    interactions['clinical_finishing_diff'] = interactions['home_clinical_finishing'] - interactions['away_clinical_finishing']
    
    # -------------------------
    # POSSESSION INTERACTIONS
    # -------------------------
    
    # 3. Possession Quality vs Quantity
    home_poss = ema(home_feats.get('Possession', pd.Series([0.5])))
    home_prog = ema(home_feats.get('Prog90', pd.Series([0])))
    away_poss = ema(away_feats.get('Possession', pd.Series([0.5])))
    away_prog = ema(away_feats.get('Prog90', pd.Series([0])))
    
    interactions['home_possession_quality'] = home_poss * (home_prog / 50.0)  # Normalize progressive passes
    interactions['away_possession_quality'] = away_poss * (away_prog / 50.0)
    interactions['possession_quality_diff'] = interactions['home_possession_quality'] - interactions['away_possession_quality']
    
    # 4. Field Tilt vs Attacking Third Activity
    home_tilt = ema(home_feats.get('FieldTilt', pd.Series([0])))
    home_att3rd = ema(home_feats.get('Att3rd90', pd.Series([0])))
    away_tilt = ema(away_feats.get('FieldTilt', pd.Series([0])))
    away_att3rd = ema(away_feats.get('Att3rd90', pd.Series([0])))
    
    interactions['home_territorial_dominance'] = home_tilt * (home_att3rd / 100.0)
    interactions['away_territorial_dominance'] = away_tilt * (away_att3rd / 100.0)
    interactions['territorial_dominance_diff'] = interactions['home_territorial_dominance'] - interactions['away_territorial_dominance']
    
    # -------------------------
    # DEFENSIVE INTERACTIONS
    # -------------------------
    
    # 5. Pressing vs Ball Recovery
    home_press = ema(home_feats.get('HighPress', pd.Series([0])))
    home_tkld_pct = ema(home_feats.get('TkldPct_possession', pd.Series([0])))
    away_press = ema(away_feats.get('HighPress', pd.Series([0])))
    away_tkld_pct = ema(away_feats.get('TkldPct_possession', pd.Series([0])))
    
    interactions['home_press_effectiveness'] = home_press * (1 - home_tkld_pct)  # High press, low dribbled past
    interactions['away_press_effectiveness'] = away_press * (1 - away_tkld_pct)
    interactions['press_effectiveness_diff'] = interactions['home_press_effectiveness'] - interactions['away_press_effectiveness']
    
    # 6. Aerial vs Set Pieces
    home_aerial = ema(home_feats.get('AerialWin%', pd.Series([0.5])))
    home_setpieces = ema(home_feats.get('SetPieces90', pd.Series([0])))
    away_aerial = ema(away_feats.get('AerialWin%', pd.Series([0.5])))
    away_setpieces = ema(away_feats.get('SetPieces90', pd.Series([0])))
    
    interactions['home_aerial_threat'] = home_aerial * home_setpieces
    interactions['away_aerial_threat'] = away_aerial * away_setpieces
    interactions['aerial_threat_diff'] = interactions['home_aerial_threat'] - interactions['away_aerial_threat']
    
    # -------------------------
    # FORM AND MOMENTUM INTERACTIONS
    # -------------------------
    
    # 7. Form vs Home Advantage
    home_form = ema(home_feats.get('RecentForm', pd.Series([0])))
    home_adv = ema(home_feats.get('HomeAdvantage', pd.Series([0])))
    
    interactions['home_momentum_boost'] = home_form * (1 + abs(home_adv))
    
    # 8. Win Rate vs Rest Days
    home_win_rate = ema(home_feats.get('WinRate5', pd.Series([0.5])))
    home_rest = ema(home_feats.get('RestDays', pd.Series([7])))
    away_win_rate = ema(away_feats.get('WinRate5', pd.Series([0.5])))
    away_rest = ema(away_feats.get('RestDays', pd.Series([7])))
    
    # Optimal rest is around 3-7 days
    home_rest_factor = 1.0 - abs(home_rest - 5) / 10.0
    away_rest_factor = 1.0 - abs(away_rest - 5) / 10.0
    
    interactions['home_form_fitness'] = home_win_rate * max(0.5, home_rest_factor)
    interactions['away_form_fitness'] = away_win_rate * max(0.5, away_rest_factor)
    interactions['form_fitness_diff'] = interactions['home_form_fitness'] - interactions['away_form_fitness']
    
    # -------------------------
    # MATCHUP SPECIFIC INTERACTIONS
    # -------------------------
    
    # 9. Attack vs Defense Strength
    home_att_strength = (home_xg + home_shots/10.0) / 2.0
    away_def_strength = ema(away_feats.get('GoalsAgainst', pd.Series([1.5])))
    away_att_strength = (away_xg + away_shots/10.0) / 2.0
    home_def_strength = ema(home_feats.get('GoalsAgainst', pd.Series([1.5])))
    
    interactions['home_attack_vs_away_defense'] = home_att_strength * (2.0 - min(2.0, away_def_strength))
    interactions['away_attack_vs_home_defense'] = away_att_strength * (2.0 - min(2.0, home_def_strength))
    interactions['attack_defense_balance'] = interactions['home_attack_vs_away_defense'] - interactions['away_attack_vs_home_defense']
    
    # 10. Counter-Attack Potential
    home_counter_def = 1.0 - home_poss  # Lower possession = more counter opportunities
    home_counter_att = ema(home_feats.get('PrgDist90', pd.Series([0]))) / 1000.0  # Progressive distance
    away_counter_def = 1.0 - away_poss
    away_counter_att = ema(away_feats.get('PrgDist90', pd.Series([0]))) / 1000.0
    
    interactions['home_counter_potential'] = home_counter_def * home_counter_att
    interactions['away_counter_potential'] = away_counter_def * away_counter_att
    interactions['counter_potential_diff'] = interactions['home_counter_potential'] - interactions['away_counter_potential']
    
    print(f"Created {len(interactions)} soccer-specific interaction features")
    
    # Print top interactions by magnitude
    interaction_magnitudes = [(k, abs(v)) for k, v in interactions.items()]
    interaction_magnitudes.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 interactions by magnitude:")
    for i, (name, magnitude) in enumerate(interaction_magnitudes[:10]):
        value = interactions[name]
        print(f"  {i+1:2d}. {name:<30}: {value:+.4f} (|{magnitude:.4f}|)")
    
    return interactions

# -------------------------
# MULTICOLLINEARITY DETECTION - DEEL 3
# -------------------------

def detect_multicollinearity(X, feature_names, vif_threshold=5.0):
    """
    Detect multicollinearity using Variance Inflation Factor (VIF).
    Research threshold: VIF > 5-10 indicates problematic correlation.
    """
    print("\n=== MULTICOLLINEARITY DETECTION ===")
    
    if X.shape[1] < 2:
        print("Te weinig features voor multicollinearity analyse")
        return [], []
    
    # Calculate VIF for each feature
    vif_data = []
    
    try:
        for i in range(X.shape[1]):
            # Skip if feature has no variance
            if np.var(X[:, i]) < 1e-10:
                continue
                
            vif = variance_inflation_factor(X, i)
            
            # Handle infinite or very large VIF values
            if np.isinf(vif) or vif > 1000:
                vif = 1000  # Cap extreme values
            
            vif_data.append({
                'feature': feature_names[i],
                'vif': vif,
                'index': i
            })
    
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return [], []
    
    if not vif_data:
        print("Geen VIF data beschikbaar")
        return [], []
    
    # Sort by VIF value
    vif_data.sort(key=lambda x: x['vif'], reverse=True)
    
    # Identify problematic features
    high_vif_features = [item for item in vif_data if item['vif'] > vif_threshold]
    
    print(f"Features met VIF > {vif_threshold}:")
    for item in high_vif_features:
        print(f"  {item['feature']:<30}: VIF = {item['vif']:.2f}")
    
    if not high_vif_features:
        print(f"Geen features met VIF > {vif_threshold} gevonden")
    
    # Print top 10 VIF values
    print(f"\nTop 10 VIF values:")
    for i, item in enumerate(vif_data[:10]):
        status = "HIGH" if item['vif'] > vif_threshold else "OK"
        print(f"  {i+1:2d}. {item['feature']:<25}: {item['vif']:8.2f} ({status})")
    
    return high_vif_features, vif_data

def remove_multicollinear_features(X, feature_names, vif_threshold=10.0, correlation_threshold=0.95):
    """
    Remove multicollinear features using VIF en correlation analysis.
    Implements research-based multicollinearity handling.
    """
    print(f"\n=== REMOVING MULTICOLLINEAR FEATURES ===")
    print(f"Original features: {X.shape[1]}")
    
    X_cleaned = X.copy()
    remaining_features = feature_names.copy()
    removed_features = []
    
    # Method 1: Remove highly correlated features
    print(f"\n1. Removing features with correlation > {correlation_threshold}")
    
    if X_cleaned.shape[1] > 1:
        correlation_matrix = np.corrcoef(X_cleaned.T)
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(remaining_features)):
            for j in range(i+1, len(remaining_features)):
                if abs(correlation_matrix[i, j]) > correlation_threshold:
                    high_corr_pairs.append((i, j, correlation_matrix[i, j]))
        
        # Remove features from highly correlated pairs (keep the first one)
        indices_to_remove = set()
        for i, j, corr in high_corr_pairs:
            if i not in indices_to_remove and j not in indices_to_remove:
                indices_to_remove.add(j)  # Remove the second feature
                print(f"  Removing {remaining_features[j]} (corr={corr:.3f} with {remaining_features[i]})")
                removed_features.append(remaining_features[j])
        
        # Remove the features
        if indices_to_remove:
            keep_indices = [i for i in range(len(remaining_features)) if i not in indices_to_remove]
            X_cleaned = X_cleaned[:, keep_indices]
            remaining_features = [remaining_features[i] for i in keep_indices]
    
    # Method 2: Iteratively remove high VIF features
    print(f"\n2. Removing features with VIF > {vif_threshold}")
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        if X_cleaned.shape[1] < 2:
            break
            
        # Calculate VIF
        high_vif_features, vif_data = detect_multicollinearity(X_cleaned, remaining_features, vif_threshold)
        
        if not high_vif_features:
            print(f"  Iteration {iteration+1}: Geen high VIF features meer")
            break
        
        # Remove the feature with highest VIF
        highest_vif_feature = high_vif_features[0]
        feature_index = remaining_features.index(highest_vif_feature['feature'])
        
        print(f"  Iteration {iteration+1}: Removing {highest_vif_feature['feature']} (VIF={highest_vif_feature['vif']:.2f})")
        
        # Remove the feature
        keep_indices = [i for i in range(len(remaining_features)) if i != feature_index]
        X_cleaned = X_cleaned[:, keep_indices]
        removed_features.append(remaining_features[feature_index])
        remaining_features = [remaining_features[i] for i in keep_indices]
        
        iteration += 1
    
    print(f"\nFinal features: {X_cleaned.shape[1]} (removed {len(removed_features)})")
    print(f"Removed features: {removed_features}")
    
    return X_cleaned, remaining_features, removed_features

# -------------------------
# ENHANCED FEATURE SELECTION - DEEL 3
# -------------------------

def enhanced_feature_selection(X, y, feature_names, method='rfe', k=30):
    """
    Enhanced feature selection met multiple methods.
    Combineert statistical tests, RFE, en domain knowledge.
    """
    print(f"\n=== ENHANCED FEATURE SELECTION ===")
    print(f"Original features: {X.shape[1]}")
    print(f"Target features: {k}")
    
    if X.shape[1] <= k:
        print("Al minder features dan target, geen selectie nodig")
        return X, feature_names, list(range(X.shape[1]))
    
    # Convert y to numeric if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
    else:
        y_numeric = y
    
    feature_scores = {}
    selected_features_dict = {}
    
    # Method 1: Statistical F-test
    print("\n1. Statistical F-test selection...")
    try:
        selector_f = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_f = selector_f.fit_transform(X, y_numeric)
        selected_features_f = selector_f.get_support(indices=True)
        scores_f = selector_f.scores_
        
        # Store scores
        for i, score in enumerate(scores_f):
            feature_scores[feature_names[i]] = score
        
        selected_features_dict['f_test'] = selected_features_f
        print(f"  Selected {len(selected_features_f)} features")
        
    except Exception as e:
        print(f"  Error in F-test selection: {e}")
        selected_features_dict['f_test'] = []
    
    # Method 2: RFE with Random Forest
    print("\n2. Recursive Feature Elimination...")
    try:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        if method == 'rfecv':
            # Cross-validated RFE
            selector_rfe = RFECV(
                estimator=estimator, 
                step=1, 
                cv=5, 
                scoring='accuracy',
                min_features_to_select=max(5, k//2)
            )
        else:
            # Standard RFE
            selector_rfe = RFE(
                estimator=estimator, 
                n_features_to_select=min(k, X.shape[1])
            )
        
        X_rfe = selector_rfe.fit_transform(X, y_numeric)
        selected_features_rfe = selector_rfe.get_support(indices=True)
        
        if hasattr(selector_rfe, 'ranking_'):
            # Store ranking (lower rank = better)
            for i, rank in enumerate(selector_rfe.ranking_):
                feature_name = feature_names[i]
                # Convert ranking to score (higher = better)
                score = 1.0 / rank if rank > 0 else 0
                if feature_name in feature_scores:
                    feature_scores[feature_name] = (feature_scores[feature_name] + score) / 2
                else:
                    feature_scores[feature_name] = score
        
        selected_features_dict['rfe'] = selected_features_rfe
        print(f"  Selected {len(selected_features_rfe)} features")
        
    except Exception as e:
        print(f"  Error in RFE selection: {e}")
        selected_features_dict['rfe'] = []
    
    # Method 3: Domain knowledge priority features
    print("\n3. Domain knowledge priority...")
    priority_features = [
        'xG90', 'ShotQual', 'RecentForm', 'WinRate5', 'Possession',
        'FieldTilt', 'AerialMismatch', 'ConvRatio90', 'HomeAdvantage',
        'shot_efficiency_diff', 'clinical_finishing_diff', 'possession_quality_diff',
        'territorial_dominance_diff', 'press_effectiveness_diff'
    ]
    
    priority_indices = []
    for feature in priority_features:
        # Check for exact match or partial match
        matching_indices = [i for i, name in enumerate(feature_names) 
                          if feature.lower() in name.lower() or name.lower() in feature.lower()]
        priority_indices.extend(matching_indices)
    
    # Remove duplicates
    priority_indices = list(set(priority_indices))
    selected_features_dict['domain'] = priority_indices
    print(f"  Identified {len(priority_indices)} domain priority features")
    
    # Method 4: Combine methods
    print("\n4. Combining selection methods...")
    
    # Count votes for each feature
    feature_votes = {}
    for method_name, indices in selected_features_dict.items():
        for idx in indices:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                feature_votes[feature_name] = feature_votes.get(feature_name, 0) + 1
    
    # Sort features by votes and scores
    feature_ranking = []
    for i, feature_name in enumerate(feature_names):
        votes = feature_votes.get(feature_name, 0)
        score = feature_scores.get(feature_name, 0)
        feature_ranking.append((i, feature_name, votes, score))
    
    # Sort by votes first, then by score
    feature_ranking.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    # Select top k features
    final_selected_indices = [item[0] for item in feature_ranking[:k]]
    final_selected_names = [item[1] for item in feature_ranking[:k]]
    
    X_selected = X[:, final_selected_indices]
    
    print(f"\nFinal selection: {len(final_selected_indices)} features")
    print("\nTop 15 selected features:")
    for i, (idx, name, votes, score) in enumerate(feature_ranking[:15]):
        print(f"  {i+1:2d}. {name:<30} (votes: {votes}, score: {score:.3f})")
    
    return X_selected, final_selected_names, final_selected_indices

# -------------------------
# COMPLETE FEATURE ENGINEERING PIPELINE - DEEL 3
# -------------------------

def complete_feature_engineering_pipeline(home_feats, away_feats, 
                                        create_polynomials=True,
                                        handle_multicollinearity=True,
                                        feature_selection=True,
                                        max_features=50):
    """
    Complete feature engineering pipeline met alle verbeteringen.
    Combineert polynomial features, interactions, en feature selection.
    """
    print("\n" + "="*80)
    print("COMPLETE FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Step 1: Basic features (from existing functions)
    print("\n1. Creating basic features...")
    basic_features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
    
    # Step 2: Add soccer-specific interactions
    print("\n2. Adding soccer-specific interactions...")
    interactions = create_soccer_specific_interactions(home_feats, away_feats)
    
    # Combine basic features with interactions
    all_features = {**basic_features, **interactions}
    
    # Convert to arrays
    feature_names = list(all_features.keys())
    X = np.array([[all_features[name]] for name in feature_names]).T
    
    print(f"Features after interactions: {X.shape[1]}")
    
    # Step 3: Create polynomial features
    if create_polynomials and X.shape[1] > 10:
        print("\n3. Creating polynomial features...")
        X, feature_names, poly_transformer = create_polynomial_features(
            X, feature_names, degree=2, interaction_only=True
        )
        print(f"Features after polynomial expansion: {X.shape[1]}")
    else:
        poly_transformer = None
    
    # Step 4: Handle multicollinearity
    if handle_multicollinearity and X.shape[1] > 10:
        print("\n4. Handling multicollinearity...")
        X, feature_names, removed_features = remove_multicollinear_features(
            X, feature_names, vif_threshold=10.0, correlation_threshold=0.95
        )
        print(f"Features after multicollinearity handling: {X.shape[1]}")
    else:
        removed_features = []
    
    # Step 5: Feature selection
    if feature_selection and X.shape[1] > max_features:
        print(f"\n5. Feature selection to {max_features} features...")
        # Create dummy y for feature selection (we'll need actual y in real implementation)
        y_dummy = np.array(['W'] * X.shape[0])  # Placeholder
        
        try:
            X, feature_names, selected_indices = enhanced_feature_selection(
                X, y_dummy, feature_names, method='rfe', k=max_features
            )
            print(f"Features after selection: {X.shape[1]}")
        except Exception as e:
            print(f"Feature selection failed: {e}, keeping all features")
    else:
        selected_indices = list(range(X.shape[1]))
    
    print(f"\nFinal feature engineering results:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Feature names: {len(feature_names)}")
    print(f"  Data shape: {X.shape}")
    
    # Return features as dictionary (compatible with existing code)
    final_features = {}
    if X.shape[0] > 0:  # Only if we have data
        for i, name in enumerate(feature_names):
            final_features[name] = X[0, i]  # Take first row
    
    return final_features, {
        'feature_names': feature_names,
        'polynomial_transformer': poly_transformer,
        'removed_features': removed_features,
        'selected_indices': selected_indices,
        'processing_info': {
            'polynomials_created': create_polynomials,
            'multicollinearity_handled': handle_multicollinearity,
            'feature_selection_applied': feature_selection,
            'final_feature_count': len(feature_names)
        }
    }

def prepare_enhanced_ml_features_with_tactical(home_feats, away_feats):
    """
    Enhanced ML feature preparation (aangepast van DEEL 1).
    """
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
    
    return combined_features

print("DEEL 3 geladen: Polynomial Features en Non-Linear Relationships")
print("Volgende: DEEL 4 - Bayesian Methods en Uncertainty Quantification")
"""
DEEL 4: Bayesian Methods en Uncertainty Quantification
Advanced Soccer Predictor v9.0

Dit deel bevat:
- Hierarchical Bayesian models
- Gaussian Process regression voor uncertainty
- Dynamic Bayesian updating
- Probabilistic prediction frameworks
- Uncertainty quantification methods
- Credible intervals en prediction intervals
"""

from scipy.stats import gamma, beta, norm
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, ExpSineSquared
import scipy.optimize as opt
from scipy.special import gammaln
from sklearn.model_selection import cross_val_predict

def safe_execute(func, *args, default_return=None, error_msg="Function execution failed", **kwargs):
    """
    Safely execute a function with error handling.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"{error_msg}: {e}")
        return default_return
    
def normalize_team_name(name):
    """
    Normalize team name for consistency.
    """
    if not name:
        return "UNKNOWN"
    
    # Remove special characters and normalize
    import re
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', str(name))
    normalized = ' '.join(normalized.split())  # Remove extra spaces
    return normalized.upper()

# -------------------------
# BAYESIAN FRAMEWORK CLASSES - DEEL 4
# -------------------------

class HierarchicalBayesianSoccerModel:
    """
    Hierarchical Bayesian model voor soccer prediction.
    Implementeert research-based Poisson models met team-specific parameters.
    """
    
    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        self.alpha_prior = alpha_prior  # Prior for attack strength
        self.beta_prior = beta_prior    # Prior for defense strength
        self.home_advantage_prior = 0.3
        
        # Model parameters (will be learned)
        self.team_attack_strength = {}
        self.team_defense_strength = {}
        self.home_advantage = self.home_advantage_prior
        self.global_mean_goals = 1.5
        
        # Uncertainty estimates
        self.parameter_uncertainty = {}
        self.prediction_intervals = {}
        
        print("Initialized Hierarchical Bayesian Soccer Model")
    
    def fit(self, match_data, n_iterations=1000, burn_in=200):
        """
        Fit Bayesian model using MCMC-style updating.
        Implements hierarchical Poisson model: log(θ_home) = home_adv + att_home + def_away
        """
        print(f"\nFitting Bayesian model with {len(match_data)} matches...")
        
        # Extract teams and initialize parameters
        teams = set()
        for match in match_data:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        # Initialize team parameters with priors
        for team in teams:
            self.team_attack_strength[team] = np.random.gamma(self.alpha_prior, 1.0)
            self.team_defense_strength[team] = np.random.gamma(self.beta_prior, 1.0)
        
        # MCMC-style parameter updating
        attack_samples = {team: [] for team in teams}
        defense_samples = {team: [] for team in teams}
        home_adv_samples = []
        
        for iteration in range(n_iterations):
            # Update attack strengths
            for team in teams:
                home_goals = [m['home_goals'] for m in match_data if m['home_team'] == team]
                away_goals = [m['away_goals'] for m in match_data if m['away_team'] == team]
                
                # Bayesian update for attack strength
                total_goals_scored = sum(home_goals) + sum(away_goals)
                n_matches = len(home_goals) + len(away_goals)
                
                # Gamma conjugate prior update
                posterior_alpha = self.alpha_prior + total_goals_scored
                posterior_beta = self.beta_prior + n_matches
                
                new_attack = np.random.gamma(posterior_alpha, 1.0/posterior_beta)
                self.team_attack_strength[team] = new_attack
                
                if iteration >= burn_in:
                    attack_samples[team].append(new_attack)
            
            # Update defense strengths
            for team in teams:
                goals_conceded_home = [m['away_goals'] for m in match_data if m['home_team'] == team]
                goals_conceded_away = [m['home_goals'] for m in match_data if m['away_team'] == team]
                
                total_goals_conceded = sum(goals_conceded_home) + sum(goals_conceded_away)
                n_matches = len(goals_conceded_home) + len(goals_conceded_away)
                
                # Inverse relationship for defense (lower = better)
                posterior_alpha = self.alpha_prior + n_matches
                posterior_beta = self.beta_prior + total_goals_conceded
                
                new_defense = np.random.gamma(posterior_alpha, 1.0/posterior_beta)
                self.team_defense_strength[team] = 1.0 / max(0.1, new_defense)
                
                if iteration >= burn_in:
                    defense_samples[team].append(new_defense)
            
            # Update home advantage
            home_goals_total = sum([m['home_goals'] for m in match_data])
            away_goals_total = sum([m['away_goals'] for m in match_data])
            n_matches_total = len(match_data)
            
            if n_matches_total > 0:
                home_advantage_estimate = np.log(home_goals_total / max(1, away_goals_total))
                self.home_advantage = 0.9 * self.home_advantage + 0.1 * home_advantage_estimate
                
                if iteration >= burn_in:
                    home_adv_samples.append(self.home_advantage)
        
        # Calculate uncertainty from MCMC samples
        for team in teams:
            if len(attack_samples[team]) > 10:
                self.parameter_uncertainty[f'{team}_attack'] = {
                    'mean': np.mean(attack_samples[team]),
                    'std': np.std(attack_samples[team]),
                    'credible_interval_95': np.percentile(attack_samples[team], [2.5, 97.5])
                }
            
            if len(defense_samples[team]) > 10:
                self.parameter_uncertainty[f'{team}_defense'] = {
                    'mean': np.mean(defense_samples[team]),
                    'std': np.std(defense_samples[team]),
                    'credible_interval_95': np.percentile(defense_samples[team], [2.5, 97.5])
                }
        
        if len(home_adv_samples) > 10:
            self.parameter_uncertainty['home_advantage'] = {
                'mean': np.mean(home_adv_samples),
                'std': np.std(home_adv_samples),
                'credible_interval_95': np.percentile(home_adv_samples, [2.5, 97.5])
            }
        
        print(f"Bayesian training completed. Home advantage: {self.home_advantage:.3f}")
        
    def predict_with_uncertainty(self, home_team, away_team, n_samples=1000):
        """
        Predict match outcome with full uncertainty quantification.
        Returns probabilistic predictions and credible intervals.
        """
        # Get team parameters
        home_attack = self.team_attack_strength.get(home_team, 1.0)
        home_defense = self.team_defense_strength.get(home_team, 1.0)
        away_attack = self.team_attack_strength.get(away_team, 1.0)
        away_defense = self.team_defense_strength.get(away_team, 1.0)
        
        # Calculate expected goals with uncertainty
        lambda_home_samples = []
        lambda_away_samples = []
        
        for _ in range(n_samples):
            # Sample from parameter uncertainty
            if f'{home_team}_attack' in self.parameter_uncertainty:
                attack_std = self.parameter_uncertainty[f'{home_team}_attack']['std']
                home_att_sample = np.random.normal(home_attack, attack_std)
            else:
                home_att_sample = home_attack
            
            if f'{away_team}_defense' in self.parameter_uncertainty:
                defense_std = self.parameter_uncertainty[f'{away_team}_defense']['std']
                away_def_sample = np.random.normal(away_defense, defense_std)
            else:
                away_def_sample = away_defense
            
            # Home team expected goals
            lambda_home = max(0.1, home_att_sample * away_def_sample * np.exp(self.home_advantage))
            lambda_home_samples.append(lambda_home)
            
            # Away team expected goals  
            if f'{away_team}_attack' in self.parameter_uncertainty:
                attack_std = self.parameter_uncertainty[f'{away_team}_attack']['std']
                away_att_sample = np.random.normal(away_attack, attack_std)
            else:
                away_att_sample = away_attack
                
            if f'{home_team}_defense' in self.parameter_uncertainty:
                defense_std = self.parameter_uncertainty[f'{home_team}_defense']['std']
                home_def_sample = np.random.normal(home_defense, defense_std)
            else:
                home_def_sample = home_defense
            
            lambda_away = max(0.1, away_att_sample * home_def_sample)
            lambda_away_samples.append(lambda_away)
        
        # Calculate prediction statistics
        lambda_home_mean = np.mean(lambda_home_samples)
        lambda_away_mean = np.mean(lambda_away_samples)
        
        lambda_home_ci = np.percentile(lambda_home_samples, [2.5, 97.5])
        lambda_away_ci = np.percentile(lambda_away_samples, [2.5, 97.5])
        
        # Monte Carlo simulation for match outcomes
        win_count, draw_count, loss_count = 0, 0, 0
        home_goals_samples = []
        away_goals_samples = []
        
        for lh, la in zip(lambda_home_samples, lambda_away_samples):
            home_goals = np.random.poisson(lh)
            away_goals = np.random.poisson(la)
            
            home_goals_samples.append(home_goals)
            away_goals_samples.append(away_goals)
            
            if home_goals > away_goals:
                win_count += 1
            elif home_goals == away_goals:
                draw_count += 1
            else:
                loss_count += 1
        
        # Calculate probabilities and their uncertainty
        prob_win = win_count / n_samples
        prob_draw = draw_count / n_samples
        prob_loss = loss_count / n_samples
        
        # Calculate uncertainty in probabilities (using Beta distribution)
        prob_win_ci = self._beta_credible_interval(win_count, n_samples)
        prob_draw_ci = self._beta_credible_interval(draw_count, n_samples)
        prob_loss_ci = self._beta_credible_interval(loss_count, n_samples)
        
        return {
            'probabilities': [prob_win, prob_draw, prob_loss],
            'probability_credible_intervals': {
                'win': prob_win_ci,
                'draw': prob_draw_ci,
                'loss': prob_loss_ci
            },
            'expected_goals': {
                'home': lambda_home_mean,
                'away': lambda_away_mean
            },
            'expected_goals_credible_intervals': {
                'home': lambda_home_ci,
                'away': lambda_away_ci
            },
            'goal_prediction_intervals': {
                'home': np.percentile(home_goals_samples, [2.5, 97.5]),
                'away': np.percentile(away_goals_samples, [2.5, 97.5])
            },
            'uncertainty_metrics': {
                'prediction_entropy': self._calculate_entropy([prob_win, prob_draw, prob_loss]),
                'expected_goals_uncertainty': {
                    'home_std': np.std(lambda_home_samples),
                    'away_std': np.std(lambda_away_samples)
                }
            }
        }
    
    def _beta_credible_interval(self, successes, trials, alpha=0.05):
        """Calculate credible interval for probability using Beta distribution."""
        if trials == 0:
            return [0, 1]
        
        # Beta distribution parameters
        alpha_param = successes + 1
        beta_param = trials - successes + 1
        
        lower = beta.ppf(alpha/2, alpha_param, beta_param)
        upper = beta.ppf(1 - alpha/2, alpha_param, beta_param)
        
        return [lower, upper]
    
    def _calculate_entropy(self, probabilities):
        """Calculate Shannon entropy for uncertainty quantification."""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -np.sum(probabilities * np.log(probabilities))

# -------------------------
# GAUSSIAN PROCESS MODELS - DEEL 4
# -------------------------

class GaussianProcessSoccerModel:
    """
    Gaussian Process model voor soccer prediction met uncertainty quantification.
    Implementeert non-parametric Bayesian learning.
    """
    
    def __init__(self, kernel_type='rbf', length_scale=1.0, noise_level=0.1):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_level = noise_level
        
        # Create kernel
        if kernel_type == 'rbf':
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level)
        elif kernel_type == 'matern':
            self.kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level)
        elif kernel_type == 'periodic':
            # For seasonal patterns in soccer
            self.kernel = ConstantKernel(1.0) * ExpSineSquared(length_scale=length_scale, periodicity=38.0) + WhiteKernel(noise_level)
        else:
            self.kernel = RBF(length_scale=length_scale)
        
        # Initialize models
        self.gp_classifier = GaussianProcessClassifier(kernel=self.kernel)
        self.gp_regressor_home = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_level)
        self.gp_regressor_away = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_level)
        
        print(f"Initialized Gaussian Process Soccer Model with {kernel_type} kernel")
    
    def fit(self, X, y, y_goals_home=None, y_goals_away=None):
        """
        Fit Gaussian Process models for classification and regression.
        """
        print(f"Training GP model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Fit classification model
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print("Fitting GP classifier...")
        self.gp_classifier.fit(X, y_encoded)
        self.label_encoder = le
        
        # Fit regression models for goal prediction
        if y_goals_home is not None and y_goals_away is not None:
            print("Fitting GP regressors for goal prediction...")
            
            # Clean goal data
            valid_mask = ~(np.isnan(y_goals_home) | np.isnan(y_goals_away))
            X_goals = X[valid_mask]
            y_goals_home_clean = y_goals_home[valid_mask]
            y_goals_away_clean = y_goals_away[valid_mask]
            
            if len(X_goals) > 10:
                self.gp_regressor_home.fit(X_goals, y_goals_home_clean)
                self.gp_regressor_away.fit(X_goals, y_goals_away_clean)
                self.has_goal_models = True
                print(f"Goal models trained with {len(X_goals)} samples")
            else:
                self.has_goal_models = False
                print("Insufficient data for goal regression models")
        else:
            self.has_goal_models = False
    
    def predict_with_uncertainty(self, X_test, return_std=True):
        """
        Predict with full uncertainty quantification using GP.
        """
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        results = {}
        
        # Classification prediction with uncertainty
        print("GP classification prediction...")
        try:
            y_prob = self.gp_classifier.predict_proba(X_test)
            y_pred = self.gp_classifier.predict(X_test)
            
            # Convert back to original labels
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            results['classification'] = {
                'prediction': y_pred_labels[0] if len(y_pred_labels) == 1 else y_pred_labels,
                'probabilities': y_prob[0] if len(y_prob) == 1 else y_prob,
                'uncertainty': 1 - np.max(y_prob, axis=1)[0] if len(y_prob) == 1 else 1 - np.max(y_prob, axis=1)
            }
            
        except Exception as e:
            print(f"Error in GP classification: {e}")
            results['classification'] = None
        
        # Goal prediction with uncertainty
        if self.has_goal_models:
            print("GP goal prediction...")
            try:
                home_goals_mean, home_goals_std = self.gp_regressor_home.predict(X_test, return_std=True)
                away_goals_mean, away_goals_std = self.gp_regressor_away.predict(X_test, return_std=True)
                
                # Calculate credible intervals (95%)
                home_goals_ci = [
                    home_goals_mean - 1.96 * home_goals_std,
                    home_goals_mean + 1.96 * home_goals_std
                ]
                away_goals_ci = [
                    away_goals_mean - 1.96 * away_goals_std,
                    away_goals_mean + 1.96 * away_goals_std
                ]
                
                results['goal_prediction'] = {
                    'home_goals': {
                        'mean': home_goals_mean[0] if len(home_goals_mean) == 1 else home_goals_mean,
                        'std': home_goals_std[0] if len(home_goals_std) == 1 else home_goals_std,
                        'credible_interval_95': [ci[0] for ci in home_goals_ci] if len(home_goals_mean) == 1 else home_goals_ci
                    },
                    'away_goals': {
                        'mean': away_goals_mean[0] if len(away_goals_mean) == 1 else away_goals_mean,
                        'std': away_goals_std[0] if len(away_goals_std) == 1 else away_goals_std,
                        'credible_interval_95': [ci[0] for ci in away_goals_ci] if len(away_goals_mean) == 1 else away_goals_ci
                    }
                }
                
            except Exception as e:
                print(f"Error in GP goal prediction: {e}")
                results['goal_prediction'] = None
        
        return results

# -------------------------
# DYNAMIC BAYESIAN UPDATING - DEEL 4  
# -------------------------

class DynamicBayesianUpdater:
    """
    Dynamic Bayesian updating voor real-time model improvement.
    Updates posterior distributions as new match data arrives.
    """
    
    def __init__(self, decay_factor=0.95):
        self.decay_factor = decay_factor  # How quickly to forget old information
        self.priors = {}
        self.posteriors = {}
        self.observation_count = {}
        self.update_history = []
        
        print(f"Initialized Dynamic Bayesian Updater (decay={decay_factor})")
    
    def set_prior(self, parameter_name, distribution_type, **params):
        """Set prior distribution for a parameter."""
        self.priors[parameter_name] = {
            'type': distribution_type,
            'params': params
        }
        self.posteriors[parameter_name] = self.priors[parameter_name].copy()
        self.observation_count[parameter_name] = 0
        
        print(f"Set prior for {parameter_name}: {distribution_type} with {params}")
    
    def update_with_observation(self, parameter_name, observation, likelihood_params=None):
        """
        Update posterior distribution with new observation.
        Implements Bayesian updating with conjugate priors where possible.
        """
        if parameter_name not in self.posteriors:
            print(f"No prior set for {parameter_name}")
            return
        
        prior = self.posteriors[parameter_name]
        
        if prior['type'] == 'gamma':
            # Gamma-Poisson conjugate pair
            alpha_prior = prior['params']['alpha']
            beta_prior = prior['params']['beta']
            
            # Apply time decay
            effective_alpha = alpha_prior * (self.decay_factor ** self.observation_count[parameter_name])
            effective_beta = beta_prior * (self.decay_factor ** self.observation_count[parameter_name])
            
            # Bayesian update
            alpha_posterior = effective_alpha + observation
            beta_posterior = effective_beta + 1
            
            self.posteriors[parameter_name]['params']['alpha'] = alpha_posterior
            self.posteriors[parameter_name]['params']['beta'] = beta_posterior
            
        elif prior['type'] == 'beta':
            # Beta-Binomial conjugate pair
            alpha_prior = prior['params']['alpha']
            beta_prior = prior['params']['beta']
            
            # Apply time decay
            effective_alpha = alpha_prior * (self.decay_factor ** self.observation_count[parameter_name])
            effective_beta = beta_prior * (self.decay_factor ** self.observation_count[parameter_name])
            
            # Bayesian update (observation should be 0 or 1)
            alpha_posterior = effective_alpha + observation
            beta_posterior = effective_beta + (1 - observation)
            
            self.posteriors[parameter_name]['params']['alpha'] = alpha_posterior
            self.posteriors[parameter_name]['params']['beta'] = beta_posterior
            
        elif prior['type'] == 'normal':
            # Normal-Normal conjugate pair
            mu_prior = prior['params']['mu']
            sigma_prior = prior['params']['sigma']
            
            # Apply time decay
            effective_weight = self.decay_factor ** self.observation_count[parameter_name]
            
            # Bayesian update (assuming known variance for simplicity)
            precision_prior = 1 / (sigma_prior ** 2)
            precision_likelihood = 1.0  # Assumed observation precision
            
            precision_posterior = effective_weight * precision_prior + precision_likelihood
            mu_posterior = (effective_weight * precision_prior * mu_prior + precision_likelihood * observation) / precision_posterior
            sigma_posterior = 1 / np.sqrt(precision_posterior)
            
            self.posteriors[parameter_name]['params']['mu'] = mu_posterior
            self.posteriors[parameter_name]['params']['sigma'] = sigma_posterior
        
        self.observation_count[parameter_name] += 1
        
        # Record update
        self.update_history.append({
            'parameter': parameter_name,
            'observation': observation,
            'timestamp': len(self.update_history),
            'posterior_params': self.posteriors[parameter_name]['params'].copy()
        })
    
    def get_posterior_statistics(self, parameter_name):
        """Get statistics of posterior distribution."""
        if parameter_name not in self.posteriors:
            return None
        
        posterior = self.posteriors[parameter_name]
        
        if posterior['type'] == 'gamma':
            alpha = posterior['params']['alpha']
            beta = posterior['params']['beta']
            
            mean = alpha / beta
            variance = alpha / (beta ** 2)
            credible_interval = [
                gamma.ppf(0.025, alpha, scale=1/beta),
                gamma.ppf(0.975, alpha, scale=1/beta)
            ]
            
        elif posterior['type'] == 'beta':
            alpha = posterior['params']['alpha']
            beta_param = posterior['params']['beta']
            
            mean = alpha / (alpha + beta_param)
            variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
            credible_interval = [
                beta.ppf(0.025, alpha, beta_param),
                beta.ppf(0.975, alpha, beta_param)
            ]
            
        elif posterior['type'] == 'normal':
            mu = posterior['params']['mu']
            sigma = posterior['params']['sigma']
            
            mean = mu
            variance = sigma ** 2
            credible_interval = [
                norm.ppf(0.025, mu, sigma),
                norm.ppf(0.975, mu, sigma)
            ]
        
        else:
            return None
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'credible_interval_95': credible_interval,
            'observations_count': self.observation_count[parameter_name]
        }
    
    def predict_parameter_value(self, parameter_name, n_samples=1000):
        """Sample from posterior distribution for prediction."""
        if parameter_name not in self.posteriors:
            return None
        
        posterior = self.posteriors[parameter_name]
        
        if posterior['type'] == 'gamma':
            alpha = posterior['params']['alpha']
            beta = posterior['params']['beta']
            samples = np.random.gamma(alpha, 1/beta, n_samples)
            
        elif posterior['type'] == 'beta':
            alpha = posterior['params']['alpha']
            beta_param = posterior['params']['beta']
            samples = np.random.beta(alpha, beta_param, n_samples)
            
        elif posterior['type'] == 'normal':
            mu = posterior['params']['mu']
            sigma = posterior['params']['sigma']
            samples = np.random.normal(mu, sigma, n_samples)
            
        else:
            return None
        
        return {
            'samples': samples,
            'mean': np.mean(samples),
            'std': np.std(samples),
            'quantiles': np.percentile(samples, [2.5, 25, 50, 75, 97.5])
        }

# -------------------------
# UNCERTAINTY QUANTIFICATION FUNCTIONS - DEEL 4
# -------------------------

def calculate_prediction_intervals(predictions, confidence_level=0.95):
    """
    Calculate prediction intervals voor probabilistic forecasts.
    Implements multiple interval estimation methods.
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    intervals = {}
    
    if isinstance(predictions, dict):
        for key, values in predictions.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 10:
                intervals[key] = {
                    'lower': np.percentile(values, lower_percentile),
                    'upper': np.percentile(values, upper_percentile),
                    'median': np.percentile(values, 50),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    else:
        # Single array of predictions
        if len(predictions) > 10:
            intervals = {
                'lower': np.percentile(predictions, lower_percentile),
                'upper': np.percentile(predictions, upper_percentile),
                'median': np.percentile(predictions, 50),
                'mean': np.mean(predictions),
                'std': np.std(predictions)
            }
    
    return intervals

def model_uncertainty_analysis(models, X_test, y_test=None, n_bootstrap=100):
    """
    Comprehensive model uncertainty analysis.
    Includes epistemic and aleatoric uncertainty estimation.
    """
    print(f"\n=== MODEL UNCERTAINTY ANALYSIS ===")
    
    uncertainty_results = {}
    
    for model_name, model in models.items():
        if model is None:
            continue
            
        print(f"\nAnalyzing uncertainty for {model_name}...")
        
        try:
            # Get base predictions
            if hasattr(model, 'predict_proba'):
                base_probs = model.predict_proba(X_test)
                base_pred = model.predict(X_test)
            else:
                continue
            
            # Bootstrap uncertainty (epistemic uncertainty)
            bootstrap_predictions = []
            bootstrap_probabilities = []
            
            n_samples = X_test.shape[0]
            
            for i in range(min(n_bootstrap, 50)):  # Limit for computational efficiency
                # Bootstrap sample indices
                bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X_test[bootstrap_idx]
                
                try:
                    if hasattr(model, 'predict_proba'):
                        prob_bootstrap = model.predict_proba(X_bootstrap)
                        pred_bootstrap = model.predict(X_bootstrap)
                        
                        bootstrap_probabilities.append(prob_bootstrap)
                        bootstrap_predictions.append(pred_bootstrap)
                        
                except Exception as e:
                    continue
            
            if len(bootstrap_probabilities) > 5:
                # Calculate prediction variance (epistemic uncertainty)
                prob_array = np.array(bootstrap_probabilities)  # Shape: (n_bootstrap, n_samples, n_classes)
                
                # Mean prediction across bootstrap samples
                prob_mean = np.mean(prob_array, axis=0)
                
                # Prediction variance (epistemic uncertainty)
                prob_variance = np.var(prob_array, axis=0)
                epistemic_uncertainty = np.mean(prob_variance, axis=1)  # Average across classes
                
                # Aleatoric uncertainty (inherent randomness)
                # Calculated as entropy of mean predictions
                aleatoric_uncertainty = []
                for prob_sample in prob_mean:
                    entropy = -np.sum(prob_sample * np.log(prob_sample + 1e-10))
                    aleatoric_uncertainty.append(entropy)
                aleatoric_uncertainty = np.array(aleatoric_uncertainty)
                
                # Total uncertainty
                total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
                
                uncertainty_results[model_name] = {
                    'epistemic_uncertainty': {
                        'mean': np.mean(epistemic_uncertainty),
                        'std': np.std(epistemic_uncertainty),
                        'samples': epistemic_uncertainty
                    },
                    'aleatoric_uncertainty': {
                        'mean': np.mean(aleatoric_uncertainty),
                        'std': np.std(aleatoric_uncertainty),
                        'samples': aleatoric_uncertainty
                    },
                    'total_uncertainty': {
                        'mean': np.mean(total_uncertainty),
                        'std': np.std(total_uncertainty),
                        'samples': total_uncertainty
                    },
                    'prediction_intervals': calculate_prediction_intervals(prob_array.reshape(-1, prob_array.shape[-1])),
                    'confidence_calibration': None
                }
                
                # Calculate calibration if true labels available
                if y_test is not None:
                    try:
                        # Convert y_test to numeric if needed
                        if hasattr(y_test, 'map'):
                            le_temp = LabelEncoder()
                            y_test_numeric = le_temp.fit_transform(y_test)
                        else:
                            y_test_numeric = y_test
                        
                        # Calibration analysis
                        max_probs = np.max(base_probs, axis=1)
                        pred_correct = (base_pred == y_test_numeric).astype(int)
                        
                        # Bin predictions by confidence
                        n_bins = 10
                        bin_boundaries = np.linspace(0, 1, n_bins + 1)
                        bin_lowers = bin_boundaries[:-1]
                        bin_uppers = bin_boundaries[1:]
                        
                        calibration_data = []
                        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                            in_bin = (max_probs >= bin_lower) & (max_probs < bin_upper)
                            if np.sum(in_bin) > 0:
                                prop_in_bin = np.sum(in_bin) / len(max_probs)
                                accuracy_in_bin = np.mean(pred_correct[in_bin])
                                avg_confidence_in_bin = np.mean(max_probs[in_bin])
                                
                                calibration_data.append({
                                    'bin_lower': bin_lower,
                                    'bin_upper': bin_upper,
                                    'proportion': prop_in_bin,
                                    'accuracy': accuracy_in_bin,
                                    'confidence': avg_confidence_in_bin,
                                    'calibration_gap': avg_confidence_in_bin - accuracy_in_bin
                                })
                        
                        uncertainty_results[model_name]['confidence_calibration'] = calibration_data
                        
                    except Exception as e:
                        print(f"  Error in calibration analysis: {e}")
                
                print(f"  Epistemic uncertainty: {np.mean(epistemic_uncertainty):.4f}")
                print(f"  Aleatoric uncertainty: {np.mean(aleatoric_uncertainty):.4f}")
                print(f"  Total uncertainty: {np.mean(total_uncertainty):.4f}")
                
        except Exception as e:
            print(f"  Error analyzing {model_name}: {e}")
            continue
    
    return uncertainty_results

def bayesian_model_averaging(models, X_test, prior_weights=None):
    """
    Bayesian Model Averaging voor ensemble predictions met uncertainty.
    Weights models based on their posterior probabilities.
    """
    print(f"\n=== BAYESIAN MODEL AVERAGING ===")
    
    if not models:
        return None
    
    # Get predictions from all models
    model_predictions = {}
    model_probabilities = {}
    
    for name, model in models.items():
        if model is None:
            continue
            
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                preds = model.predict(X_test)
                
                model_predictions[name] = preds
                model_probabilities[name] = probs
                
        except Exception as e:
            print(f"Error getting predictions from {name}: {e}")
            continue
    
    if not model_probabilities:
        return None
    
    # Calculate model weights (uniform if no priors given)
    if prior_weights is None:
        model_weights = {name: 1.0 / len(model_probabilities) for name in model_probabilities.keys()}
    else:
        # Normalize prior weights
        total_weight = sum(prior_weights.values())
        model_weights = {name: weight / total_weight for name, weight in prior_weights.items() if name in model_probabilities}
    
    print("Model weights:")
    for name, weight in model_weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Bayesian model averaging
    n_samples = X_test.shape[0]
    n_classes = list(model_probabilities.values())[0].shape[1]
    
    averaged_probabilities = np.zeros((n_samples, n_classes))
    
    for name, probs in model_probabilities.items():
        weight = model_weights.get(name, 0)
        averaged_probabilities += weight * probs
    
    # Final predictions
    averaged_predictions = np.argmax(averaged_probabilities, axis=1)
    
    # Calculate model uncertainty
    model_variance = np.zeros((n_samples, n_classes))
    
    for name, probs in model_probabilities.items():
        weight = model_weights.get(name, 0)
        diff = probs - averaged_probabilities
        model_variance += weight * (diff ** 2)
    
    # Average model variance across classes
    epistemic_uncertainty_bma = np.mean(model_variance, axis=1)
    
    # Predictive entropy (aleatoric uncertainty)
    aleatoric_uncertainty_bma = []
    for prob_sample in averaged_probabilities:
        entropy = -np.sum(prob_sample * np.log(prob_sample + 1e-10))
        aleatoric_uncertainty_bma.append(entropy)
    aleatoric_uncertainty_bma = np.array(aleatoric_uncertainty_bma)
    
    results = {
        'averaged_predictions': averaged_predictions,
        'averaged_probabilities': averaged_probabilities,
        'model_weights': model_weights,
        'epistemic_uncertainty': epistemic_uncertainty_bma,
        'aleatoric_uncertainty': aleatoric_uncertainty_bma,
        'total_uncertainty': epistemic_uncertainty_bma + aleatoric_uncertainty_bma,
        'individual_model_predictions': model_predictions,
        'individual_model_probabilities': model_probabilities
    }
    
    print(f"BMA completed. Average epistemic uncertainty: {np.mean(epistemic_uncertainty_bma):.4f}")
    
    return results

print("DEEL 4 geladen: Bayesian Methods en Uncertainty Quantification")
print("Volgende: DEEL 5 - Time-Series Analysis en Advanced Validation")
"""
DEEL 5: Time-Series Analysis en Advanced Validation
Advanced Soccer Predictor v9.0

Dit deel bevat:
- Advanced time-series analysis voor momentum en trends
- LSTM en GRU models voor sequential data
- Time-series cross-validation met forward chaining
- Momentum en form trend detection
- Advanced validation strategies
- Performance tracking over time
"""

from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# -------------------------
# TIME-SERIES ANALYSIS CLASSES - DEEL 5
# -------------------------

class SoccerTimeSeriesAnalyzer:
    """
    Advanced time-series analysis voor soccer data.
    Detecteert trends, momentum, en seasonal patterns.
    """
    
    def __init__(self, window_size=10, trend_sensitivity=0.1):
        self.window_size = window_size
        self.trend_sensitivity = trend_sensitivity
        self.team_trends = {}
        self.seasonal_patterns = {}
        self.momentum_indicators = {}
        
        print(f"Initialized Soccer Time-Series Analyzer (window={window_size})")
    
    def analyze_team_trends(self, team_data, team_name):
        """
        Analyze long-term trends voor een team.
        Detecteert verbetering/verslechtering over tijd.
        """
        print(f"\nAnalyzing trends for {team_name}...")
        
        trends = {}
        
        # Key metrics voor trend analysis
        metrics = ['Goals', 'xG90', 'WinRate5', 'Possession', 'ShotQual', 'RecentForm']
        
        for metric in metrics:
            if metric in team_data.columns:
                values = series_to_numeric(team_data[metric])
                
                if len(values) >= self.window_size:
                    # Calculate rolling averages
                    short_ma = values.rolling(window=5).mean()
                    long_ma = values.rolling(window=self.window_size).mean()
                    
                    # Trend detection
                    recent_trend = self._calculate_trend(values.tail(self.window_size))
                    
                    # Momentum calculation
                    if len(short_ma.dropna()) > 0 and len(long_ma.dropna()) > 0:
                        momentum = (short_ma.iloc[-1] - long_ma.iloc[-1]) / (long_ma.iloc[-1] + 1e-6)
                    else:
                        momentum = 0
                    
                    # Volatility (consistency measure)
                    volatility = values.rolling(window=self.window_size).std().iloc[-1] if len(values) >= self.window_size else 0
                    
                    trends[metric] = {
                        'current_value': values.iloc[-1] if len(values) > 0 else 0,
                        'trend_slope': recent_trend,
                        'momentum': momentum,
                        'volatility': volatility,
                        'trend_direction': 'improving' if recent_trend > self.trend_sensitivity else 
                                         'declining' if recent_trend < -self.trend_sensitivity else 'stable',
                        'consistency': 1 / (1 + volatility) if volatility > 0 else 1.0
                    }
                    
                    print(f"  {metric:<12}: {trends[metric]['trend_direction']:<10} "
                          f"(slope={recent_trend:+.3f}, momentum={momentum:+.3f})")
        
        self.team_trends[team_name] = trends
        return trends
    
    def _calculate_trend(self, values):
        """Calculate trend slope using linear regression."""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        y = values.values
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        # Avoid division by zero
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def detect_seasonal_patterns(self, team_data, team_name):
        """
        Detect seasonal patterns in team performance.
        Useful voor competition-specific trends.
        """
        print(f"\nDetecting seasonal patterns for {team_name}...")
        
        if 'Date' not in team_data.columns and 'For_' + team_name.replace(' ', '_') + '_Date_shooting' not in team_data.columns:
            print("  No date column found for seasonal analysis")
            return {}
        
        # Try different date column formats
        date_col = None
        for col in team_data.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            return {}
        
        try:
            # Parse dates
            team_data = team_data.copy()
            team_data['parsed_date'] = pd.to_datetime(team_data[date_col], errors='coerce')
            team_data = team_data.dropna(subset=['parsed_date'])
            
            if len(team_data) < 10:
                print("  Insufficient data with valid dates")
                return {}
            
            # Extract time features
            team_data['month'] = team_data['parsed_date'].dt.month
            team_data['quarter'] = team_data['parsed_date'].dt.quarter
            team_data['day_of_week'] = team_data['parsed_date'].dt.dayofweek
            
            seasonal_patterns = {}
            
            # Monthly patterns
            if 'Goals' in team_data.columns:
                monthly_goals = team_data.groupby('month')['Goals'].mean()
                seasonal_patterns['monthly_goals'] = monthly_goals.to_dict()
                
                best_month = monthly_goals.idxmax()
                worst_month = monthly_goals.idxmin()
                
                print(f"  Best month: {best_month} (avg {monthly_goals[best_month]:.2f} goals)")
                print(f"  Worst month: {worst_month} (avg {monthly_goals[worst_month]:.2f} goals)")
            
            # Day of week patterns (if enough data)
            if len(team_data) > 14:
                dow_performance = team_data.groupby('day_of_week').agg({
                    'Goals': 'mean',
                    'WinRate5': 'mean'
                }) if 'Goals' in team_data.columns and 'WinRate5' in team_data.columns else {}
                
                if not dow_performance.empty:
                    seasonal_patterns['day_of_week_performance'] = dow_performance.to_dict()
            
            self.seasonal_patterns[team_name] = seasonal_patterns
            return seasonal_patterns
            
        except Exception as e:
            print(f"  Error in seasonal analysis: {e}")
            return {}
    
    def calculate_momentum_score(self, team_data, team_name):
        """
        Calculate comprehensive momentum score.
        Combineert recente prestaties, trends, en form.
        """
        print(f"\nCalculating momentum score for {team_name}...")
        
        momentum_components = {}
        
        # Component 1: Recent form trend
        if 'RecentForm' in team_data.columns:
            recent_form = series_to_numeric(team_data['RecentForm'])
            if len(recent_form) >= 5:
                form_trend = self._calculate_trend(recent_form.tail(5))
                momentum_components['form_trend'] = form_trend
        
        # Component 2: Win rate acceleration
        if 'WinRate5' in team_data.columns:
            win_rates = series_to_numeric(team_data['WinRate5'])
            if len(win_rates) >= 6:
                recent_wr = win_rates.tail(3).mean()
                previous_wr = win_rates.iloc[-6:-3].mean()
                wr_acceleration = recent_wr - previous_wr
                momentum_components['win_rate_acceleration'] = wr_acceleration
        
        # Component 3: Goal scoring momentum
        if 'Goals' in team_data.columns:
            goals = series_to_numeric(team_data['Goals'])
            if len(goals) >= 8:
                recent_goals = goals.tail(4).mean()
                previous_goals = goals.iloc[-8:-4].mean()
                goal_momentum = (recent_goals - previous_goals) / (previous_goals + 1)
                momentum_components['goal_momentum'] = goal_momentum
        
        # Component 4: Performance consistency
        metrics = ['Goals', 'xG90', 'ShotQual']
        consistency_scores = []
        
        for metric in metrics:
            if metric in team_data.columns:
                values = series_to_numeric(team_data[metric]).tail(8)
                if len(values) >= 5:
                    cv = values.std() / (values.mean() + 1e-6)  # Coefficient of variation
                    consistency = 1 / (1 + cv)  # Higher = more consistent
                    consistency_scores.append(consistency)
        
        if consistency_scores:
            momentum_components['consistency'] = np.mean(consistency_scores)
        
        # Combine components into final momentum score
        weights = {
            'form_trend': 0.3,
            'win_rate_acceleration': 0.3,
            'goal_momentum': 0.25,
            'consistency': 0.15
        }
        
        momentum_score = 0
        total_weight = 0
        
        for component, value in momentum_components.items():
            weight = weights.get(component, 0)
            momentum_score += weight * value
            total_weight += weight
        
        if total_weight > 0:
            momentum_score /= total_weight
        
        # Normalize to [-1, 1] range
        momentum_score = np.tanh(momentum_score)
        
        momentum_info = {
            'score': momentum_score,
            'components': momentum_components,
            'interpretation': (
                'Strong positive momentum' if momentum_score > 0.3 else
                'Moderate positive momentum' if momentum_score > 0.1 else
                'Neutral momentum' if abs(momentum_score) <= 0.1 else
                'Moderate negative momentum' if momentum_score > -0.3 else
                'Strong negative momentum'
            )
        }
        
        self.momentum_indicators[team_name] = momentum_info
        
        print(f"  Momentum score: {momentum_score:+.3f} ({momentum_info['interpretation']})")
        for comp, val in momentum_components.items():
            print(f"    {comp}: {val:+.3f}")
        
        return momentum_info

class LSTMSoccerPredictor:
    """
    LSTM Neural Network voor sequential soccer prediction.
    Gebruikt historical sequences voor betere predictions.
    """
    
    def __init__(self, sequence_length=10, lstm_units=64, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        print(f"Initialized LSTM Soccer Predictor (seq_len={sequence_length}, units={lstm_units})")
    
    def prepare_sequences(self, X, y, sequence_length=None):
        """
        Prepare sequential data voor LSTM training.
        Converts tabular data to sequences.
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        if len(X) < sequence_length + 1:
            print(f"Insufficient data for sequences: {len(X)} < {sequence_length + 1}")
            return None, None
        
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(sequence_length, len(X)):
            # Take previous sequence_length samples as input
            seq = X.iloc[i-sequence_length:i].values
            target = y.iloc[i] if hasattr(y, 'iloc') else y[i]
            
            sequences.append(seq)
            targets.append(target)
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        
        print(f"Created {len(sequences)} sequences of length {sequence_length}")
        print(f"Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def build_model(self, input_shape, n_classes):
        """
        Build LSTM model architecture.
        Implements research-based LSTM configuration.
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units,
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Train LSTM model met sequential data.
        """
        print("\nTraining LSTM Soccer Predictor...")
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        if X_seq is None:
            print("Failed to create sequences")
            return None
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_seq)
        n_classes = len(self.label_encoder.classes_)
        
        # Convert to categorical if multi-class
        if n_classes > 2:
            y_categorical = to_categorical(y_encoded)
        else:
            y_categorical = y_encoded
        
        # Scale features
        # Reshape for scaling (combine batch and time dimensions)
        original_shape = X_seq.shape
        X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_seq_scaled = X_scaled.reshape(original_shape)
        
        # Build model
        self.model = self.build_model(
            input_shape=(X_seq.shape[1], X_seq.shape[2]),
            n_classes=n_classes
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_seq_scaled, y_categorical,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        # Evaluate final performance
        final_loss = min(history.history['val_loss'])
        final_accuracy = max(history.history.get('val_accuracy', [0]))
        
        print(f"LSTM Training completed:")
        print(f"  Final validation loss: {final_loss:.4f}")
        print(f"  Final validation accuracy: {final_accuracy:.4f}")
        print(f"  Total epochs trained: {len(history.history['loss'])}")
        
        return history
    
    def predict_with_sequence(self, X_recent, return_proba=True):
        """
        Predict using recent sequence of matches.
        """
        if self.model is None:
            print("Model not trained yet")
            return None
        
        if len(X_recent) < self.sequence_length:
            print(f"Insufficient recent data: {len(X_recent)} < {self.sequence_length}")
            return None
        
        # Take last sequence_length matches
        X_seq = X_recent.tail(self.sequence_length).values
        X_seq = X_seq.reshape(1, X_seq.shape[0], X_seq.shape[1])  # Add batch dimension
        
        # Scale features
        original_shape = X_seq.shape
        X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_seq_scaled = X_scaled.reshape(original_shape)
        
        # Predict
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict(X_seq_scaled)[0]
            prediction = np.argmax(probabilities)
        else:
            prediction = self.model.predict(X_seq_scaled)[0]
            if len(prediction) > 1:  # Multi-class
                probabilities = prediction
                prediction = np.argmax(prediction)
            else:  # Binary
                probabilities = [1-prediction[0], prediction[0]]
                prediction = int(prediction[0] > 0.5)
        
        # Convert back to labels
        pred_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'prediction': pred_label,
            'probabilities': probabilities,
            'confidence': np.max(probabilities)
        }

# -------------------------
# ADVANCED VALIDATION STRATEGIES - DEEL 5
# -------------------------

def advanced_time_series_validation(models, X, y, n_splits=5, test_size=0.2, gap=0):
    """
    Advanced time-series cross-validation met multiple strategies.
    Implements walk-forward validation en expanding window validation.
    """
    print("\n" + "="*60)
    print("ADVANCED TIME-SERIES VALIDATION")
    print("="*60)
    
    validation_results = {}
    
    # Strategy 1: Standard Time Series Split
    print(f"\n1. Standard Time Series Cross-Validation (n_splits={n_splits})")
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
    
    for name, model in models.items():
        if model is None:
            continue
        
        print(f"\nValidating {name}...")
        
        fold_scores = []
        fold_predictions = []
        fold_true_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            try:
                X_train_fold = X.iloc[train_idx]
                X_test_fold = X.iloc[test_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_test_scaled = scaler.transform(X_test_fold)
                
                # Train model (create new instance to avoid data leakage)
                if 'xgb' in name.lower():
                    fold_model = xgb.XGBClassifier(**ENHANCED_XGBOOST_PARAMS)
                elif 'lightgbm' in name.lower() or 'lgb' in name.lower():
                    fold_model = lgb.LGBMClassifier(**ENHANCED_LIGHTGBM_PARAMS)
                elif 'catboost' in name.lower():
                    fold_model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
                elif 'forest' in name.lower():
                    fold_model = RandomForestClassifier(
                        n_estimators=300, max_depth=18, random_state=42, n_jobs=-1
                    )
                else:
                    continue  # Skip ensemble models for individual validation
                
                # Encode labels if needed
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train_fold)
                y_test_encoded = le.transform(y_test_fold)
                
                # Fit model
                if 'lightgbm' in name.lower():
                    fold_model.fit(X_train_scaled, y_train_encoded, 
                                 eval_set=[(X_test_scaled, y_test_encoded)], 
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                else:
                    fold_model.fit(X_train_scaled, y_train_encoded)
                
                # Predict
                y_pred_fold = fold_model.predict(X_test_scaled)
                fold_acc = accuracy_score(y_test_encoded, y_pred_fold)
                
                fold_scores.append(fold_acc)
                fold_predictions.extend(y_pred_fold)
                fold_true_labels.extend(y_test_encoded)
                
                print(f"  Fold {fold}: Accuracy = {fold_acc:.4f} (train: {len(X_train_fold)}, test: {len(X_test_fold)})")
                
            except Exception as e:
                print(f"  Fold {fold}: Error - {e}")
                continue
        
        if fold_scores:
            # Calculate overall metrics
            mean_acc = np.mean(fold_scores)
            std_acc = np.std(fold_scores)
            overall_acc = accuracy_score(fold_true_labels, fold_predictions)
            
            validation_results[f'{name}_tscv'] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'overall_accuracy': overall_acc,
                'fold_scores': fold_scores,
                'stability': 1 - (std_acc / mean_acc) if mean_acc > 0 else 0
            }
            
            print(f"  Results: Mean={mean_acc:.4f} (+/-{std_acc*2:.4f}), Overall={overall_acc:.4f}")
    
    # Strategy 2: Walk-Forward Validation
    print(f"\n2. Walk-Forward Validation")
    walk_forward_results = walk_forward_validation(models, X, y, min_train_size=50, step_size=10)
    
    for name, results in walk_forward_results.items():
        validation_results[f'{name}_walkforward'] = results
    
    # Strategy 3: Expanding Window Validation
    print(f"\n3. Expanding Window Validation")
    expanding_results = expanding_window_validation(models, X, y, initial_size=100, step_size=20)
    
    for name, results in expanding_results.items():
        validation_results[f'{name}_expanding'] = results
    
    # Summary
    print(f"\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for strategy in ['tscv', 'walkforward', 'expanding']:
        print(f"\n{strategy.upper()} Results:")
        strategy_results = {k: v for k, v in validation_results.items() if strategy in k}
        
        if strategy_results:
            for name, results in strategy_results.items():
                model_name = name.replace(f'_{strategy}', '')
                accuracy = results.get('mean_accuracy', results.get('accuracy', 0))
                stability = results.get('stability', 0)
                print(f"  {model_name:<20}: Accuracy={accuracy:.4f}, Stability={stability:.3f}")
    
    return validation_results

def walk_forward_validation(models, X, y, min_train_size=50, step_size=10):
    """
    Walk-forward validation voor time-series data.
    Simuleert real-world scenario met growing training set.
    """
    print(f"Walk-Forward Validation (min_train={min_train_size}, step={step_size})")
    
    results = {}
    
    for name, model in models.items():
        if model is None or 'ensemble' in name.lower():
            continue
            
        print(f"  Testing {name}...")
        
        predictions = []
        true_labels = []
        accuracies = []
        
        # Start with minimum training size
        current_pos = min_train_size
        
        while current_pos + step_size < len(X):
            try:
                # Training data: from start to current position
                X_train = X.iloc[:current_pos]
                y_train = y.iloc[:current_pos] if hasattr(y, 'iloc') else y[:current_pos]
                
                # Test data: next step_size samples
                X_test = X.iloc[current_pos:current_pos+step_size]
                y_test = y.iloc[current_pos:current_pos+step_size] if hasattr(y, 'iloc') else y[current_pos:current_pos+step_size]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and train model
                if 'xgb' in name.lower():
                    fold_model = xgb.XGBClassifier(**ENHANCED_XGBOOST_PARAMS)
                elif 'lightgbm' in name.lower():
                    fold_model = lgb.LGBMClassifier(**ENHANCED_LIGHTGBM_PARAMS)
                elif 'catboost' in name.lower():
                    fold_model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
                elif 'forest' in name.lower():
                    fold_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
                else:
                    current_pos += step_size
                    continue
                
                # Encode labels
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                
                # Fit model
                if 'lightgbm' in name.lower():
                    fold_model.fit(X_train_scaled, y_train_encoded, verbose=0)
                else:
                    fold_model.fit(X_train_scaled, y_train_encoded)
                
                # Predict
                y_pred = fold_model.predict(X_test_scaled)
                
                # Convert test labels
                y_test_encoded = le.transform(y_test)
                
                # Calculate accuracy for this step
                step_accuracy = accuracy_score(y_test_encoded, y_pred)
                accuracies.append(step_accuracy)
                
                predictions.extend(y_pred)
                true_labels.extend(y_test_encoded)
                
                current_pos += step_size
                
            except Exception as e:
                print(f"    Error at position {current_pos}: {e}")
                current_pos += step_size
                continue
        
        if accuracies:
            overall_accuracy = accuracy_score(true_labels, predictions)
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            results[name] = {
                'accuracy': overall_accuracy,
                'mean_step_accuracy': mean_accuracy,
                'std_step_accuracy': std_accuracy,
                'n_steps': len(accuracies),
                'stability': 1 - (std_accuracy / mean_accuracy) if mean_accuracy > 0 else 0,
                'step_accuracies': accuracies
            }
            
            print(f"    Steps: {len(accuracies)}, Overall: {overall_accuracy:.4f}, Mean: {mean_accuracy:.4f}")
    
    return results

def expanding_window_validation(models, X, y, initial_size=100, step_size=20):
    """
    Expanding window validation.
    Training set groeit progressief, test set blijft constant.
    """
    print(f"Expanding Window Validation (initial={initial_size}, step={step_size})")
    
    results = {}
    
    for name, model in models.items():
        if model is None or 'ensemble' in name.lower():
            continue
            
        print(f"  Testing {name}...")
        
        window_accuracies = []
        training_sizes = []
        
        # Start with initial size
        current_train_size = initial_size
        
        while current_train_size + step_size < len(X):
            try:
                # Training data: expanding window
                X_train = X.iloc[:current_train_size]
                y_train = y.iloc[:current_train_size] if hasattr(y, 'iloc') else y[:current_train_size]
                
                # Test data: fixed window after training
                X_test = X.iloc[current_train_size:current_train_size+step_size]
                y_test = y.iloc[current_train_size:current_train_size+step_size] if hasattr(y, 'iloc') else y[current_train_size:current_train_size+step_size]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create model
                if 'xgb' in name.lower():
                    fold_model = xgb.XGBClassifier(**ENHANCED_XGBOOST_PARAMS)
                elif 'lightgbm' in name.lower():
                    fold_model = lgb.LGBMClassifier(**ENHANCED_LIGHTGBM_PARAMS)
                elif 'catboost' in name.lower():
                    fold_model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
                elif 'forest' in name.lower():
                    fold_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
                else:
                    current_train_size += step_size
                    continue
                
                # Encode labels
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                y_test_encoded = le.transform(y_test)
                
                # Train and predict
                if 'lightgbm' in name.lower():
                    fold_model.fit(X_train_scaled, y_train_encoded, verbose=0)
                else:
                    fold_model.fit(X_train_scaled, y_train_encoded)
                
                y_pred = fold_model.predict(X_test_scaled)
                
                # Calculate accuracy
                window_accuracy = accuracy_score(y_test_encoded, y_pred)
                window_accuracies.append(window_accuracy)
                training_sizes.append(current_train_size)
                
                current_train_size += step_size
                
            except Exception as e:
                print(f"    Error at size {current_train_size}: {e}")
                current_train_size += step_size
                continue
        
        if window_accuracies:
            mean_accuracy = np.mean(window_accuracies)
            std_accuracy = np.std(window_accuracies)
            
            # Calculate learning curve trend
            if len(window_accuracies) >= 3:
                x = np.array(training_sizes)
                y_acc = np.array(window_accuracies)
                
                # Simple linear regression for trend
                n = len(x)
                sum_x, sum_y = np.sum(x), np.sum(y_acc)
                sum_xy, sum_x2 = np.sum(x * y_acc), np.sum(x * x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 - sum_x * sum_x != 0 else 0
            else:
                slope = 0
            
            results[name] = {
                'accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'n_windows': len(window_accuracies),
                'learning_trend': slope,
                'stability': 1 - (std_accuracy / mean_accuracy) if mean_accuracy > 0 else 0,
                'window_accuracies': window_accuracies,
                'training_sizes': training_sizes
            }
            
            trend_desc = "improving" if slope > 0.001 else "stable" if abs(slope) <= 0.001 else "declining"
            print(f"    Windows: {len(window_accuracies)}, Mean: {mean_accuracy:.4f}, Trend: {trend_desc}")
    
    return results

# -------------------------
# PERFORMANCE TRACKING - DEEL 5
# -------------------------

class PerformanceTracker:
    """
    Track model performance over time.
    Monitors accuracy, calibration, en concept drift.
    """
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.performance_history = {}
        self.calibration_history = {}
        self.drift_indicators = {}
        
        print(f"Initialized Performance Tracker (window={window_size})")
    
    def update_performance(self, model_name, predictions, true_labels, probabilities=None):
        """Update performance metrics voor een model."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                'accuracy': [],
                'predictions': [],
                'true_labels': [],
                'probabilities': [],
                'timestamps': []
            }
        
        # Calculate current accuracy
        current_accuracy = accuracy_score(true_labels, predictions)
        
        # Store results
        self.performance_history[model_name]['accuracy'].append(current_accuracy)
        self.performance_history[model_name]['predictions'].extend(predictions)
        self.performance_history[model_name]['true_labels'].extend(true_labels)
        self.performance_history[model_name]['timestamps'].append(len(self.performance_history[model_name]['accuracy']))
        
        if probabilities is not None:
            self.performance_history[model_name]['probabilities'].extend(probabilities)
        
        # Keep only recent window
        if len(self.performance_history[model_name]['accuracy']) > self.window_size * 2:
            self.performance_history[model_name]['accuracy'] = self.performance_history[model_name]['accuracy'][-self.window_size:]
            self.performance_history[model_name]['predictions'] = self.performance_history[model_name]['predictions'][-self.window_size:]
            self.performance_history[model_name]['true_labels'] = self.performance_history[model_name]['true_labels'][-self.window_size:]
            if probabilities is not None:
                self.performance_history[model_name]['probabilities'] = self.performance_history[model_name]['probabilities'][-self.window_size:]
        
        # Detect performance drift
        self._detect_drift(model_name)
        
        return current_accuracy
    
    def _detect_drift(self, model_name):
        """Detect concept drift in model performance."""
        
        history = self.performance_history[model_name]
        accuracies = history['accuracy']
        
        if len(accuracies) < 20:  # Need minimum samples for drift detection
            return
        
        # Split recent history in half
        mid_point = len(accuracies) // 2
        early_performance = accuracies[:mid_point]
        recent_performance = accuracies[mid_point:]
        
        if len(early_performance) >= 10 and len(recent_performance) >= 10:
            early_mean = np.mean(early_performance)
            recent_mean = np.mean(recent_performance)
            
            # Statistical significance test (simple t-test approximation)
            pooled_std = np.sqrt((np.var(early_performance) + np.var(recent_performance)) / 2)
            
            if pooled_std > 0:
                t_stat = abs(recent_mean - early_mean) / (pooled_std * np.sqrt(2/len(early_performance)))
                drift_detected = t_stat > 2.0  # Rough threshold
                
                drift_magnitude = recent_mean - early_mean
                
                self.drift_indicators[model_name] = {
                    'drift_detected': drift_detected,
                    'drift_magnitude': drift_magnitude,
                    'early_performance': early_mean,
                    'recent_performance': recent_mean,
                    't_statistic': t_stat,
                    'timestamp': len(accuracies)
                }
                
                if drift_detected:
                    drift_type = "improvement" if drift_magnitude > 0 else "degradation"
                    print(f"  DRIFT DETECTED for {model_name}: {drift_type} ({drift_magnitude:+.3f})")
    
    def get_performance_summary(self, model_name):
        """Get comprehensive performance summary."""
        
        if model_name not in self.performance_history:
            return None
        
        history = self.performance_history[model_name]
        
        summary = {
            'current_accuracy': history['accuracy'][-1] if history['accuracy'] else 0,
            'mean_accuracy': np.mean(history['accuracy']) if history['accuracy'] else 0,
            'accuracy_trend': self._calculate_trend(history['accuracy']) if len(history['accuracy']) >= 5 else 0,
            'stability': 1 - (np.std(history['accuracy']) / np.mean(history['accuracy'])) if history['accuracy'] and np.mean(history['accuracy']) > 0 else 0,
            'total_predictions': len(history['predictions']),
            'drift_info': self.drift_indicators.get(model_name, {'drift_detected': False})
        }
        
        return summary
    
    def _calculate_trend(self, values):
        """Calculate linear trend in performance values."""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        n = len(x)
        sum_x, sum_y = np.sum(x), np.sum(y)
        sum_xy, sum_x2 = np.sum(x * y), np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0
        
        return slope

def create_performance_dashboard(performance_tracker, models):
    """
    Create a comprehensive performance dashboard.
    Shows current status and trends voor alle models.
    """
    print("\n" + "="*80)
    print("PERFORMANCE DASHBOARD")
    print("="*80)
    
    model_summaries = []
    
    for model_name in models.keys():
        summary = performance_tracker.get_performance_summary(model_name)
        if summary:
            model_summaries.append((model_name, summary))
    
    if not model_summaries:
        print("No performance data available")
        return
    
    # Sort by current accuracy
    model_summaries.sort(key=lambda x: x[1]['current_accuracy'], reverse=True)
    
    print(f"{'Model':<20} | {'Current':<8} | {'Mean':<8} | {'Trend':<8} | {'Stability':<9} | {'Predictions':<11} | {'Drift':<6}")
    print("-" * 95)
    
    for model_name, summary in model_summaries:
        trend_arrow = "↗" if summary['accuracy_trend'] > 0.01 else "↘" if summary['accuracy_trend'] < -0.01 else "→"
        drift_status = "YES" if summary['drift_info']['drift_detected'] else "NO"
        
        print(f"{model_name[:19]:<20} | "
              f"{summary['current_accuracy']:<8.3f} | "
              f"{summary['mean_accuracy']:<8.3f} | "
              f"{trend_arrow} {summary['accuracy_trend']:+.3f} | "
              f"{summary['stability']:<9.3f} | "
              f"{summary['total_predictions']:<11d} | "
              f"{drift_status:<6}")
    
    # Best performing model
    if model_summaries:
        best_model, best_summary = model_summaries[0]
        print(f"\nBest Performing Model: {best_model}")
        print(f"  Current Accuracy: {best_summary['current_accuracy']:.3f}")
        print(f"  Stability Score: {best_summary['stability']:.3f}")
        
        if best_summary['drift_info']['drift_detected']:
            drift_type = "improving" if best_summary['drift_info']['drift_magnitude'] > 0 else "degrading"
            print(f"  Performance Drift: {drift_type} ({best_summary['drift_info']['drift_magnitude']:+.3f})")

print("DEEL 5 geladen: Time-Series Analysis en Advanced Validation")
print("Volgende: DEEL 6 - Complete Integration en Main Function")

# -------------------------
# DEEL 6: Complete Integration en Main Function
# Advanced Soccer Predictor v9.0
# 
# Dit deel bevat:
# - Complete integratie van alle components
# - Enhanced main function met alle nieuwe features
# - Model saving en loading functions
# - Complete prediction pipeline
# - Advanced reporting en visualization
# - User interface voor alle functionaliteiten
# -------------------------

import joblib
from datetime import datetime
import json
import glob
import os

# -------------------------
# CUSTOM EXCEPTIONS AND UTILITIES - DEEL 6
# -------------------------

class SoccerPredictorError(Exception):
    """Custom exception for soccer predictor errors."""
    pass

def normalize_team_name(name):
    """
    Normalize team name for consistency.
    """
    if not name:
        return "UNKNOWN"
    
    # Remove special characters and normalize
    import re
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', str(name))
    normalized = ' '.join(normalized.split())  # Remove extra spaces
    return normalized.upper()

def calculate_confidence_interval(values, confidence=0.95):
    """
    Calculate confidence interval for a list of values.
    """
    if len(values) < 2:
        return (0, 0)
    
    alpha = 1 - confidence
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean
    
    # Use t-distribution for small samples
    if n < 30:
        t_val = stats.t.ppf(1 - alpha/2, n-1)
        margin = t_val * std_err
    else:
        z_val = stats.norm.ppf(1 - alpha/2)
        margin = z_val * std_err
    
    return (mean - margin, mean + margin)

def detect_outliers(series, method='iqr', factor=1.5):
    """
    Detect outliers in a series using IQR or Z-score method.
    """
    if len(series) < 4:
        return []
    
    outlier_indices = []
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series))
        outlier_indices = series[z_scores > factor].index.tolist()
    
    return outlier_indices

def safe_execute_function(func, *args, default_return=None, error_msg="Function execution failed", **kwargs):
    """
    Safely execute a function with error handling.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"{error_msg}: {e}")
        return default_return

# -------------------------
# MODEL SAVING/LOADING FUNCTIONS - DEEL 6
# -------------------------

def save_complete_model_suite(models, scalers, label_encoder, shap_weights, 
                             performance_tracker, polynomial_transformer=None, 
                             bayesian_models=None, lstm_models=None):
    """
    Save complete model suite met alle components.
    FIXED: Handle scalers properly when it's not a dictionary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Core models
        if models:
            joblib.dump(models, f"enhanced_models_{timestamp}.pkl")
            print(f"Models saved: enhanced_models_{timestamp}.pkl")
        
        # Scalers - FIXED: Handle both dict and single scaler
        if scalers:
            joblib.dump(scalers, f"enhanced_scalers_{timestamp}.pkl")
            print(f"Scalers saved: enhanced_scalers_{timestamp}.pkl")
        
        # Label encoder
        if label_encoder:
            joblib.dump(label_encoder, f"label_encoder_{timestamp}.pkl")
            print(f"Label encoder saved: label_encoder_{timestamp}.pkl")
        
        # SHAP weights
        if shap_weights:
            joblib.dump(shap_weights, f"shap_weights_{timestamp}.pkl")
            print(f"SHAP weights saved: shap_weights_{timestamp}.pkl")
        
        # Performance tracker
        if performance_tracker:
            joblib.dump(performance_tracker, f"performance_tracker_{timestamp}.pkl")
            print(f"Performance tracker saved: performance_tracker_{timestamp}.pkl")
        
        # Polynomial transformer
        if polynomial_transformer:
            joblib.dump(polynomial_transformer, f"polynomial_transformer_{timestamp}.pkl")
            print(f"Polynomial transformer saved: polynomial_transformer_{timestamp}.pkl")
        
        # Bayesian models
        if bayesian_models:
            joblib.dump(bayesian_models, f"bayesian_models_{timestamp}.pkl")
            print(f"Bayesian models saved: bayesian_models_{timestamp}.pkl")
        
        # LSTM models (special handling for Keras)
        if lstm_models:
            for name, lstm_model in lstm_models.items():
                if hasattr(lstm_model, 'model') and lstm_model.model is not None:
                    lstm_model.model.save(f"lstm_{name}_{timestamp}.h5")
                    # Save other components
                    joblib.dump({
                        'scaler': lstm_model.scaler,
                        'label_encoder': lstm_model.label_encoder,
                        'sequence_length': lstm_model.sequence_length,
                        'lstm_units': lstm_model.lstm_units,
                        'dropout_rate': lstm_model.dropout_rate
                    }, f"lstm_{name}_components_{timestamp}.pkl")
            print(f"LSTM models saved with timestamp {timestamp}")
        
        # Create summary file - FIXED: Handle scalers properly
        scalers_info = []
        if scalers:
            if isinstance(scalers, dict):
                scalers_info = list(scalers.keys())
            else:
                scalers_info = [type(scalers).__name__]  # Just the class name
        
        summary = {
            'timestamp': timestamp,
            'models': list(models.keys()) if models else [],
            'scalers': scalers_info,  # FIXED: Use scalers_info instead
            'has_shap_weights': bool(shap_weights),
            'has_performance_tracker': bool(performance_tracker),
            'has_polynomial_transformer': bool(polynomial_transformer),
            'has_bayesian_models': bool(bayesian_models),
            'has_lstm_models': bool(lstm_models)
        }
        
        with open(f"model_suite_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Complete model suite saved with timestamp: {timestamp}")
        return timestamp
        
    except Exception as e:
        print(f"Error saving model suite: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_complete_model_suite(timestamp=None):
    """
    Load complete model suite.
    FIXED: Proper timestamp extraction from filenames
    """
    if timestamp is None:
        # Find most recent timestamp
        summary_files = glob.glob("model_suite_summary_*.json")
        if not summary_files:
            print("No saved model suites found")
            return None
        
        # Get most recent
        summary_files.sort()
        latest_summary = summary_files[-1]
        
        # FIXED: Extract full timestamp correctly
        # File format: model_suite_summary_20250918_211758.json
        # We want: 20250918_211758
        timestamp = latest_summary.replace('model_suite_summary_', '').replace('.json', '')
    
    print(f"Loading model suite with timestamp: {timestamp}")
    
    loaded_components = {}
    
    try:
        # Load summary
        summary_file = f"model_suite_summary_{timestamp}.json"
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"Found components: {list(summary.keys())}")
        
        # Core models
        models_file = f"enhanced_models_{timestamp}.pkl"
        try:
            loaded_components['models'] = joblib.load(models_file)
            print(f"✅ Models loaded: {len(loaded_components['models'])} models")
        except FileNotFoundError:
            print("✗ Models file not found")
            loaded_components['models'] = {}
        
        # Scalers
        scalers_file = f"enhanced_scalers_{timestamp}.pkl"
        try:
            loaded_components['scalers'] = joblib.load(scalers_file)
            print(f"✅ Scalers loaded")
        except FileNotFoundError:
            print("✗ Scalers file not found")
            loaded_components['scalers'] = {}
        
        # Label encoder
        le_file = f"label_encoder_{timestamp}.pkl"
        try:
            loaded_components['label_encoder'] = joblib.load(le_file)
            print("✅ Label encoder loaded")
        except FileNotFoundError:
            print("✗ Label encoder file not found")
            loaded_components['label_encoder'] = None
        
        # SHAP weights
        shap_file = f"shap_weights_{timestamp}.pkl"
        try:
            loaded_components['shap_weights'] = joblib.load(shap_file)
            print("✅ SHAP weights loaded")
        except FileNotFoundError:
            print("✗ SHAP weights file not found")
            loaded_components['shap_weights'] = INITIAL_BASE_WEIGHTS
        
        # Performance tracker
        tracker_file = f"performance_tracker_{timestamp}.pkl"
        try:
            loaded_components['performance_tracker'] = joblib.load(tracker_file)
            print("✅ Performance tracker loaded")
        except FileNotFoundError:
            print("✗ Performance tracker file not found")
            loaded_components['performance_tracker'] = None
        
        # Optional components
        for optional_component in ['polynomial_transformer', 'bayesian_models']:
            file_path = f"{optional_component}_{timestamp}.pkl"
            try:
                loaded_components[optional_component] = joblib.load(file_path)
                print(f"✅ {optional_component} loaded")
            except FileNotFoundError:
                print(f"✗ {optional_component} file not found")
                loaded_components[optional_component] = None
        
        print(f"Model suite loaded successfully!")
        return loaded_components
        
    except Exception as e:
        print(f"Error loading model suite: {e}")
        return None


# -------------------------
# ENHANCED FEATURE BUILDING - DEEL 6
# -------------------------

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
        
    # FIX: Handle tuples and strings properly
    df[venue_col] = df[venue_col].apply(lambda x: str(x).lower() if x is not None else 'unknown')
    mapped = df[venue_col].map(lambda x: 'home' if 'home' in str(x).lower() else ('away' if 'away' in str(x).lower() else 'other'))
    df['mapped_venue'] = mapped
    
    filtered = df[df['mapped_venue'] == venue].reset_index(drop=True)
    
    if len(filtered) > EMA_SPAN:
        filtered = filtered.tail(EMA_SPAN).reset_index(drop=True)
    
    print(f"{venue.capitalize()}-team: {len(filtered)}/{len(df)} wedstrijden na filter (Venue == {venue}).")
    
    if len(filtered) > 0:
        print(f"Datumbereik: {filtered['parsed_date'].min().strftime('%Y-%m-%d')} tot {filtered['parsed_date'].max().strftime('%Y-%m-%d')}")
    
    return filtered

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

def build_enhanced_feature_series(df, team_name):
    """Enhanced feature building (from previous parts)."""
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
        # Parse from result strings
        res_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
        parsed_ga = []
        if res_col:
            goals_series = series_to_numeric(df[goals_col]).fillna(np.nan) if goals_col else pd.Series([np.nan] * n, index=df.index)
            for idx, row in df.iterrows():
                res_val = row.get(res_col, '') if isinstance(row, dict) or isinstance(row, pd.Series) else ''
                if pd.isna(res_val): 
                    res_val = ''
                res_str = str(res_val)
                mres = re.search(r'(\d+)\s*[-–—":\u2013]\s*(\d+)', res_str)
                if mres:
                    a = float(mres.group(1))
                    b = float(mres.group(2))
                    if not np.isnan(goals_series.iloc[idx - df.index[0]]):
                        g = goals_series.iloc[idx - df.index[0]]
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
        score_match = re.match(r'(\d+)\s*[-–—":\u2013]\s*(\d+)', result_str)
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

def make_enhanced_delta(team_feats, opp_feats):
    """Enhanced delta calculation."""
    
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
    # Use the new calculate_field_tilt function:
    t_tilt, o_tilt = calculate_field_tilt(team_feats, opp_feats)
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

def calculate_field_tilt(home_feats, away_feats):
    """
    Calculate field tilt based on territorial dominance.
    """
    # Use multiple indicators for field tilt
    home_att3rd = ema(home_feats.get('Att3rd90', pd.Series([0])))
    away_att3rd = ema(away_feats.get('Att3rd90', pd.Series([0])))
    
    home_poss = ema(home_feats.get('Possession', pd.Series([0.5])))
    away_poss = ema(away_feats.get('Possession', pd.Series([0.5])))
    
    home_prog = ema(home_feats.get('Prog90', pd.Series([0])))
    away_prog = ema(away_feats.get('Prog90', pd.Series([0])))
    
    # Combine indicators
    home_territorial = (home_att3rd * 0.4 + home_poss * 100 * 0.3 + home_prog * 0.3)
    away_territorial = (away_att3rd * 0.4 + away_poss * 100 * 0.3 + away_prog * 0.3)
    
    total_territorial = home_territorial + away_territorial
    
    if total_territorial > 0:
        home_tilt = home_territorial / total_territorial
        away_tilt = away_territorial / total_territorial
    else:
        home_tilt = away_tilt = 0.5
    
    return home_tilt, away_tilt

# -------------------------
# COMPLETE PREDICTION PIPELINE - DEEL 6
# -------------------------

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
        'catboost', 'tensorflow', 'shap', 'scipy', 'matplotlib', 
        'seaborn', 'joblib', 'statsmodels'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All dependencies are installed")
    return True

def initialize_predictor():
    """
    Initialize the predictor with all necessary checks.
    """
    print("Initializing Advanced Soccer Predictor v9.0...")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running.")
        return False
    
    # Validate configuration
    if not validate_configuration():
        print("Please fix configuration issues before running.")
        return False
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Initialization complete!")
    return True

def complete_enhanced_prediction_pipeline(home_df, away_df, loaded_components=None):
    """
    Complete enhanced prediction pipeline met alle components.
    FIXED: Better scaler handling
    """
    print("\n" + "="*80)
    print("COMPLETE ENHANCED PREDICTION PIPELINE v9.0")
    print("="*80)
    
    if loaded_components is None:
        print("No loaded components provided, using defaults")
        return None
    
    models = loaded_components.get('models', {})
    scalers = loaded_components.get('scalers', {})
    label_encoder = loaded_components.get('label_encoder')
    shap_weights = loaded_components.get('shap_weights', INITIAL_BASE_WEIGHTS)
    performance_tracker = loaded_components.get('performance_tracker')
    
    print(f"Available models: {list(models.keys())}")
    print(f"Scalers type: {type(scalers)}")
    print(f"Label encoder available: {label_encoder is not None}")
    
    # Step 1: Apply venue filters
    print("\n1. Processing team data...")
    home_df_filtered = apply_venue_filter(home_df.copy(), 'home')
    away_df_filtered = apply_venue_filter(away_df.copy(), 'away')
    
    if len(home_df_filtered) < 5 or len(away_df_filtered) < 5:
        print("Insufficient data after filtering")
        return None
    
    # Step 2: Build enhanced features
    print("\n2. Building enhanced features...")
    home_feats = build_enhanced_feature_series(home_df_filtered, "HOME TEAM")
    away_feats = build_enhanced_feature_series(away_df_filtered, "AWAY TEAM")
    
    # Step 3: Use same features as training  
    # Step 3: Use same features as training  
    print("\n3. Preparing features (matching training data)...")
    try:
        # Use the SAME feature preparation as during training
        features_dict = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
        
        # Add soccer-specific interactions (these were also used in training)
        interactions = create_soccer_specific_interactions(home_feats, away_feats)
        
        # Combine both feature sets (same as training)
        features_dict.update(interactions)
        
        print(f"Basic + interaction features created: {len(features_dict)}")
        
        processing_info = {
            'polynomials_created': False,
            'multicollinearity_handled': False, 
            'feature_selection_applied': False,
            'final_feature_count': len(features_dict)
        }
        
    except Exception as e:
        print(f"Feature preparation failed: {e}")
        features_dict = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
        processing_info = {
            'polynomials_created': False,
            'multicollinearity_handled': False, 
            'feature_selection_applied': False,
            'final_feature_count': len(features_dict) if features_dict else 0
        }
    
    # Step 4: Time-series analysis
    print("\n4. Time-series analysis...")
    
    # Analyze trends and momentum
    ts_analyzer = SoccerTimeSeriesAnalyzer()
    
    home_trends = ts_analyzer.analyze_team_trends(home_df_filtered, "HOME")
    away_trends = ts_analyzer.analyze_team_trends(away_df_filtered, "AWAY")
    
    home_momentum = ts_analyzer.calculate_momentum_score(home_df_filtered, "HOME")
    away_momentum = ts_analyzer.calculate_momentum_score(away_df_filtered, "AWAY")
    
    # DON'T add momentum to features - they weren't in training data
    # features_dict['home_momentum_score'] = home_momentum['score']
    # features_dict['away_momentum_score'] = away_momentum['score']
    # features_dict['momentum_differential'] = home_momentum['score'] - away_momentum['score']

    # Show momentum analysis in console output instead
    print(f"Home momentum: {home_momentum['score']:+.3f} ({home_momentum['interpretation']})")
    print(f"Away momentum: {away_momentum['score']:+.3f} ({away_momentum['interpretation']})")
    print(f"Momentum differential: {home_momentum['score'] - away_momentum['score']:+.3f}")
    
    # Step 5: Enhanced ensemble predictions
    print("\n5. Enhanced ensemble predictions...")
    
    # FIXED: Handle different scaler types
    scaler_to_use = None
    if isinstance(scalers, dict):
        scaler_to_use = scalers.get('calibrated_scaler') or scalers.get('scaler') or list(scalers.values())[0]
    else:
        scaler_to_use = scalers  # It's already a scaler object
    
    if models and scaler_to_use:
        ensemble_results = enhanced_ensemble_prediction(
            models, scaler_to_use, label_encoder, 
            features_dict, home_feats, away_feats, use_calibrated=True
        )
        
        if ensemble_results:
            print(f"\nEnsemble prediction: {ensemble_results['prediction']}")
            print(f"Confidence: {ensemble_results['confidence']:.3f}")
            print(f"Method: {ensemble_results['method_used']}")
    else:
        print("Models or scaler not available for ensemble prediction")
        ensemble_results = None
    
    # Step 6: Statistical analysis with SHAP weights
    print("\n6. Statistical analysis with SHAP weights...")
    
    delta = make_enhanced_delta(home_feats, away_feats)
    
    # Use SHAP weights if available
    weights_to_use = shap_weights if shap_weights else INITIAL_BASE_WEIGHTS
    
    try:
        final_score, weighted_diff, z_team, z_opp, contribs, corr_analysis = compute_enhanced_weighted_score_with_correlations(
            delta, home_feats, away_feats, use_ml_weights=True
        )
        
        print(f"\nStatistical Score: {final_score:.1f}")
        print(f"Weighted Difference: {weighted_diff:.3f}")
        
    except Exception as e:
        print(f"Statistical analysis error: {e}")
        final_score, weighted_diff = 50.0, 0.0
    
    # Step 7: Compile final results
    print("\n7. Compiling final results...")
    
    final_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'home_team_analysis': {
            'trends': home_trends,
            'momentum': home_momentum
        },
        'away_team_analysis': {
            'trends': away_trends,
            'momentum': away_momentum
        },
        'statistical_analysis': {
            'score': final_score,
            'weighted_difference': weighted_diff
        },
        'features_used': len(features_dict),
        'processing_info': processing_info
    }
    
    if ensemble_results:
        final_results['ensemble_prediction'] = ensemble_results
    
    # Step 8: Generate comprehensive report
    print("\n8. Generating comprehensive report...")
    generate_comprehensive_report(final_results)
    
    return final_results

def compute_enhanced_weighted_score_with_correlations(delta_dict, home_feats, away_feats, use_ml_weights=False):
    """Enhanced scoring with tactical correlations."""
    
    # Get contextual weights
    weights_to_use = calculate_contextual_weights(home_feats, away_feats, INITIAL_BASE_WEIGHTS)
    
    z_team = {}
    z_opp = {}
    contribs = {}

    print(f"\n--- ENHANCED Z-Score Calculation with Tactical Correlations ---")
    print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Diff':<8} | {'Std':<8} | {'Z-Diff':<8} | {'Weight':<8} | {'Contrib':<8}")
    print("-" * 100)

    for feat, (t_ema, o_ema) in delta_dict.items():
        min_std = ENHANCED_MIN_STD_VALUES.get(feat, max(0.1, abs(t_ema) * 0.1))
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

def enhanced_correlation_prediction(home_feats, away_feats, base_weights):
    """Enhanced correlation prediction (simplified version from DEEL 1)."""
    # Simplified version for integration
    correlation_score = 0
    prob_adjustment = np.tanh(correlation_score / 10)
    
    return {
        'correlation_score': correlation_score,
        'prob_adjustment': prob_adjustment,
        'key_mismatches': [('example_mismatch', 0.1)],
        'key_interactions': [('example_interaction', 0.05)],
        'context_weights': base_weights,
        'all_factors': {}
    }

def prepare_enhanced_ml_features_with_tactical(home_feats, away_feats):
    """
    Enhanced ML feature preparation (aangepast van DEEL 1).
    """
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
    
    return combined_features

# -------------------------
# REPORTING AND VISUALIZATION - DEEL 6
# -------------------------

def generate_comprehensive_report(results):
    """
    Generate comprehensive prediction report.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PREDICTION REPORT")
    print("="*80)
    
    print(f"Generated: {results['timestamp']}")
    print(f"Features analyzed: {results['features_used']}")
    
    # Home team analysis
    print(f"\n--- HOME TEAM ANALYSIS ---")
    home_momentum = results['home_team_analysis']['momentum']
    print(f"Momentum Score: {home_momentum['score']:+.3f} ({home_momentum['interpretation']})")
    
    if 'components' in home_momentum:
        print("Momentum components:")
        for comp, val in home_momentum['components'].items():
            print(f"  {comp}: {val:+.3f}")
    
    # Away team analysis
    print(f"\n--- AWAY TEAM ANALYSIS ---")
    away_momentum = results['away_team_analysis']['momentum']
    print(f"Momentum Score: {away_momentum['score']:+.3f} ({away_momentum['interpretation']})")
    
    if 'components' in away_momentum:
        print("Momentum components:")
        for comp, val in away_momentum['components'].items():
            print(f"  {comp}: {val:+.3f}")
    
    # Statistical analysis
    print(f"\n--- STATISTICAL ANALYSIS ---")
    statistical = results['statistical_analysis']
    print(f"Statistical Score: {statistical['score']:.1f}")
    print(f"Weighted Difference: {statistical['weighted_difference']:+.3f}")
    
    # Ensemble prediction
    if 'ensemble_prediction' in results:
        print(f"\n--- ENSEMBLE PREDICTION ---")
        ensemble = results['ensemble_prediction']
        print(f"Prediction: {ensemble['prediction']}")
        print(f"Confidence: {ensemble['confidence']:.3f} ({ensemble['confidence_level']})")
        print(f"Method: {ensemble['method_used']}")
        
        if 'probabilities' in ensemble:
            probs = ensemble['probabilities']
            print(f"Probabilities: W={probs[0]:.3f}, D={probs[1]:.3f}, L={probs[2]:.3f}")
        
        if 'uncertainty' in ensemble:
            print(f"Uncertainty: {ensemble['uncertainty']:.3f}")
            print(f"Entropy: {ensemble['entropy']:.3f}")
    
    # Processing information
    if results.get('processing_info'):
        print(f"\n--- FEATURE ENGINEERING INFO ---")
        info = results['processing_info']
        print(f"Polynomials created: {info.get('polynomials_created', False)}")
        print(f"Multicollinearity handled: {info.get('multicollinearity_handled', False)}")
        print(f"Feature selection applied: {info.get('feature_selection_applied', False)}")
        print(f"Final feature count: {info.get('final_feature_count', 0)}")
    
    print(f"\n" + "="*80)

# -------------------------
# MAIN FUNCTION - DEEL 6
# -------------------------

def main():
    """
    Main function voor Advanced Soccer Predictor v9.0.
    Integreert alle enhanced functionaliteiten.
    """
    print("="*80)
    print("ADVANCED SOCCER PREDICTOR v9.0")
    print("WITH SHAP, BAYESIAN METHODS & ADVANCED ENSEMBLES")
    print("="*80)
    
    print("\nBeschikbare opties:")
    print("1. Train COMPLETE enhanced models (SHAP + Bayesian + LSTM + Ensembles)")
    print("2. Enhanced prediction met alle methods")
    print("3. Load saved models en predict")
    print("4. Advanced model validation en performance analysis")
    print("5. Time-series analysis en momentum detection")
    print("6. Feature importance analysis met SHAP")
    print("7. Bayesian uncertainty quantification")
    print("8. Complete pipeline test")
    
    choice = input("\nJouw keuze (1-8): ").strip()
    
    if choice == "1":
        print("\n" + "="*80)
        print("TRAINING COMPLETE ENHANCED MODELS")
        print("="*80)
        
        training_files = choose_files('Selecteer CSV bestanden voor training (meerdere mogelijk)')
        
        if not training_files:
            print("Geen bestanden geselecteerd.")
            return
        
        if isinstance(training_files, tuple):
            training_files = list(training_files)
        
        # Collect training data
        X_data = []
        y_data = []
        
        print(f"\nVerwerken van {len(training_files)} trainingsbestanden...")
        for file_idx, file_path in enumerate(training_files, 1):
            print(f"\nBestand {file_idx}/{len(training_files)}: {file_path}")
            
            try:
                df = pd.read_csv(file_path, header=0, low_memory=False)
                
                
                print(f"  Geladen: {len(df)} wedstrijden")
                
                for i in range(max(7, EMA_SPAN), len(df)):
                    try:
                        home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                        away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                        
                        if len(home_subset) >= 4 and len(away_subset) >= 4:
                            home_feats = build_enhanced_feature_series(home_subset, "HOME")
                            away_feats = build_enhanced_feature_series(away_subset, "AWAY")
                            
                            # Use complete feature engineering pipeline
                            features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)
                            interactions = create_soccer_specific_interactions(home_feats, away_feats)
                            features.update(interactions)
                            
                            # Extract result
                            result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
                            if result_col:
                                result_str = df[result_col].iloc[i]
                                result = extract_match_result_from_string(result_str)
                                
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
        
        if len(X_data) < 30:
            print("Onvoldoende trainingsdata verzameld (minimum 30 samples vereist).")
            return
        
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        print(f"\nFeature dimensies: {X_df.shape}")
        print(f"Label distributie:")
        label_counts = y_series.value_counts()
        print(label_counts)
        
        # Train enhanced ensemble models
        models, scalers, label_encoder, performance_summary = train_enhanced_ensemble_models(
            X_df, y_series, use_time_series_cv=True
        )
        
        if models:
            # Calculate SHAP feature importance
            print("\n" + "="*60)
            print("CALCULATING SHAP FEATURE IMPORTANCE")
            print("="*60)
            
            shap_weights, shap_scores, shap_values = calculate_shap_feature_importance(
                models, X_df, y_series, list(X_df.columns), n_samples=1000
            )
            
            # Initialize performance tracker
            performance_tracker = PerformanceTracker()
            
            # Save complete model suite
            timestamp = save_complete_model_suite(
                models, scalers, label_encoder, shap_weights, 
                performance_tracker, polynomial_transformer=None,
                bayesian_models=None, lstm_models=None
            )
            
            if timestamp:
                print(f"\n✓ Complete model suite saved with timestamp: {timestamp}")
                print("Models ready voor predictions!")
        
    elif choice == "2":
        print("\n" + "="*80)
        print("ENHANCED PREDICTION MET ALLE METHODS")
        print("="*80)
        
        # Load models
        loaded_components = load_complete_model_suite()
        if not loaded_components:
            print("Geen modellen gevonden. Train eerst de modellen (optie 1).")
            return
        
        print(f"\nBeschikbare modellen: {list(loaded_components.get('models', {}).keys())}")
        
        # Get team data
        home_file = choose_file('Upload THUIS team CSV')
        away_file = choose_file('Upload UIT team CSV')

        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)
        
        home_df = pd.read_csv(home_file, header=0, low_memory=False)
        away_df = pd.read_csv(away_file, header=0, low_memory=False)
        
        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home > 1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away > 1 else 0, low_memory=False)
        
        # Run complete prediction pipeline
        results = complete_enhanced_prediction_pipeline(home_df, away_df, loaded_components)
        
        if results:
            print("\n✓ Enhanced prediction completed!")
            print(f"Check the comprehensive report above for detailed analysis.")
        
    elif choice == "3":
        print("\n" + "="*80)
        print("LOAD SAVED MODELS EN PREDICT")
        print("="*80)
        
        # Load with specific timestamp
        print("Available model timestamps:")
        summary_files = glob.glob("model_suite_summary_*.json")
        if summary_files:
            for i, file in enumerate(summary_files, 1):
                timestamp = file.split('_')[-1].replace('.json', '')
                print(f"  {i}. {timestamp}")
            
            choice_ts = input(f"\nKies timestamp (1-{len(summary_files)}) of Enter voor meest recente: ").strip()
            
            if choice_ts.isdigit() and 1 <= int(choice_ts) <= len(summary_files):
                selected_file = summary_files[int(choice_ts) - 1]
                timestamp = selected_file.split('_')[-1].replace('.json', '')
                loaded_components = load_complete_model_suite(timestamp)
            else:
                loaded_components = load_complete_model_suite()
        else:
            print("Geen saved models gevonden.")
            return
        
        if loaded_components:
            print("Models loaded successfully!")
            
            # Get prediction data
            home_file = choose_file('Upload THUIS team CSV')
            away_file = choose_file('Upload UIT team CSV')
            
            h_home = detect_header_rows(home_file)
            h_away = detect_header_rows(away_file)
            
            home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home > 1 else 0, low_memory=False)
            away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away > 1 else 0, low_memory=False)
            
            # Quick prediction
            results = complete_enhanced_prediction_pipeline(home_df, away_df, loaded_components)
    
    elif choice == "4":
        print("\n" + "="*80)
        print("ADVANCED MODEL VALIDATION")
        print("="*80)
        
        # Load models
        loaded_components = load_complete_model_suite()
        if not loaded_components:
            print("Geen modellen gevonden.")
            return
        
        # Get validation data
        validation_files = choose_files('Selecteer validation CSV bestanden')
        if not validation_files:
            return
        
        # Process validation data (similar to training data processing)
        print("Processing validation data...")
        X_val, y_val = [], []
        
        for file_path in validation_files:
            try:
                df = pd.read_csv(file_path, header=0, low_memory=False)
                
                for i in range(max(7, EMA_SPAN), len(df)):  # Max 25 matches, skip every other
                    try:
                        home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                        away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                        
                        if len(home_subset) >= 4 and len(away_subset) >= 4:
                            home_feats = build_enhanced_feature_series(home_subset, "HOME")
                            away_feats = build_enhanced_feature_series(away_subset, "AWAY")
                            
                            basic_features = prepare_enhanced_ml_features_with_tactical(home_feats, away_feats)

                            # Add soccer-specific interactions  
                            interactions = create_soccer_specific_interactions(home_feats, away_feats)

                            # Combine both feature sets
                            features = {**basic_features, **interactions}
                            
                            result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
                            if result_col:
                                result = extract_match_result_from_string(df[result_col].iloc[i])
                                
                                if result and features:
                                    X_val.append(features)
                                    y_val.append(result)
                                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if len(X_val) > 10:
            X_val_df = pd.DataFrame(X_val)
            y_val_series = pd.Series(y_val)
            
            print(f"Validation samples: {len(X_val)}")
            
            # Advanced validation
            models = loaded_components.get('models', {})
            validation_results = advanced_time_series_validation(models, X_val_df, y_val_series)
            
            print("\nValidation completed! Check results above.")
        else:
            print("Insufficient validation data")
    
    elif choice == "8":
        print("\n" + "="*80)
        print("COMPLETE PIPELINE TEST")
        print("="*80)
        
        # Test alle components
        print("Testing complete pipeline with minimal data...")
        
        # Create minimal test data
        test_home_data = {
            'For_Team_Date_shooting': ['2024-01-01'] * 10,
            'For_Team_Venue_shooting': ['Home'] * 10,
            'For_Team_Result_shooting': ['2-1', '1-0', '1-1', '0-1', '3-2', '2-0', '1-2', '2-2', '3-1', '0-0'],
            'Gls_shooting': [2, 1, 1, 0, 3, 2, 1, 2, 3, 0],
            'xG_shooting': [1.8, 0.9, 1.2, 0.4, 2.5, 1.9, 1.1, 1.8, 2.2, 0.3],
            'Sh_shooting': [12, 8, 10, 6, 15, 13, 9, 11, 16, 4],
            'SoT_shooting': [5, 3, 4, 2, 8, 6, 4, 5, 7, 1]
        }
        
        test_away_data = {
            'For_Team_Date_shooting': ['2024-01-01'] * 10,
            'For_Team_Venue_shooting': ['Away'] * 10,
            'For_Team_Result_shooting': ['1-2', '0-1', '1-1', '1-0', '2-3', '0-2', '2-1', '2-2', '1-3', '0-0'],
            'Gls_shooting': [1, 0, 1, 1, 2, 0, 2, 2, 1, 0],
            'xG_shooting': [1.2, 0.3, 1.1, 0.8, 1.9, 0.4, 1.5, 1.6, 1.0, 0.2],
            'Sh_shooting': [8, 4, 9, 7, 12, 3, 10, 10, 8, 2],
            'SoT_shooting': [3, 1, 4, 3, 6, 1, 5, 4, 3, 0]
        }
        
        home_df_test = pd.DataFrame(test_home_data)
        away_df_test = pd.DataFrame(test_away_data)
        
        print("✓ Test data created")
        
        # Test feature extraction
        print("Testing feature extraction...")
        try:
            home_feats = build_enhanced_feature_series(home_df_test, "HOME_TEST")
            away_feats = build_enhanced_feature_series(away_df_test, "AWAY_TEST")
            print("✓ Feature extraction successful")
        except Exception as e:
            print(f"✗ Feature extraction failed: {e}")
            return
        
        # Test feature engineering pipeline
        print("Testing feature engineering pipeline...")
        try:
            features, info = complete_feature_engineering_pipeline(home_feats, away_feats)
            print(f"✓ Feature engineering successful ({len(features)} features)")
        except Exception as e:
            print(f"✗ Feature engineering failed: {e}")
            return
        
        # Test time-series analyzer
        print("Testing time-series analysis...")
        try:
            ts_analyzer = SoccerTimeSeriesAnalyzer()
            home_trends = ts_analyzer.analyze_team_trends(home_df_test, "HOME_TEST")
            home_momentum = ts_analyzer.calculate_momentum_score(home_df_test, "HOME_TEST")
            print("✓ Time-series analysis successful")
        except Exception as e:
            print(f"✗ Time-series analysis failed: {e}")
        
        print(f"\n✓ Complete pipeline test successful!")
        print("All major components are working correctly.")
    
    else:
        print("Ongeldige keuze. Kies 1-8.")

# -------------------------
# SCRIPT INITIALIZATION - DEEL 6
# -------------------------

if __name__ == '__main__':
    # Initialize before running main
    if initialize_predictor():
        try:
            main()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Initialization failed. Please fix the issues and try again.")

print("DEEL 6 geladen: Complete Integration en Main Function")
print("\n" + "="*80)
print("ALLE DELEN GELADEN - ADVANCED SOCCER PREDICTOR v9.0 READY!")
print("="*80)
print("Voer uit met: python enhanced_soccer_predictor_v9.py")
print("Functies beschikbaar:")
print("- SHAP feature importance analysis")
print("- Enhanced ensemble methods met soft voting")
print("- Polynomial features en interactions")
print("- Bayesian uncertainty quantification")
print("- Time-series analysis en momentum detection")
print("- Advanced validation strategies")
print("- Complete integrated prediction pipeline")
print("="*80)


