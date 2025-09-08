#!/usr/bin/env python3
"""
ml_soccer_predictor.py
Machine learning model voor voetbalwedstrijdvoorspellingen met dynamische weging.
Gebaseerd op experiment_home_away.py met ML-functionaliteit toegevoegd.
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config
# -------------------------
EMA_SPAN = 7
SCALE_TO_SCORE = 4.0
MODEL_FILE = "soccer_model.pkl"
SCALER_FILE = "scaler.pkl"
WEIGHTS_FILE = "dynamic_weights.pkl"

# Minimum standard deviations
MIN_STD_VALUES = {
    'xG90': 0.3,
    'Sh90': 1.0,
    'SoT90': 0.5,
    'ShotQual': 0.05,
    'ConvRatio90': 0.1,
    'Goals': 0.5,
    'Prog90': 5.0,
    'PrgDist90': 100.0,
    'Att3rd90': 10.0,
    'Possession': 0.05,
    'FieldTilt': 0.05,
    'HighPress': 1.0,
    'AerialMismatch': 5.0,
    'KeeperPSxGdiff': 0.2,
    # Voor nieuwe features
    'TkldPct_possession': 0.05,
    'WonPct_misc': 0.05,
    'Att_3rd_defense': 1.0,
    'SavePct_keeper': 0.05,
}

# Initiele gewichten (worden dynamisch aangepast door ML)
WEIGHTS = {
    'xG90': 1.1,
    'Sh90': 1.6,
    'SoT90': 0.8,
    'ShotQual': 1.5,
    'ConvRatio90': 1.8,
    'Goals': 0.8,
    'Prog90': 0.35,
    'PrgDist90': 0.25,
    'Att3rd90': 0.6,
    'FieldTilt': 0.8,
    'HighPress': 0.95,
    'AerialMismatch': 0.6,
    'Possession': 0.35,
    'KeeperPSxGdiff': -0.44,
    'GoalsAgainst': -2.481,
    # Toegevoegd:
    'TkldPct_possession': 0.4,
    'WonPct_misc': 0.4,
    'Att_3rd_defense': 0.8,
    'SavePct_keeper': 0.2,
}

# Dynamische gewichten (worden bijgewerkt door ML)
ML_WEIGHTS = WEIGHTS.copy()

# -------------------------
# Helpers (behouden uit origineel script)
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
    print(f"⚠️ Kon datum niet parseren: {date_str}")
    return None

# -------------------------
# Filter op Venue en sorteer op datum
# -------------------------
def apply_venue_filter(df, venue='home'):
    date_col = find_column(df, ['date'])
    if date_col is None:
        print("❌ Geen datumkolom gevonden! Kan niet sorteren op datum.")
        return df
    df['parsed_date'] = df[date_col].apply(parse_date)
    original_len = len(df)
    df = df[df['parsed_date'].notna()].copy()
    if len(df) < original_len:
        print(f"⚠️ {original_len - len(df)} wedstrijden verwijderd vanwege ongeldige datums.")
    df.sort_values('parsed_date', ascending=True, inplace=True)
    venue_col = find_column(df, ['venue'])
    if venue_col is None:
        print("❌ Geen Venue-kolom gevonden!")
        return df
    df[venue_col] = df[venue_col].astype(str).str.lower()
    mapped = df[venue_col].map(lambda x: 'home' if 'home' in x else ('away' if 'away' in x else 'other'))
    df['mapped_venue'] = mapped
    filtered = df[df['mapped_venue']==venue].reset_index(drop=True)
    if len(filtered) > EMA_SPAN:
        filtered = filtered.tail(EMA_SPAN).reset_index(drop=True)
    print(f"🔍 {venue.capitalize()}-team: {len(filtered)}/{len(df)} wedstrijden na filter (Venue == {venue}).")
    if len(filtered) > 0:
        print(f"📅 Datumbereik: {filtered['parsed_date'].min().strftime('%Y-%m-%d')} tot {filtered['parsed_date'].max().strftime('%Y-%m-%d')}")
    return filtered

# -------------------------
# Build features per match
# -------------------------
def build_feature_series(df, team_name):
    n = len(df)
    feats = {}
    minutes = pd.Series([90.0]*n)

    # Dynamische kolomzoeker per team
    xg_col = find_column_flexible(df, [['expected_xg_shooting'], ['xg_shooting'], ['xg'], ['npxg_shooting'], ['npxg']])
    sh_col = find_column_flexible(df, [['standard_sh_shooting'], ['sh_shooting'], ['sh'], ['shots']])
    sot_col = find_column_flexible(df, [['standard_sot_shooting'], ['sot_shooting'], ['sot'], ['shots on target']])
    goals_col = find_column_flexible(df, [['standard_gls_shooting'], ['gls_shooting'], ['gf_shooting'], ['gf'], ['goals'], ['gls'], ['for_', 'gf'], ['_gf_']])
    prgp_col = find_column_flexible(df, [['prgp_passing'], ['prgr_possession'], ['prgp'], ['progressive']])
    prgdist_col = find_column_flexible(df, [['total_prgdist_passing'], ['prgdist_possession'], ['prgdist'], ['progressive', 'distance']])
    setpieces_col = find_column_flexible(df, [['standard_fk_shooting'], ['fk_shooting'], ['pass types_ck_passing_types'], ['ck_passing_types'], ['fk_passing_types'], ['corner'], ['free', 'kick'], ['setpiece']])
    att3rd_col = find_column_flexible(df, [['touches_att 3rd_possession'], ['touches_att_3rd_possession'], ['att_3rd_possession'], ['att_3rd'], ['att', '3rd']])
    poss_col = find_column_flexible(df, [['for cf montréal_poss_possession'], ['for_', 'poss_possession'], ['poss_possession'], ['possession'], ['poss']])
    sota_col = find_column_flexible(df, [['performance_sota_keeper'], ['sota_keeper'], ['sota'], ['shots on target against']])
    saves_col = find_column_flexible(df, [['performance_saves_keeper'], ['saves_keeper'], ['saves']])
    psxg_col = find_column_flexible(df, [['performance_psxg_keeper'], ['psxg_keeper'], ['psxg']])
    aerial_win_col = find_column_flexible(df, [['aerial duels_won%_misc'], ['wonpct_misc'], ['won_misc'], ['aerial', 'won'], ['aerial', '%'], ['duel']])
    def3rd_col = find_column_flexible(df, [['tackles_def 3rd_defense'], ['def_3rd_defense'], ['tackles_att 3rd_defense'], ['att_3rd_defense'], ['def_3rd'], ['defensive', '3rd']])
    int_col = find_column_flexible(df, [['performance_int_misc'], ['int_misc'], ['interceptions'], ['int']])

    # Toegevoegde nieuwe features
    tkldpct_col = find_column_flexible(df, [['tkldpct_possession'], ['tkld%']])
    wonpct_col = find_column_flexible(df, [['wonpct_misc'], ['aerial', 'won%'], ['duel', '%']])
    att3rddef_col = find_column_flexible(df, [['att_3rd_defense'], ['att', '3rd', 'defense']])
    savepct_col = find_column_flexible(df, [['savepct_keeper'], ['cmppct_keeper'], ['save', '%']])

    # Basis features
    def per90(col):
        return series_to_numeric(df[col])/minutes*90.0 if col else pd.Series([0.0]*n)
    
    feats['xG90'] = per90(xg_col)
    feats['Sh90'] = per90(sh_col)
    feats['SoT90'] = per90(sot_col)

    sh_safe = series_to_numeric(df[sh_col].replace(0,np.nan)) if sh_col else pd.Series([1.0]*n)
    feats['ShotQual'] = 0.6*(series_to_numeric(df[xg_col])/sh_safe).fillna(0.0) + 0.4*(series_to_numeric(df[sot_col])/sh_safe).fillna(0.0)
    feats['Goals'] = series_to_numeric(df[goals_col]).fillna(0.0) if goals_col else pd.Series([0.0]*n)

    # Goals against
    GA_WEIGHT = 0.25
    ga_col = find_column_flexible(df, [['standard_ga_shooting'], ['ga_shooting'], ['goals_against'], ['against'], ['gaa'], ['_ga_'], ['ga']])
    if ga_col:
        feats['GoalsAgainst'] = series_to_numeric(df[ga_col]).fillna(0.0)
    else:
        res_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
        parsed_ga = []
        if res_col:
            goals_series = series_to_numeric(df[goals_col]).fillna(np.nan) if goals_col else pd.Series([np.nan]*n, index=df.index)
            for idx_row, row in df.iterrows():
                res_val = row.get(res_col, '') if isinstance(row, dict) or isinstance(row, pd.Series) else ''
                if pd.isna(res_val): res_val = ''
                res_str = str(res_val)
                mres = re.search(r'(\d+)\s*[-–:\u2013]\s*(\d+)', res_str)
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
            feats['GoalsAgainst'] = pd.Series([0.0]*n, index=df.index)

    feats['GoalsAgainstWeighted'] = feats['GoalsAgainst'] * GA_WEIGHT
    feats['ConvRatio90'] = (series_to_numeric(df[goals_col])/series_to_numeric(df[sot_col])).fillna(0.0) if goals_col and sot_col else pd.Series([0.0]*n)
    feats['Prog90'] = per90(prgp_col)
    feats['PrgDist90'] = per90(prgdist_col)
    feats['SetPieces90'] = per90(setpieces_col)
    feats['Att3rd90'] = per90(att3rd_col)

    poss_raw = series_to_numeric(df[poss_col]) if poss_col else pd.Series([0.0]*n)
    feats['Possession'] = poss_raw/100.0 if poss_raw.max()>1.5 else poss_raw

    feats['SoTA90'] = per90(sota_col)
    feats['SaveRate'] = (series_to_numeric(df[saves_col])/series_to_numeric(df[sota_col])).fillna(0.0) if saves_col and sota_col else pd.Series([0.0]*n)
    feats['PSxG'] = series_to_numeric(df[psxg_col]).fillna(0.0) if psxg_col else pd.Series([0.0]*n)

    feats['AerialWin%'] = series_to_numeric(df[aerial_win_col]).fillna(0.0) if aerial_win_col else pd.Series([0.0]*n)
    feats['HighPress'] = per90(def3rd_col) if def3rd_col else (per90(int_col) if int_col else pd.Series([0.0]*n))

    # Toegevoegde nieuwe features
    feats['TkldPct_possession'] = series_to_numeric(df[tkldpct_col]).fillna(0.0) if tkldpct_col else pd.Series([0.0]*n)
    feats['WonPct_misc'] = series_to_numeric(df[wonpct_col]).fillna(0.0) if wonpct_col else pd.Series([0.0]*n)
    feats['Att_3rd_defense'] = series_to_numeric(df[att3rddef_col]).fillna(0.0) if att3rddef_col else pd.Series([0.0]*n)
    feats['SavePct_keeper'] = series_to_numeric(df[savepct_col]).fillna(0.0) if savepct_col else pd.Series([0.0]*n)

    return feats

# -------------------------
# EMA helper
# -------------------------
def ema(series, span=EMA_SPAN):
    return pd.Series(series).ewm(span=span, adjust=False).mean().iloc[-1]

# -------------------------
# Delta & matchup
# -------------------------
def make_delta(team_feats, opp_feats):
    delta={}
    keys=[
        'xG90','Sh90','SoT90','ShotQual','ConvRatio90','Goals','GoalsAgainst','Prog90','PrgDist90','SetPieces90','Att3rd90','Possession',
        # Toegevoegd (nieuw):
        'TkldPct_possession', 'WonPct_misc', 'Att_3rd_defense', 'SavePct_keeper'
    ]
    for k in keys:
        t = team_feats.get(k,pd.Series([0.0]))
        o = opp_feats.get(k,pd.Series([0.0]))
        delta[k] = (ema(t), ema(o))
    t_att = team_feats.get('Att3rd90',pd.Series([0.0]))
    o_att = opp_feats.get('Att3rd90',pd.Series([0.0]))
    t_tilt = np.where((t_att+o_att)>0, t_att/(t_att+o_att),0.0)
    o_tilt = np.where((o_att+t_att)>0, o_att/(o_att+t_att),0.0)
    delta['FieldTilt'] = (ema(pd.Series(t_tilt)), ema(pd.Series(o_tilt)))
    delta['HighPress'] = (ema(team_feats.get('HighPress',pd.Series([0.0]))), ema(opp_feats.get('HighPress',pd.Series([0.0]))))
    delta['AerialMismatch'] = (ema(team_feats.get('AerialWin%',pd.Series([0.0]))), ema(opp_feats.get('AerialWin%',pd.Series([0.0]))))
    delta['KeeperPSxGdiff'] = (ema(team_feats.get('PSxG',pd.Series([0.0]))), ema(opp_feats.get('PSxG',pd.Series([0.0]))))
    return delta

def compute_weighted_score(delta_dict, use_ml_weights=False):
    z_team={}
    z_opp={}
    contribs={}

    print(f"\n--- Z-Score Calculation Debug ---")
    print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Diff':<8} | {'Std':<8} | {'Z-Diff':<8}")
    print("-" * 90)

    all_ema_values = []
    for feat, (t_ema, o_ema) in delta_dict.items():
        all_ema_values.extend([t_ema, o_ema])
    global_mean = np.mean(all_ema_values)
    global_std = np.std(all_ema_values)

    # Gebruik ML gewichten als gevraagd
    weights_to_use = ML_WEIGHTS if use_ml_weights else WEIGHTS

    for feat, (t_ema, o_ema) in delta_dict.items():
        min_std = MIN_STD_VALUES.get(feat, max(0.1, abs(t_ema) * 0.1))
        combined = np.array([t_ema, o_ema])
        mean, std = np.nanmean(combined), np.nanstd(combined)
        robust_std = max(std, min_std, abs(mean) * 0.1)
        if robust_std < 0.01:
            robust_std = max(0.1, abs(global_mean) * 0.1)
        zt = (t_ema - mean) / robust_std
        zo = (o_ema - mean) / robust_std
        zt = np.clip(zt, -3.0, 3.0)
        zo = np.clip(zo, -3.0, 3.0)
        z_team[feat] = zt
        z_opp[feat] = zo
        weight = weights_to_use.get(feat, 0.0)
        contribs[feat] = weight * (zt - zo)
        diff = t_ema - o_ema
        z_diff = zt - zo
        print(f"{feat:<20} | {t_ema:<10.3f} | {o_ema:<10.3f} | {diff:<+8.3f} | {robust_std:<8.3f} | {z_diff:<+8.3f}")

    if 'KeeperPSxGdiff' in contribs:
        contribs['KeeperPSxGdiff'] = -contribs['KeeperPSxGdiff']
    weighted_diff = sum(contribs.values())
    final = 50.0 + SCALE_TO_SCORE * weighted_diff
    return final, weighted_diff, z_team, z_opp, contribs

# -------------------------
# Nieuwe ML-functionaliteit
# -------------------------
def extract_match_result(df):
    """Extraheert het wedstrijdresultaat uit de dataframe."""
    result_col = find_column_flexible(df, [['result'], ['score'], ['result_shooting']])
    
    if result_col is None:
        print("❌ Geen resultaatkolom gevonden!")
        return None
    
    # Haal het resultaat op voor de laatste wedstrijd (meest recente)
    result = df[result_col].iloc[-1] if len(df) > 0 else None
    
    if result is None:
        return None
    
    # Parse het resultaat (bijv. "W", "D", "L" of "3-1")
    result_str = str(result).upper().strip()
    
    if result_str in ['W', 'WIN', 'WON']:
        return 'win'
    elif result_str in ['D', 'DRAW', 'TIE']:
        return 'draw'
    elif result_str in ['L', 'LOSS', 'LOST']:
        return 'loss'
    else:
        # Probeer scoreformaten zoals "2-1"
        score_match = re.match(r'(\d+)\s*[-–:\u2013]\s*(\d+)', result_str)
        if score_match:
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))
            
            # Bepaal of dit een thuis- of uitwedstrijd is
            venue_col = find_column(df, ['venue'])
            if venue_col:
                venue = str(df[venue_col].iloc[-1]).lower() if len(df) > 0 else 'home'
                is_home = 'home' in venue
                
                if (is_home and home_goals > away_goals) or (not is_home and away_goals > home_goals):
                    return 'win'
                elif home_goals == away_goals:
                    return 'draw'
                else:
                    return 'loss'
        
        print(f"⚠️ Kon resultaat niet parseren: {result_str}")
        return None

def prepare_ml_features(home_feats, away_feats):
    """Bereidt features voor voor machine learning."""
    # Bereken EMA voor alle features
    home_ema = {f"home_{k}": ema(v) for k, v in home_feats.items()}
    away_ema = {f"away_{k}": ema(v) for k, v in away_feats.items()}
    
    # Combineer features
    combined_features = {**home_ema, **away_ema}
    
    # Voeg verschil-features toe
    for key in home_feats.keys():
        if key in away_feats:
            home_val = home_ema.get(f"home_{key}", 0)
            away_val = away_ema.get(f"away_{key}", 0)
            combined_features[f"diff_{key}"] = home_val - away_val
            combined_features[f"ratio_{key}"] = home_val / away_val if away_val != 0 else 0
    
    return combined_features

def update_weights_from_model(model, feature_names):
    """
    Update de gewichten based on feature importance from ML model.
    Dit combineert expert knowledge (initiële gewichten) met ML insights.
    """
    global ML_WEIGHTS
    
    if hasattr(model, 'feature_importances_'):
        # Normaliseer de feature importanties
        feature_importance = model.feature_importances_
        max_importance = max(feature_importance)
        if max_importance > 0:
            feature_importance = feature_importance / max_importance
        
        # Update de gewichten based on ML feature importance
        for i, feature_name in enumerate(feature_names):
            # Zoek de basis feature naam (zonder prefix)
            base_feature = None
            for feat in ML_WEIGHTS.keys():
                if f"diff_{feat}" == feature_name or f"ratio_{feat}" == feature_name:
                    base_feature = feat
                    break
                elif feature_name.startswith(f"home_{feat}") or feature_name.startswith(f"away_{feat}"):
                    base_feature = feat
                    break
            
            if base_feature and base_feature in ML_WEIGHTS:
                # Combineer expert knowledge met ML insights
                ML_WEIGHTS[base_feature] = (WEIGHTS[base_feature] + feature_importance[i]) / 2
                
        print("✅ Gewichten succesvol bijgewerkt based on ML model")
        
        # Sla de bijgewerkte gewichten op
        with open(WEIGHTS_FILE, 'wb') as f:
            pickle.dump(ML_WEIGHTS, f)
        print(f"💾 Dynamische gewichten opgeslagen als {WEIGHTS_FILE}")

def train_model(X, y):
    """Traint een machine learning model en update de gewichten."""
    # Split de data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Schaal de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train het model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evalueer het model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Modelnauwkeurigheid: {accuracy:.2f}")
    print("\n📈 Classificatierapport:")
    print(classification_report(y_test, y_pred))
    
    # Update de gewichten based on feature importance
    feature_names = X.columns.tolist()
    update_weights_from_model(model, feature_names)
    
    # Sla model en scaler op
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"💾 Model opgeslagen als {MODEL_FILE}")
    print(f"💾 Scaler opgeslagen als {SCALER_FILE}")
    print("🔄 Gewichten bijgewerkt based on feature importance")
    
    return model, scaler

def load_model():
    """Laadt een getraind model, scaler en gewichten."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
            
        with open(WEIGHTS_FILE, 'rb') as f:
            global ML_WEIGHTS
            ML_WEIGHTS = pickle.load(f)
        
        print("✅ Model, scaler en gewichten succesvol geladen")
        return model, scaler
    except FileNotFoundError:
        print("❌ Geen getraind model gevonden. Train eerst een model.")
        return None, None

def predict_match(model, scaler, home_features, away_features):
    """Voorspelt de uitkomst van een wedstrijd."""
    # Bereken EMA voor alle features
    home_ema = {f"home_{k}": ema(v) for k, v in home_features.items()}
    away_ema = {f"away_{k}": ema(v) for k, v in away_features.items()}
    
    # Combineer features
    combined_features = {**home_ema, **away_ema}
    
    # Voeg verschil-features toe
    for key in home_features.keys():
        if key in away_features:
            home_val = home_ema.get(f"home_{key}", 0)
            away_val = away_ema.get(f"away_{key}", 0)
            combined_features[f"diff_{key}"] = home_val - away_val
            combined_features[f"ratio_{key}"] = home_val / away_val if away_val != 0 else 0
    
    # Maak een DataFrame van de features
    features_df = pd.DataFrame([combined_features])
    
    # Schaal de features
    features_scaled = scaler.transform(features_df)
    
    # Maak een voorspelling
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities

def combined_prediction(ml_prediction, ml_probabilities, statistical_score):
    """
    Combineert ML voorspelling met statistische score voor betere nauwkeurigheid.
    """
    # Converteer statistical score naar kans
    stat_win_prob = max(0, min(1, (statistical_score - 45) / 10)) if statistical_score > 50 else 0
    stat_draw_prob = max(0, min(1, 1 - abs(statistical_score - 50) / 10))
    stat_loss_prob = max(0, min(1, (45 - statistical_score) / 10)) if statistical_score < 50 else 0
    
    # Normaliseer
    total = stat_win_prob + stat_draw_prob + stat_loss_prob
    if total > 0:
        stat_win_prob /= total
        stat_draw_prob /= total
        stat_loss_prob /= total
    
    # Combineer met ML kansen (gewicht: 70% ML, 30% statistisch)
    ml_weight = 0.7
    stat_weight = 0.3
    
    combined_probs = (
        ml_weight * ml_probabilities[0] + stat_weight * stat_win_prob,
        ml_weight * ml_probabilities[1] + stat_weight * stat_draw_prob,
        ml_weight * ml_probabilities[2] + stat_weight * stat_loss_prob
    )
    
    # Bepaal voorspelling based on combined probabilities
    final_prediction = ['win', 'draw', 'loss'][np.argmax(combined_probs)]
    
    return final_prediction, combined_probs

# -------------------------
# Main met ML-functionaliteit
# -------------------------
if __name__=='__main__':
    print("⚽ Machine Learning Voetbalwedstrijdvoorspeller")
    print("=" * 50)
    
    # Vraag de gebruiker wat ze willen doen
    print("\nKies een optie:")
    print("1. Train een nieuw model")
    print("2. Maak een voorspelling met een bestaand model")
    print("3. Alleen traditionele score berekenen")
    choice = input("Jouw keuze (1, 2 of 3): ").strip()
    
    if choice == "1":
        print("\n🎯 Train een nieuw model")
        print("Upload CSV-bestanden met historische wedstrijdgegevens")
        
        # Vraag om meerdere bestanden voor training
        training_files = []
        while True:
            file_path = choose_file('📂 Upload een CSV-bestand voor training (leeg laten om te stoppen)')
            if not file_path:
                break
            training_files.append(file_path)
        
        if not training_files:
            print("❌ Geen bestanden geselecteerd voor training.")
            exit()
        
        # Verzamel features en labels voor training
        X_data = []
        y_data = []
        
        for file_path in training_files:
            print(f"\n📊 Verwerken van {file_path}")
            
            # Detecteer header rows
            h_rows = detect_header_rows(file_path)
            df = pd.read_csv(file_path, header=list(range(h_rows)) if h_rows>1 else 0, low_memory=False)
            
            # Verwerk elke wedstrijd in het bestand
            for i in range(len(df)):
                # Maak subsets voor home en away
                home_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'home')
                away_subset = apply_venue_filter(df.iloc[:i+1].copy(), 'away')
                
                if len(home_subset) > 0 and len(away_subset) > 0:
                    # Bouw features
                    home_feats = build_feature_series(home_subset, "HOME TEAM")
                    away_feats = build_feature_series(away_subset, "AWAY TEAM")
                    
                    # Bereken EMA-features
                    features = prepare_ml_features(home_feats, away_feats)
                    
                    # Extraheer resultaat
                    result = extract_match_result(df.iloc[:i+1])
                    
                    if result:
                        X_data.append(features)
                        y_data.append(result)
        
        if not X_data:
            print("❌ Geen trainingsdata gevonden.")
            exit()
        
        # Converteer naar DataFrame
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        print(f"\n📦 Grootte van de dataset: {X_df.shape}")
        print(f"🏷️  Verdeling van labels:")
        print(y_series.value_counts())
        
        # Train het model
        model, scaler = train_model(X_df, y_series)
        
    elif choice == "2":
        print("\n🔮 Maak een voorspelling met een bestaand model")
        
        # Laad model en scaler
        model, scaler = load_model()
        if model is None:
            exit()
        
        # CSV upload voor voorspelling
        home_file = choose_file('📂 Upload het CSV bestand van het THUIS team')
        away_file = choose_file('📂 Upload het CSV bestand van het UIT team')

        # Detecteer header rows
        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)

        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home>1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away>1 else 0, low_memory=False)

        # Filter op home/away en sorteer op datum
        home_df = apply_venue_filter(home_df, 'home')
        away_df = apply_venue_filter(away_df, 'away')

        # Bouw features
        home_feats = build_feature_series(home_df, "HOME TEAM")
        away_feats = build_feature_series(away_df, "AWAY TEAM")

        # Maak een voorspelling met ML model
        prediction, probabilities = predict_match(model, scaler, home_feats, away_feats)
        
        # Bereken de traditionele statistische score met ML gewichten
        delta = make_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs = compute_weighted_score(delta, use_ml_weights=True)
        
        # Combineer ML voorspelling met statistische score
        final_prediction, combined_probs = combined_prediction(prediction, probabilities, final_score)
        
        print(f"\n🎯 ML Voorspelling: {prediction}")
        print(f"📊 ML Kansen: Win={probabilities[0]:.2f}, Draw={probabilities[1]:.2f}, Loss={probabilities[2]:.2f}")
        print(f"📈 Traditionele score (met ML gewichten): {final_score:.2f}")
        print(f"🎯 Gecombineerde voorspelling: {final_prediction}")
        print(f"📊 Gecombineerde kansen: Win={combined_probs[0]:.2f}, Draw={combined_probs[1]:.2f}, Loss={combined_probs[2]:.2f}")
        
        # Toon de belangrijkste features
        print(f"\n🔍 Belangrijkste features bijdragen:")
        sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, contrib in sorted_contribs[:5]:
            print(f"  {feat}: {contrib:+.3f}")
            
    elif choice == "3":
        print("\n📊 Alleen traditionele score berekenen")
        
        # CSV upload voor traditionele analyse
        home_file = choose_file('📂 Upload het CSV bestand van het THUIS team')
        away_file = choose_file('📂 Upload het CSV bestand van het UIT team')

        # Detecteer header rows
        h_home = detect_header_rows(home_file)
        h_away = detect_header_rows(away_file)

        home_df = pd.read_csv(home_file, header=list(range(h_home)) if h_home>1 else 0, low_memory=False)
        away_df = pd.read_csv(away_file, header=list(range(h_away)) if h_away>1 else 0, low_memory=False)

        # Filter op home/away en sorteer op datum
        home_df = apply_venue_filter(home_df, 'home')
        away_df = apply_venue_filter(away_df, 'away')

        # Bouw features
        home_feats = build_feature_series(home_df, "HOME TEAM")
        away_feats = build_feature_series(away_df, "AWAY TEAM")

        # Delta + weighted score
        delta = make_delta(home_feats, away_feats)
        final_score, weighted_diff, zt, zo, contribs = compute_weighted_score(delta)

        # Enhanced Output with EMA values
        print('\n=== FINAL ANALYSIS ===')
        print('\n--- EMA Values & Feature Contributions ---')
        print(f"{'Feature':<20} | {'Home EMA':<10} | {'Away EMA':<10} | {'Contribution':<12}")
        print("-" * 70)
        
        for k in sorted(contribs.keys()):
            home_ema, away_ema = delta[k]
            print(f"{k:<20} | {home_ema:<10.3f} | {away_ema:<10.3f} | {contribs[k]:<+12.3f}")
        
        print('=' * 70)
        print(f'Weighted difference = {weighted_diff:.3f}')
        print(f'Final score (50 + {SCALE_TO_SCORE}*diff) = {final_score:.3f}')
        print(f"\n⚡ Totale teamsterkte-score: {final_score:.3f} (gewogen verschil: {weighted_diff:.3f})")
        
        # Interpretation
        if final_score > 52:
            print("🏠 HOME TEAM heeft een significante statistieke voorkeur")
        elif final_score < 48:
            print("✈️ AWAY TEAM heeft een significante statistieke voorkeur") 
        else:
            print("⚖️ Teams zijn statistisch gezien ongeveer gelijk")
            
    else:
        print("❌ Ongeldige keuze. Kies 1, 2 of 3.")
