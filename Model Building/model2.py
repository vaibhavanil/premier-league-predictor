import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor, XGBClassifier
import scipy.stats as stats
# ── LOAD DATA ──
df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered_ewma.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.sort_values('Date').reset_index(drop=True)

# ── DERIVED COLUMNS ──
df['total_goals']   = df['FTHG'] + df['FTAG']
df['total_corners'] = df['HC'] + df['AC']
df['total_cards']   = df['HY'] + df['AY'] + df['HR'] + df['AR']
df['btts']          = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
df['over_2_5']      = (df['total_goals'] > 2.5).astype(int)

# ── TRAIN/TEST SPLIT ──
train = df[df['Date'] < '2024-08-01']
test  = df[df['Date'] >= '2024-08-01']

print("Train rows:", len(train))
print("Test rows: ", len(test))

# ── CORNERS — POISSON ──
corner_features = [
    'home_corners_for',
    'home_corners_against',
    'away_corners_for',
    'away_corners_against',
    'post_2024_transition'
]

X_train_c = train[corner_features]
X_test_c  = test[corner_features]

y_train_hc = train['HC']
y_test_hc  = test['HC']
y_train_ac = train['AC']
y_test_ac  = test['AC']
y_train_tc = train['total_corners']
y_test_tc  = test['total_corners']

# Baseline
baseline_c = y_train_tc.mean()
baseline_c_mae = mean_absolute_error(
    y_test_tc, [baseline_c] * len(y_test_tc)
)

# Poisson home corners
poisson_hc = PoissonRegressor(max_iter=300)
poisson_hc.fit(X_train_c, y_train_hc)
preds_hc = poisson_hc.predict(X_test_c)

# Poisson away corners
poisson_ac = PoissonRegressor(max_iter=300)
poisson_ac.fit(X_train_c, y_train_ac)
preds_ac = poisson_ac.predict(X_test_c)

# Combined total corners
preds_tc = preds_hc + preds_ac
corners_mae = mean_absolute_error(y_test_tc, preds_tc)

print("\nCorners — Poisson with EWMA features:")
print("Baseline MAE:", round(baseline_c_mae, 3))
print("Poisson MAE: ", round(corners_mae, 3))
print("Improvement: ", round((baseline_c_mae - corners_mae) / baseline_c_mae * 100, 1), "%")

# ── CORNERS — XGBOOST FOR COMPARISON ──
xgb_c = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
xgb_c.fit(X_train_c, y_train_tc)
preds_xgb_c = xgb_c.predict(X_test_c)
xgb_corners_mae = mean_absolute_error(y_test_tc, preds_xgb_c)

print("\nCorners — XGBoost with EWMA features:")
print("Baseline MAE:", round(baseline_c_mae, 3))
print("XGBoost MAE: ", round(xgb_corners_mae, 3))
print("Improvement: ", round((baseline_c_mae - xgb_corners_mae) / baseline_c_mae * 100, 1), "%")

# ── GOALS — POISSON ──
goal_features = [
    'home_attack',
    'home_defence',
    'away_attack',
    'away_defence',
    'home_sot_for',     
    'away_sot_for',      
    'home_sot_ratio',   
    'away_sot_ratio',    
    'post_2024_transition'
]

X_train_g = train[goal_features]
X_test_g  = test[goal_features]

y_train_goals = train['total_goals']
y_test_goals  = test['total_goals']

baseline_g = y_train_goals.mean()
baseline_g_mae = mean_absolute_error(
    y_test_goals, [baseline_g] * len(y_test_goals)
)

poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(X_train_g, train['FTHG'])
preds_home = poisson_home.predict(X_test_g)

poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(X_train_g, train['FTAG'])
preds_away = poisson_away.predict(X_test_g)

goals_mae = mean_absolute_error(y_test_goals, preds_home + preds_away)

print("\nGoals — Poisson with EWMA features:")
print("Baseline MAE:", round(baseline_g_mae, 3))
print("Poisson MAE: ", round(goals_mae, 3))
print("Improvement: ", round((baseline_g_mae - goals_mae) / baseline_g_mae * 100, 1), "%")

# ── MATCH RESULT — XGBOOST ──
result_features = [
    'home_attack',
    'home_defence',
    'away_attack',
    'away_defence',
    'home_corners_for',
    'away_corners_for',
    'home_form',
    'away_form',
    'post_2024_transition'
]

df['attack_diff'] = df['home_attack'] - df['away_attack']
df['defence_diff'] = df['home_defence'] - df['away_defence']

result_features_full = result_features + ['attack_diff', 'defence_diff']

train = df[df['Date'] < '2024-08-01']
test  = df[df['Date'] >= '2024-08-01']

X_train_r = train[result_features_full]
X_test_r  = test[result_features_full]

result_map = {'H': 0, 'D': 1, 'A': 2}
y_train_r = train['FTR'].map(result_map)
y_test_r  = test['FTR'].map(result_map)

baseline_acc = (y_test_r == 0).mean()

xgb_result = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_result.fit(X_train_r, y_train_r)
preds_r = xgb_result.predict(X_test_r)
result_acc = (preds_r == y_test_r).mean()

print("\nMatch Result — XGBoost with EWMA features:")
print("Baseline accuracy:", round(baseline_acc, 3))
print("XGBoost accuracy: ", round(result_acc, 3))
print("Improvement:      ", round((result_acc - baseline_acc) / baseline_acc * 100, 1), "%")

# Threshold analysis
probs = xgb_result.predict_proba(X_test_r)
results_df = test[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
results_df = results_df.reset_index(drop=True)
results_df['prob_home'] = probs[:, 0].round(3)
results_df['prob_draw'] = probs[:, 1].round(3)
results_df['prob_away'] = probs[:, 2].round(3)
results_df['predicted'] = preds_r
results_df['predicted'] = results_df['predicted'].map({0:'H', 1:'D', 2:'A'})
results_df['correct'] = (results_df['predicted'] == results_df['FTR'])

print("\nThreshold analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    high_conf = results_df[
        (results_df['prob_home'] > threshold) |
        (results_df['prob_away'] > threshold)
    ]
    if len(high_conf) > 0:
        print(f"{threshold:<12} {len(high_conf):<10} {high_conf['correct'].mean():.1%}")

# ── BASELINE CALCULATIONS ──

# Corners baseline
y_test_corners = test['total_corners']
baseline_corners = train['total_corners'].mean()
baseline_corners_mae = mean_absolute_error(
    y_test_corners,
    [baseline_corners] * len(y_test_corners)
)
print(f"Corners baseline MAE: {baseline_corners_mae:.3f}")
print(f"Corners baseline prediction: {baseline_corners:.1f}")

# BTTS baseline
y_test_btts = test['btts']
btts_rate = train['btts'].mean()
baseline_btts_pred = 1 if btts_rate > 0.5 else 0
baseline_btts_acc = (y_test_btts == baseline_btts_pred).mean()
print(f"\nBTTS rate in training: {btts_rate:.1%}")
print(f"Baseline always predicts: {'YES' if baseline_btts_pred == 1 else 'NO'}")
print(f"BTTS baseline accuracy: {baseline_btts_acc:.3f}")

# Over 2.5 baseline
y_test_o25 = test['over_2_5']
o25_rate = train['over_2_5'].mean()
baseline_o25_pred = 1 if o25_rate > 0.5 else 0
baseline_o25_acc = (y_test_o25 == baseline_o25_pred).mean()
print(f"\nOver 2.5 rate in training: {o25_rate:.1%}")
print(f"Baseline always predicts: {'YES' if baseline_o25_pred == 1 else 'NO'}")
print(f"Over 2.5 baseline accuracy: {baseline_o25_acc:.3f}")

# Define lambda predictions from goals model
lambda_home_preds = poisson_home.predict(X_test_g)
lambda_away_preds = poisson_away.predict(X_test_g)

# BTTS from Poisson
btts_probs = [
    (1 - stats.poisson.cdf(0, h)) *
    (1 - stats.poisson.cdf(0, a))
    for h, a in zip(lambda_home_preds, lambda_away_preds)
]
btts_preds = [1 if p > 0.5 else 0 for p in btts_probs]
btts_acc = (pd.Series(btts_preds) == test['btts'].values).mean()

# Over 2.5 from Poisson
o25_probs = [
    1 - stats.poisson.cdf(2, h + a)
    for h, a in zip(lambda_home_preds, lambda_away_preds)
]
o25_preds = [1 if p > 0.5 else 0 for p in o25_probs]
o25_acc = (pd.Series(o25_preds) == test['over_2_5'].values).mean()

print(f"BTTS model accuracy:     {btts_acc:.3f}")
print(f"BTTS baseline accuracy:  0.564")
print(f"BTTS improvement:        {((btts_acc - 0.564) / 0.564 * 100):.1f}%")

print(f"\nOver 2.5 model accuracy:    {o25_acc:.3f}")
print(f"Over 2.5 baseline accuracy: 0.558")
print(f"Over 2.5 improvement:       {((o25_acc - 0.558) / 0.558 * 100):.1f}%")

# ── BTTS — XGBOOST CLASSIFIER ──
btts_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_form', 'away_form',
    'attack_diff',
    'post_2024_transition'
]

X_train_btts = train[btts_features]
X_test_btts  = test[btts_features]

y_train_btts = train['btts']
y_test_btts  = test['btts']

# Baseline
baseline_btts_acc = max(
    y_test_btts.mean(),
    1 - y_test_btts.mean()
)

# XGBoost
xgb_btts = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
xgb_btts.fit(X_train_btts, y_train_btts)
preds_btts = xgb_btts.predict(X_test_btts)
btts_probs = xgb_btts.predict_proba(X_test_btts)[:, 1]
btts_acc = (preds_btts == y_test_btts).mean()

print("BTTS Results:")
print(f"Baseline accuracy: {baseline_btts_acc:.3f}")
print(f"XGBoost accuracy:  {btts_acc:.3f}")
print(f"Improvement:       {((btts_acc - baseline_btts_acc) / baseline_btts_acc * 100):.1f}%")

# Threshold analysis
btts_df = pd.DataFrame({
    'actual':    y_test_btts.values,
    'predicted': preds_btts,
    'prob_yes':  btts_probs
})
btts_df['correct'] = (btts_df['actual'] == btts_df['predicted'])

print("\nBTTS Threshold Analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)
for t in [0.5, 0.55, 0.6, 0.65, 0.7]:
    hc = btts_df[
        (btts_df['prob_yes'] > t) |
        (btts_df['prob_yes'] < 1-t)
    ]
    if len(hc) > 0:
        print(f"{t:<12} {len(hc):<10} {hc['correct'].mean():.1%}")

# ── OVER 2.5 — XGBOOST CLASSIFIER ──
o25_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_form', 'away_form',
    'home_shots_for', 'away_shots_for',
    'attack_diff',
    'post_2024_transition'
]

X_train_o25 = train[o25_features]
X_test_o25  = test[o25_features]

y_train_o25 = train['over_2_5']
y_test_o25  = test['over_2_5']

# Baseline
baseline_o25_acc = max(
    y_test_o25.mean(),
    1 - y_test_o25.mean()
)

# XGBoost
xgb_o25 = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
xgb_o25.fit(X_train_o25, y_train_o25)
preds_o25 = xgb_o25.predict(X_test_o25)
o25_probs = xgb_o25.predict_proba(X_test_o25)[:, 1]
o25_acc = (preds_o25 == y_test_o25).mean()

print("\nOver 2.5 Results:")
print(f"Baseline accuracy: {baseline_o25_acc:.3f}")
print(f"XGBoost accuracy:  {o25_acc:.3f}")
print(f"Improvement:       {((o25_acc - baseline_o25_acc) / baseline_o25_acc * 100):.1f}%")

# Threshold analysis
o25_df = pd.DataFrame({
    'actual':    y_test_o25.values,
    'predicted': preds_o25,
    'prob_yes':  o25_probs
})
o25_df['correct'] = (o25_df['actual'] == o25_df['predicted'])

print("\nOver 2.5 Threshold Analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)
for t in [0.5, 0.55, 0.6, 0.65, 0.7]:
    hc = o25_df[
        (o25_df['prob_yes'] > t) |
        (o25_df['prob_yes'] < 1-t)
    ]
    if len(hc) > 0:
        print(f"{t:<12} {len(hc):<10} {hc['correct'].mean():.1%}")

# ── CARDS — POISSON ──
card_features = [
    'home_cards',        # home team cards EWMA
    'away_cards',        # away team cards EWMA
    'ref_avg_cards',     # referee tendency ← key feature
    'post_2024_transition'
]

X_train_cards = train[card_features]
X_test_cards  = test[card_features]

y_train_hc = train['HY'] + train['HR']   # home cards
y_test_hc  = test['HY']  + test['HR']
y_train_ac = train['AY'] + train['AR']   # away cards
y_test_ac  = test['AY']  + test['AR']
y_train_tc = train['total_cards']
y_test_tc  = test['total_cards']

# Baseline
baseline_cards = y_train_tc.mean()
baseline_cards_mae = mean_absolute_error(
    y_test_tc,
    [baseline_cards] * len(y_test_tc)
)

# Poisson home cards
poisson_home_cards = PoissonRegressor(max_iter=300)
poisson_home_cards.fit(X_train_cards, y_train_hc)
preds_home_cards = poisson_home_cards.predict(X_test_cards)

# Poisson away cards
poisson_away_cards = PoissonRegressor(max_iter=300)
poisson_away_cards.fit(X_train_cards, y_train_ac)
preds_away_cards = poisson_away_cards.predict(X_test_cards)

# Total cards
preds_total_cards = preds_home_cards + preds_away_cards
cards_mae = mean_absolute_error(y_test_tc, preds_total_cards)

print("Cards — Poisson:")
print(f"Baseline MAE: {baseline_cards_mae:.3f}")
print(f"Poisson MAE:  {cards_mae:.3f}")
print(f"Improvement:  {(baseline_cards_mae - cards_mae)/baseline_cards_mae*100:.1f}%")

# Over/Under 3.5 cards
o35_cards_probs = [
    1 - stats.poisson.cdf(3, t)
    for t in preds_total_cards
]
o35_cards_preds = [1 if p > 0.5 else 0 for p in o35_cards_probs]
o35_cards_acc   = (pd.Series(o35_cards_preds) == 
                   (test['total_cards'] > 3.5).astype(int).values).mean()

baseline_o35 = max(
    (train['total_cards'] > 3.5).mean(),
    1-(train['total_cards'] > 3.5).mean()
)

print(f"\nOver 3.5 cards:")
print(f"Baseline: {baseline_o35:.3f}")
print(f"Model:    {o35_cards_acc:.3f}")