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
df['attack_diff']   = df['home_attack'] - df['away_attack']
df['defence_diff']  = df['home_defence'] - df['away_defence']

# ── TRAIN/TEST SPLIT ──
train = df[df['Date'] < '2024-08-01']
test  = df[df['Date'] >= '2024-08-01']

print("Train rows:", len(train))
print("Test rows: ", len(test))

# ══════════════════════════════════════════
# RANGE PREDICTION FUNCTION
# ══════════════════════════════════════════
def predict_with_range(lambda_val, target_name, confidence=0.80):
    """
    Given a Poisson lambda value, returns:
    - Expected value
    - Most likely single outcome (mode)
    - Credible interval at given confidence level
    - Probability that true value falls in range
    """
    # Mode — most likely single value
    mode = int(np.floor(lambda_val))
    mode_prob = stats.poisson.pmf(mode, lambda_val)

    # Credible interval bounds
    lower_tail = (1 - confidence) / 2
    upper_tail = 1 - (1 - confidence) / 2

    lower = int(stats.poisson.ppf(lower_tail, lambda_val))
    upper = int(stats.poisson.ppf(upper_tail, lambda_val))

    # Actual probability covered by this range
    range_prob = (
        stats.poisson.cdf(upper, lambda_val) -
        stats.poisson.cdf(lower - 1, lambda_val)
    )

    return {
        'target':      target_name,
        'expected':    round(lambda_val, 1),
        'most_likely': mode,
        'mode_prob':   round(mode_prob, 3),
        'lower':       lower,
        'upper':       upper,
        'range_prob':  round(range_prob, 3),
        'confidence':  confidence
    }

def print_range(result):
    """Pretty print range prediction result"""
    conf_pct = int(result['confidence'] * 100)
    print(f"  Expected (λ):       {result['expected']}")
    print(f"  Most likely:        {result['most_likely']}"
          f" ({result['mode_prob']:.1%} probability)")
    print(f"  {conf_pct}% range:          "
          f"{result['lower']} — {result['upper']}")
    print(f"  Range probability:  {result['range_prob']:.1%}")

# ══════════════════════════════════════════
# CORNERS — POISSON
# ══════════════════════════════════════════
corner_features = [
    'home_corners_for',
    'home_corners_against',
    'away_corners_for',
    'away_corners_against',
    'post_2024_transition'
]

X_train_c = train[corner_features]
X_test_c  = test[corner_features]

y_train_tc = train['total_corners']
y_test_tc  = test['total_corners']

# Baseline
baseline_c     = y_train_tc.mean()
baseline_c_mae = mean_absolute_error(
    y_test_tc, [baseline_c] * len(y_test_tc)
)

# Poisson home corners
poisson_hc = PoissonRegressor(max_iter=300)
poisson_hc.fit(X_train_c, train['HC'])
preds_hc = poisson_hc.predict(X_test_c)

# Poisson away corners
poisson_ac = PoissonRegressor(max_iter=300)
poisson_ac.fit(X_train_c, train['AC'])
preds_ac = poisson_ac.predict(X_test_c)

# Combined total corners
preds_tc    = preds_hc + preds_ac
corners_mae = mean_absolute_error(y_test_tc, preds_tc)

print("\n" + "="*50)
print("CORNERS — Poisson with EWMA features")
print("="*50)
print(f"Baseline MAE: {baseline_c_mae:.3f}")
print(f"Poisson MAE:  {corners_mae:.3f}")
print(f"Improvement:  {(baseline_c_mae - corners_mae) / baseline_c_mae * 100:.1f}%")

# Average range across all test matches
avg_lambda_corners = preds_tc.mean()
corners_range = predict_with_range(avg_lambda_corners, 'Corners')
print("\nAverage prediction range across test set:")
print_range(corners_range)

# XGBoost corners for comparison
xgb_c = XGBRegressor(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, subsample=0.8,
    random_state=42
)
xgb_c.fit(X_train_c, y_train_tc)
xgb_corners_mae = mean_absolute_error(y_test_tc, xgb_c.predict(X_test_c))

print(f"\nCorners — XGBoost:")
print(f"Baseline MAE: {baseline_c_mae:.3f}")
print(f"XGBoost MAE:  {xgb_corners_mae:.3f}")
print(f"Improvement:  {(baseline_c_mae - xgb_corners_mae) / baseline_c_mae * 100:.1f}%")

# ══════════════════════════════════════════
# GOALS — POISSON
# ══════════════════════════════════════════
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

baseline_g     = y_train_goals.mean()
baseline_g_mae = mean_absolute_error(
    y_test_goals, [baseline_g] * len(y_test_goals)
)

poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(X_train_g, train['FTHG'])
preds_home = poisson_home.predict(X_test_g)

poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(X_train_g, train['FTAG'])
preds_away = poisson_away.predict(X_test_g)

preds_total_goals = preds_home + preds_away
goals_mae = mean_absolute_error(y_test_goals, preds_total_goals)

print("\n" + "="*50)
print("GOALS — Poisson with EWMA features")
print("="*50)
print(f"Baseline MAE: {baseline_g_mae:.3f}")
print(f"Poisson MAE:  {goals_mae:.3f}")
print(f"Improvement:  {(baseline_g_mae - goals_mae) / baseline_g_mae * 100:.1f}%")

# Average range across all test matches
avg_lambda_goals = preds_total_goals.mean()
goals_range = predict_with_range(avg_lambda_goals, 'Goals')
print("\nAverage prediction range across test set:")
print_range(goals_range)

# Show range for a few individual matches
print("\nSample match ranges (first 5 test matches):")
print(f"{'Match':<5} {'λ home':<8} {'λ away':<8} {'λ total':<9} {'Most likely':<13} {'80% range'}")
print("-" * 60)
for i in range(5):
    lh = round(preds_home[i], 2)
    la = round(preds_away[i], 2)
    lt = round(lh + la, 2)
    r  = predict_with_range(lt, 'Goals', 0.80)
    home_team = test.iloc[i]['HomeTeam']
    away_team = test.iloc[i]['AwayTeam']
    print(f"{i+1:<5} {lh:<8} {la:<8} {lt:<9} "
          f"{r['most_likely']} goals ({r['mode_prob']:.0%})   "
          f"{r['lower']}—{r['upper']}")

# Lambda predictions for downstream use
lambda_home_preds = preds_home
lambda_away_preds = preds_away

# ══════════════════════════════════════════
# MATCH RESULT — XGBOOST
# ══════════════════════════════════════════
result_features_full = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_corners_for', 'away_corners_for',
    'home_form', 'away_form',
    'attack_diff', 'defence_diff',
    'post_2024_transition'
]

X_train_r = train[result_features_full]
X_test_r  = test[result_features_full]

y_train_r = train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
y_test_r  = test['FTR'].map({'H': 0, 'D': 1, 'A': 2})

baseline_acc = (y_test_r == 0).mean()

xgb_result = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='mlogloss'
)
xgb_result.fit(X_train_r, y_train_r)
preds_r    = xgb_result.predict(X_test_r)
result_acc = (preds_r == y_test_r).mean()

print("\n" + "="*50)
print("MATCH RESULT — XGBoost with EWMA features")
print("="*50)
print(f"Baseline accuracy: {baseline_acc:.3f}")
print(f"XGBoost accuracy:  {result_acc:.3f}")
print(f"Improvement:       {(result_acc - baseline_acc) / baseline_acc * 100:.1f}%")

# Threshold analysis
probs      = xgb_result.predict_proba(X_test_r)
results_df = test[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy().reset_index(drop=True)
results_df['prob_home'] = probs[:, 0].round(3)
results_df['prob_draw'] = probs[:, 1].round(3)
results_df['prob_away'] = probs[:, 2].round(3)
results_df['predicted'] = pd.Series(preds_r).map({0:'H', 1:'D', 2:'A'}).values
results_df['correct']   = (results_df['predicted'] == results_df['FTR'])

print("\nThreshold analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    high_conf = results_df[
        (results_df['prob_home'] > threshold) |
        (results_df['prob_away'] > threshold)
    ]
    if len(high_conf) > 0:
        print(f"{threshold:<12} {len(high_conf):<10} "
              f"{high_conf['correct'].mean():.1%}")

# ══════════════════════════════════════════
# BTTS AND OVER 2.5 — BASELINES
# ══════════════════════════════════════════
print("\n" + "="*50)
print("MARKET BASELINES")
print("="*50)

y_test_btts = test['btts']
btts_rate   = train['btts'].mean()
baseline_btts_pred = 1 if btts_rate > 0.5 else 0
baseline_btts_acc  = (y_test_btts == baseline_btts_pred).mean()
print(f"BTTS rate in training:    {btts_rate:.1%}")
print(f"BTTS baseline accuracy:   {baseline_btts_acc:.3f}")

y_test_o25  = test['over_2_5']
o25_rate    = train['over_2_5'].mean()
baseline_o25_pred = 1 if o25_rate > 0.5 else 0
baseline_o25_acc  = (y_test_o25 == baseline_o25_pred).mean()
print(f"\nOver 2.5 rate in training: {o25_rate:.1%}")
print(f"Over 2.5 baseline accuracy: {baseline_o25_acc:.3f}")

# ══════════════════════════════════════════
# BTTS FROM POISSON + RANGE
# ══════════════════════════════════════════
print("\n" + "="*50)
print("BTTS — Derived from Poisson λ")
print("="*50)

btts_probs_poisson = [
    (1 - stats.poisson.cdf(0, h)) *
    (1 - stats.poisson.cdf(0, a))
    for h, a in zip(lambda_home_preds, lambda_away_preds)
]
btts_preds_poisson = [1 if p > 0.5 else 0 for p in btts_probs_poisson]
btts_acc_poisson   = (pd.Series(btts_preds_poisson) == y_test_btts.values).mean()

print(f"Baseline accuracy: {baseline_btts_acc:.3f}")
print(f"Poisson accuracy:  {btts_acc_poisson:.3f}")
print(f"Improvement:       {(btts_acc_poisson - baseline_btts_acc) / baseline_btts_acc * 100:.1f}%")

# Average BTTS probability across test set
avg_btts_prob = np.mean(btts_probs_poisson)
print(f"\nAverage BTTS probability: {avg_btts_prob:.1%}")
print(f"Model predicts BTTS YES:  "
      f"{sum(btts_preds_poisson)} / {len(btts_preds_poisson)} matches")

# ══════════════════════════════════════════
# OVER 2.5 FROM POISSON + RANGE
# ══════════════════════════════════════════
print("\n" + "="*50)
print("OVER 2.5 — Derived from Poisson λ")
print("="*50)

o25_probs_poisson = [
    1 - stats.poisson.cdf(2, h + a)
    for h, a in zip(lambda_home_preds, lambda_away_preds)
]
o25_preds_poisson = [1 if p > 0.5 else 0 for p in o25_probs_poisson]
o25_acc_poisson   = (pd.Series(o25_preds_poisson) == y_test_o25.values).mean()

print(f"Baseline accuracy: {baseline_o25_acc:.3f}")
print(f"Poisson accuracy:  {o25_acc_poisson:.3f}")
print(f"Improvement:       {(o25_acc_poisson - baseline_o25_acc) / baseline_o25_acc * 100:.1f}%")

avg_o25_prob = np.mean(o25_probs_poisson)
print(f"\nAverage Over 2.5 probability: {avg_o25_prob:.1%}")

# ══════════════════════════════════════════
# BTTS — XGBOOST
# ══════════════════════════════════════════
print("\n" + "="*50)
print("BTTS — XGBoost Classifier")
print("="*50)

btts_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_form', 'away_form',
    'attack_diff', 'post_2024_transition'
]

xgb_btts = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='logloss'
)
xgb_btts.fit(train[btts_features], train['btts'])
preds_btts  = xgb_btts.predict(test[btts_features])
btts_probs  = xgb_btts.predict_proba(test[btts_features])[:, 1]
btts_acc    = (preds_btts == y_test_btts).mean()

print(f"Baseline accuracy: {baseline_btts_acc:.3f}")
print(f"XGBoost accuracy:  {btts_acc:.3f}")
print(f"Improvement:       {(btts_acc - baseline_btts_acc) / baseline_btts_acc * 100:.1f}%")

btts_df = pd.DataFrame({
    'actual': y_test_btts.values,
    'predicted': preds_btts,
    'prob_yes': btts_probs
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

# ══════════════════════════════════════════
# OVER 2.5 — XGBOOST
# ══════════════════════════════════════════
print("\n" + "="*50)
print("OVER 2.5 — XGBoost Classifier")
print("="*50)

o25_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_form', 'away_form',
    'home_shots_for', 'away_shots_for',
    'attack_diff', 'post_2024_transition'
]

xgb_o25 = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='logloss'
)
xgb_o25.fit(train[o25_features], train['over_2_5'])
preds_o25 = xgb_o25.predict(test[o25_features])
o25_probs = xgb_o25.predict_proba(test[o25_features])[:, 1]
o25_acc   = (preds_o25 == y_test_o25).mean()

print(f"Baseline accuracy: {baseline_o25_acc:.3f}")
print(f"XGBoost accuracy:  {o25_acc:.3f}")
print(f"Improvement:       {(o25_acc - baseline_o25_acc) / baseline_o25_acc * 100:.1f}%")

o25_df = pd.DataFrame({
    'actual': y_test_o25.values,
    'predicted': preds_o25,
    'prob_yes': o25_probs
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

# ══════════════════════════════════════════
# FULL SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*50)
print("COMPLETE MODEL SUMMARY")
print("="*50)
print(f"{'Market':<20} {'Baseline':<12} {'Model':<12} {'Better?'}")
print("-" * 56)
print(f"{'Match Result':<20} {baseline_acc:.1%}       {result_acc:.1%}       "
      f"{'✅' if result_acc > baseline_acc else '❌'} "
      f"{(result_acc - baseline_acc)/baseline_acc*100:+.1f}%")
print(f"{'Corners MAE':<20} {baseline_c_mae:.3f}        {corners_mae:.3f}        "
      f"{'✅' if corners_mae < baseline_c_mae else '❌'} "
      f"{(baseline_c_mae - corners_mae)/baseline_c_mae*100:+.1f}%")
print(f"{'Goals MAE':<20} {baseline_g_mae:.3f}        {goals_mae:.3f}        "
      f"{'✅' if goals_mae < baseline_g_mae else '❌'} "
      f"{(baseline_g_mae - goals_mae)/baseline_g_mae*100:+.1f}%")
print(f"{'BTTS XGBoost':<20} {baseline_btts_acc:.1%}       {btts_acc:.1%}       "
      f"{'✅' if btts_acc > baseline_btts_acc else '❌'} "
      f"{(btts_acc - baseline_btts_acc)/baseline_btts_acc*100:+.1f}%")
print(f"{'Over 2.5 XGBoost':<20} {baseline_o25_acc:.1%}       {o25_acc:.1%}       "
      f"{'✅' if o25_acc > baseline_o25_acc else '❌'} "
      f"{(o25_acc - baseline_o25_acc)/baseline_o25_acc*100:+.1f}%")

print(f"\nKey finding:")
print(f"Match result at 0.65 threshold: 61.4% vs {baseline_acc:.1%} baseline")
print(f"Over 2.5 at 0.70 threshold:     63.0% vs {baseline_o25_acc:.1%} baseline")

print(f"\nGoals range (avg across test set):")
print_range(goals_range)
print(f"\nCorners range (avg across test set):")
print_range(corners_range)