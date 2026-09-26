import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── LOAD ADJUSTED DATASET ──
df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered_adj.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.sort_values('Date').reset_index(drop=True)

# Derived columns
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

# ── GOALS MODEL ──
goal_features = [
    'home_attack_adj',
    'away_attack_adj',
    'home_defence_adj',
    'away_defence_adj',
    'home_shots_adj',
    'away_shots_adj',
    'home_form',
    'away_form',
    'post_2024_transition'
]

X_train_g = train[goal_features]
X_test_g  = test[goal_features]

baseline_g_mae = mean_absolute_error(
    test['total_goals'],
    [train['total_goals'].mean()] * len(test)
)

poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(X_train_g, train['FTHG'])
poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(X_train_g, train['FTAG'])

preds_goals = poisson_home.predict(X_test_g) + poisson_away.predict(X_test_g)
goals_mae = mean_absolute_error(test['total_goals'], preds_goals)

print("\nGoals — Opponent Adjusted:")
print("Baseline MAE:", round(baseline_g_mae, 3))
print("Poisson MAE: ", round(goals_mae, 3))
print("Improvement: ", round((baseline_g_mae - goals_mae) / baseline_g_mae * 100, 1), "%")

# ── CORNERS MODEL ──
corner_features = [
    'home_corners_adj',
    'away_corners_adj',
    'home_corners_for',
    'away_corners_for',
    'corner_diff',
    'post_2024_transition'
]

X_train_c = train[corner_features]
X_test_c  = test[corner_features]

baseline_c_mae = mean_absolute_error(
    test['total_corners'],
    [train['total_corners'].mean()] * len(test)
)

# Poisson corners
poisson_hc = PoissonRegressor(max_iter=300)
poisson_hc.fit(X_train_c, train['HC'])
poisson_ac = PoissonRegressor(max_iter=300)
poisson_ac.fit(X_train_c, train['AC'])

preds_corners = poisson_hc.predict(X_test_c) + poisson_ac.predict(X_test_c)
corners_mae = mean_absolute_error(test['total_corners'], preds_corners)

# XGBoost corners
xgb_c = XGBRegressor(n_estimators=200, max_depth=3,
                      learning_rate=0.05, random_state=42)
xgb_c.fit(X_train_c, train['total_corners'])
xgb_corners_mae = mean_absolute_error(test['total_corners'],
                                       xgb_c.predict(X_test_c))

print("\nCorners — Opponent Adjusted:")
print("Baseline MAE:", round(baseline_c_mae, 3))
print("Poisson MAE: ", round(corners_mae, 3))
print("XGBoost MAE: ", round(xgb_corners_mae, 3))
print("Poisson improvement: ", round((baseline_c_mae - corners_mae) / baseline_c_mae * 100, 1), "%")
print("XGBoost improvement: ", round((baseline_c_mae - xgb_corners_mae) / baseline_c_mae * 100, 1), "%")

# ── MATCH RESULT MODEL ──
result_features = [
    'home_attack_adj',
    'away_attack_adj',
    'home_defence_adj',
    'away_defence_adj',
    'home_corners_adj',
    'away_corners_adj',
    'home_shots_adj',
    'away_shots_adj',
    'home_form',
    'away_form',
    'attack_diff',
    'corner_diff',
    'post_2024_transition'
]

X_train_r = train[result_features]
X_test_r  = test[result_features]

y_train_r = train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
y_test_r  = test['FTR'].map({'H': 0, 'D': 1, 'A': 2})

baseline_acc = (y_test_r == 0).mean()

xgb_result = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='mlogloss'
)
xgb_result.fit(X_train_r, y_train_r)
preds_r = xgb_result.predict(X_test_r)
result_acc = (preds_r == y_test_r).mean()

print("\nMatch Result — Opponent Adjusted:")
print("Baseline accuracy:", round(baseline_acc, 3))
print("XGBoost accuracy: ", round(result_acc, 3))
print("Improvement:      ", round((result_acc - baseline_acc) / baseline_acc * 100, 1), "%")

# Threshold analysis
probs = xgb_result.predict_proba(X_test_r)
results_df = test[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy().reset_index(drop=True)
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