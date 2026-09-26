import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor, XGBClassifier

# ── LOAD DATA ──
df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['post_2024_transition'] = (df['Date'] >= '2024-08-01').astype(int)
df = df.sort_values('Date').reset_index(drop=True)

# ── ADD DERIVED COLUMNS ──
df['total_goals']   = df['FTHG'] + df['FTAG']
df['total_corners'] = df['HC'] + df['AC']
df['total_cards']   = df['HY'] + df['AY'] + df['HR'] + df['AR']
df['total_shots']   = df['HS'] + df['AS']
df['btts']          = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
df['over_2_5']      = (df['total_goals'] > 2.5).astype(int)

# ── FEATURES ──
goal_features = [
    'home_attack',
    'home_defence',
    'away_attack',
    'away_defence',
    'home_shots_for',
    'away_shots_for',
    'post_2024_transition'
]

# ── TRAIN/TEST SPLIT ──
train = df[df['Date'] < '2024-08-01']
test  = df[df['Date'] >= '2024-08-01']

print("Train rows:", len(train))
print("Test rows: ", len(test))

X_train = train[goal_features]
X_test  = test[goal_features]

y_train_home  = train['FTHG']
y_test_home   = test['FTHG']
y_train_away  = train['FTAG']
y_test_away   = test['FTAG']
y_train_goals = train['total_goals']
y_test_goals  = test['total_goals']

# ── BASELINE ──
baseline_prediction = y_train_goals.mean()
baseline_preds = [baseline_prediction] * len(y_test_goals)
baseline_mae = mean_absolute_error(y_test_goals, baseline_preds)
print("\nBaseline MAE:", round(baseline_mae, 3))

# ── POISSON MODEL ──
poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(X_train, y_train_home)
preds_home = poisson_home.predict(X_test)

poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(X_train, y_train_away)
preds_away = poisson_away.predict(X_test)

preds_total = preds_home + preds_away
poisson_mae = mean_absolute_error(y_test_goals, preds_total)

print("Poisson MAE: ", round(poisson_mae, 3))
print("Improvement: ", round((baseline_mae - poisson_mae) / baseline_mae * 100, 1), "%")

# ── CORRELATION PLOT ──
corr_cols = goal_features + ['total_goals', 'FTHG', 'FTAG']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0)
plt.title('Feature Correlation with Goals')
plt.tight_layout()
plt.savefig('correlation_plot_v2.png')
plt.show()

print("\nCorrelation with total_goals:")
print(corr_matrix['total_goals'].sort_values(ascending=False))

# ── XGBOOST FOR TOTAL GOALS ──
xgb_goals = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_goals.fit(X_train, y_train_goals)
xgb_preds = xgb_goals.predict(X_test)
xgb_mae = mean_absolute_error(y_test_goals, xgb_preds)

print("\nXGBoost Total Goals MAE:", round(xgb_mae, 3))
print("Poisson MAE:            ", round(poisson_mae, 3))
print("Baseline MAE:           ", round(baseline_mae, 3))

# Feature importance
importance = pd.Series(
    xgb_goals.feature_importances_,
    index=goal_features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance.round(3))

corner_features = [
    'home_corners_for',
    'home_corners_against',
    'away_corners_for',
    'away_corners_against',
    'home_attack',
    'away_attack',
    'post_2024_transition'
]

X_train_c = train[corner_features]
X_test_c  = test[corner_features]

y_train_corners = train['total_corners']
y_test_corners  = test['total_corners']

# Baseline
baseline_corners = y_train_corners.mean()
baseline_corners_mae = mean_absolute_error(
    y_test_corners,
    [baseline_corners] * len(y_test_corners)
)

# XGBoost
xgb_corners = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_corners.fit(X_train_c, y_train_corners)
preds_corners = xgb_corners.predict(X_test_c)
corners_mae = mean_absolute_error(y_test_corners, preds_corners)

# Feature importance
corner_importance = pd.Series(
    xgb_corners.feature_importances_,
    index=corner_features
).sort_values(ascending=False)

print("Corners Results:")
print("Baseline MAE:", round(baseline_corners_mae, 3))
print("XGBoost MAE: ", round(corners_mae, 3))
print("Improvement: ", round((baseline_corners_mae - corners_mae) / baseline_corners_mae * 100, 1), "%")
print("\nFeature Importance:")
print(corner_importance.round(3))

# Create a unified view of all matches from each team's perspective
home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 
              'HC', 'AC', 'HY', 'AY', 'HR', 'AR']].copy()
home_df.columns = ['Date', 'Team', 'GF', 'GA', 
                   'CF', 'CA', 'YF', 'YA', 'RF', 'RA']
home_df['home'] = 1

away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG',
              'AC', 'HC', 'AY', 'HY', 'AR', 'HR']].copy()
away_df.columns = ['Date', 'Team', 'GF', 'GA',
                   'CF', 'CA', 'YF', 'YA', 'RF', 'RA']
away_df['home'] = 0

# Combine into one unified team performance table
team_df = pd.concat([home_df, away_df]).sort_values('Date').reset_index(drop=True)

# Calculate rolling averages across ALL games not just home or away
team_df['roll_GF'] = team_df.groupby('Team')['GF'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)
team_df['roll_GA'] = team_df.groupby('Team')['GA'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)
team_df['roll_CF'] = team_df.groupby('Team')['CF'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)
team_df['roll_CA'] = team_df.groupby('Team')['CA'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)

print(team_df[['Date', 'Team', 'GF', 'roll_GF', 
               'CF', 'roll_CF']].head(20))

# ── MERGE RATINGS BACK INTO MATCH DATA ──

# Split back into home and away
home_ratings = team_df[team_df['home'] == 1][
    ['Date', 'Team', 'roll_GF', 'roll_GA', 'roll_CF', 'roll_CA']
].copy()
home_ratings.columns = [
    'Date', 'HomeTeam', 
    'home_roll_GF', 'home_roll_GA', 
    'home_roll_CF', 'home_roll_CA'
]

away_ratings = team_df[team_df['home'] == 0][
    ['Date', 'Team', 'roll_GF', 'roll_GA', 'roll_CF', 'roll_CA']
].copy()
away_ratings.columns = [
    'Date', 'AwayTeam',
    'away_roll_GF', 'away_roll_GA',
    'away_roll_CF', 'away_roll_CA'
]

# Merge into main dataframe
df = df.merge(home_ratings, on=['Date', 'HomeTeam'], how='left')
df = df.merge(away_ratings, on=['Date', 'AwayTeam'], how='left')

# Drop NaN rows
df_v2 = df.dropna(subset=[
    'home_roll_GF', 'home_roll_GA',
    'away_roll_GF', 'away_roll_GA',
    'home_roll_CF', 'away_roll_CF'
])

print("Rows after merge:", len(df_v2))
print(df_v2[['Date', 'HomeTeam', 'AwayTeam',
             'home_roll_GF', 'away_roll_GF',
             'home_roll_CF', 'away_roll_CF']].head(10))

# ── NEW TRAIN/TEST SPLIT ON UPDATED DATAFRAME ──
train_v2 = df_v2[df_v2['Date'] < '2024-08-01']
test_v2  = df_v2[df_v2['Date'] >= '2024-08-01']

print("Train rows:", len(train_v2))
print("Test rows: ", len(test_v2))

# ── CORNERS MODEL V2 ──
corner_features_v2 = [
    'home_roll_CF',
    'home_roll_CA',
    'away_roll_CF',
    'away_roll_CA',
]

X_train_c2 = train_v2[corner_features_v2]
X_test_c2  = test_v2[corner_features_v2]

y_train_c2 = train_v2['total_corners']
y_test_c2  = test_v2['total_corners']

# Baseline
baseline_c2 = y_train_c2.mean()
baseline_c2_mae = mean_absolute_error(
    y_test_c2,
    [baseline_c2] * len(y_test_c2)
)

# XGBoost
xgb_c2 = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
xgb_c2.fit(X_train_c2, y_train_c2)
preds_c2 = xgb_c2.predict(X_test_c2)
corners_mae_v2 = mean_absolute_error(y_test_c2, preds_c2)

print("\nCorners V2 Results:")
print("Baseline MAE:", round(baseline_c2_mae, 3))
print("XGBoost MAE: ", round(corners_mae_v2, 3))
print("Improvement: ", round((baseline_c2_mae - corners_mae_v2) / baseline_c2_mae * 100, 1), "%")

# ── GOALS MODEL V2 ──
goal_features_v2 = [
    'home_roll_GF',
    'home_roll_GA',
    'away_roll_GF',
    'away_roll_GA',
]

X_train_g2 = train_v2[goal_features_v2]
X_test_g2  = test_v2[goal_features_v2]

y_train_home2  = train_v2['FTHG']
y_test_home2   = test_v2['FTHG']
y_train_away2  = train_v2['FTAG']
y_test_away2   = test_v2['FTAG']
y_train_goals2 = train_v2['total_goals']
y_test_goals2  = test_v2['total_goals']

# Baseline
baseline_g2 = y_train_goals2.mean()
baseline_g2_mae = mean_absolute_error(
    y_test_goals2,
    [baseline_g2] * len(y_test_goals2)
)

# Poisson
poisson_home2 = PoissonRegressor(max_iter=300)
poisson_home2.fit(X_train_g2, y_train_home2)
preds_home2 = poisson_home2.predict(X_test_g2)

poisson_away2 = PoissonRegressor(max_iter=300)
poisson_away2.fit(X_train_g2, y_train_away2)
preds_away2 = poisson_away2.predict(X_test_g2)

preds_total2 = preds_home2 + preds_away2
goals_mae_v2 = mean_absolute_error(y_test_goals2, preds_total2)

print("\nGoals V2 Results:")
print("Baseline MAE:", round(baseline_g2_mae, 3))
print("Poisson MAE: ", round(goals_mae_v2, 3))
print("Improvement: ", round((baseline_g2_mae - goals_mae_v2) / baseline_g2_mae * 100, 1), "%")

# ── MATCH RESULT CLASSIFIER ──
result_features = [
    'home_roll_GF',
    'home_roll_GA',
    'away_roll_GF',
    'away_roll_GA',
    'home_roll_CF',
    'away_roll_CF',
    'post_2024_transition'
]

X_train_r = train_v2[result_features]
X_test_r  = test_v2[result_features]

# Encode result: H=0, D=1, A=2
result_map = {'H': 0, 'D': 1, 'A': 2}
y_train_r = train_v2['FTR'].map(result_map)
y_test_r  = test_v2['FTR'].map(result_map)

# Baseline — always predict home win (most common outcome)
baseline_acc = (y_test_r == 0).mean()
print("Baseline accuracy (always home win):", round(baseline_acc, 3))

# XGBoost classifier
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

print("XGBoost accuracy:                   ", round(result_acc, 3))
print("Improvement:                        ", 
      round((result_acc - baseline_acc) / baseline_acc * 100, 1), "%")

from sklearn.metrics import classification_report, confusion_matrix

# Detailed breakdown
print("Classification Report:")
print(classification_report(
    y_test_r, preds_r,
    target_names=['Home Win', 'Draw', 'Away Win']
))

# Confusion matrix
cm = confusion_matrix(y_test_r, preds_r)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Home Win', 'Draw', 'Away Win'],
            yticklabels=['Home Win', 'Draw', 'Away Win'])
plt.title('Match Result Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# The fix — add draw-specific features
# Draws happen when teams are evenly matched

# Add a strength difference feature
df_v2['attack_diff'] = df_v2['home_roll_GF'] - df_v2['away_roll_GF']
df_v2['defence_diff'] = df_v2['home_roll_GA'] - df_v2['away_roll_GA']
df_v2['form_diff'] = df_v2['home_roll_GF'] - df_v2['away_roll_GF']

# When attack_diff is close to 0 → teams are evenly matched → draw likely
# When attack_diff is large → stronger team likely wins

result_features_v2 = [
    'home_roll_GF',
    'home_roll_GA',
    'away_roll_GF',
    'away_roll_GA',
    'home_roll_CF',
    'away_roll_CF',
    'attack_diff',
    'defence_diff',
    'post_2024_transition'
]

# Rebuild train/test with new features
train_v2 = df_v2[df_v2['Date'] < '2024-08-01']
test_v2  = df_v2[df_v2['Date'] >= '2024-08-01']

X_train_r2 = train_v2[result_features_v2]
X_test_r2  = test_v2[result_features_v2]

y_train_r2 = train_v2['FTR'].map({'H': 0, 'D': 1, 'A': 2})
y_test_r2  = test_v2['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Use class weights to force model to predict more draws
xgb_result_v2 = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_result_v2.fit(X_train_r2, y_train_r2)
preds_r2 = xgb_result_v2.predict(X_test_r2)
result_acc_v2 = (preds_r2 == y_test_r2).mean()

print("V2 XGBoost accuracy:", round(result_acc_v2, 3))
print("V1 XGBoost accuracy:", round(result_acc, 3))
print("Baseline accuracy:  ", round(baseline_acc, 3))

from sklearn.metrics import classification_report
print(classification_report(
    y_test_r2, preds_r2,
    target_names=['Home Win', 'Draw', 'Away Win']
))

# Get probabilities instead of hard predictions
probs = xgb_result_v2.predict_proba(X_test_r2)

# probs columns: [Home Win, Draw, Away Win]
prob_df = pd.DataFrame(probs, 
    columns=['prob_home', 'prob_draw', 'prob_away'])
prob_df['actual'] = y_test_r2.values
prob_df['predicted'] = preds_r2

# Look at matches where draw probability is highest
draw_candidates = prob_df.sort_values('prob_draw', ascending=False).head(20)
print("Top 20 matches by draw probability:")
print(draw_candidates.round(3))

# How often is actual result a draw when model gives high draw probability?
high_draw_prob = prob_df[prob_df['prob_draw'] > 0.30]
actual_draw_rate = (high_draw_prob['actual'] == 1).mean()
print(f"\nWhen draw prob > 30%: actual draw rate = {actual_draw_rate:.1%}")
print(f"Number of such matches: {len(high_draw_prob)}")

# Add team names back to probability output
results_df = test_v2[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
results_df = results_df.reset_index(drop=True)

results_df['prob_home'] = probs[:, 0].round(3)
results_df['prob_draw'] = probs[:, 1].round(3)
results_df['prob_away'] = probs[:, 2].round(3)
results_df['predicted'] = preds_r2
results_df['predicted'] = results_df['predicted'].map(
    {0: 'H', 1: 'D', 2: 'A'}
)
results_df['correct'] = (results_df['predicted'] == results_df['FTR'])

print(results_df.head(15).to_string())
print("\nOverall accuracy:", results_df['correct'].mean().round(3))

# Show some interesting predictions
print("\nHigh confidence Home Win predictions (prob_home > 0.6):")
high_conf = results_df[results_df['prob_home'] > 0.6]
print(f"Accuracy on these: {high_conf['correct'].mean():.1%}")
print(f"Number of matches: {len(high_conf)}")

# Check accuracy at different confidence thresholds
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

print("Confidence threshold analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)

for threshold in thresholds:
    # High confidence for any outcome
    high_conf = results_df[
        (results_df['prob_home'] > threshold) |
        (results_df['prob_away'] > threshold)
    ]
    if len(high_conf) > 0:
        acc = high_conf['correct'].mean()
        print(f"{threshold:<12} {len(high_conf):<10} {acc:.1%}")

# Also check away win confidence
print("\nHigh confidence Away Win (prob_away > 0.6):")
high_away = results_df[results_df['prob_away'] > 0.6]
print(f"Accuracy: {high_away['correct'].mean():.1%}")
print(f"Matches:  {len(high_away)}")  

# BTTS Model
y_train_btts = train_v2['btts']
y_test_btts  = test_v2['btts']

btts_features = [
    'home_roll_GF',
    'home_roll_GA',
    'away_roll_GF',
    'away_roll_GA',
    'attack_diff',
    'post_2024_transition'
]

X_train_b = train_v2[btts_features]
X_test_b  = test_v2[btts_features]

# Baseline
baseline_btts = y_test_btts.mean()
print("BTTS base rate:", round(baseline_btts, 3))

xgb_btts = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
xgb_btts.fit(X_train_b, y_train_btts)
preds_btts = xgb_btts.predict(X_test_b)
btts_acc = (preds_btts == y_test_btts).mean()

print("XGBoost BTTS accuracy:", round(btts_acc, 3))
print("Baseline accuracy:    ", round(max(baseline_btts, 1-baseline_btts), 3))

from sklearn.utils.class_weight import compute_sample_weight

# Calculate sample weights to balance classes
weights = compute_sample_weight('balanced', y_train_btts)

xgb_btts_v2 = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=(1 - baseline_btts) / baseline_btts
)
xgb_btts_v2.fit(X_train_b, y_train_btts, 
                sample_weight=weights)
preds_btts_v2 = xgb_btts_v2.predict(X_test_b)
btts_acc_v2 = (preds_btts_v2 == y_test_btts).mean()

# Get probabilities
btts_probs = xgb_btts_v2.predict_proba(X_test_b)

print("Balanced XGBoost BTTS accuracy:", round(btts_acc_v2, 3))
print("Baseline accuracy:             ", round(max(baseline_btts, 1-baseline_btts), 3))

# Threshold analysis
print("\nBTTS Confidence threshold analysis:")
print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10}")
print("-" * 32)

btts_results = pd.DataFrame({
    'actual': y_test_btts.values,
    'predicted': preds_btts_v2,
    'prob_yes': btts_probs[:, 1],
    'prob_no': btts_probs[:, 0]
})
btts_results['correct'] = (btts_results['actual'] == btts_results['predicted'])

for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
    high_conf = btts_results[
        (btts_results['prob_yes'] > threshold) |
        (btts_results['prob_no'] > threshold)
    ]
    if len(high_conf) > 0:
        acc = high_conf['correct'].mean()
        print(f"{threshold:<12} {len(high_conf):<10} {acc:.1%}")