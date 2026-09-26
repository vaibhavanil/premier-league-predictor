import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

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

# ── TRAIN ON ALL DATA UP TO TODAY ──
# Use everything available to train
# No test split needed for live predictions
train = df.copy()

# ── FEATURES ──
result_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_corners_for', 'away_corners_for',
    'home_form', 'away_form',
    'attack_diff', 'defence_diff',
    'post_2024_transition'
]

corner_features = [
    'home_corners_for', 'home_corners_against',
    'away_corners_for', 'away_corners_against',
    'post_2024_transition'
]

goal_features = [
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_shots_for', 'away_shots_for',
    'post_2024_transition'
]

# ── TRAIN MODELS ──
print("Training models...")

# Match result
xgb_result = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='mlogloss'
)
xgb_result.fit(
    train[result_features],
    train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
)

# Corners
poisson_hc = PoissonRegressor(max_iter=300)
poisson_hc.fit(train[corner_features], train['HC'])

poisson_ac = PoissonRegressor(max_iter=300)
poisson_ac.fit(train[corner_features], train['AC'])

# Goals
poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(train[goal_features], train['FTHG'])

poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(train[goal_features], train['FTAG'])

print("Models trained successfully")
print(f"Trained on {len(train)} matches")

# ── FUNCTION TO GET TEAM'S CURRENT RATINGS ──
def get_team_ratings(team, is_home):
    if is_home:
        matches = df[df['HomeTeam'] == team].tail(1)
        if len(matches) == 0:
            print(f"Team {team} not found")
            return None
        return {
            'attack':           matches['home_attack'].values[0],
            'defence':          matches['home_defence'].values[0],
            'corners_for':      matches['home_corners_for'].values[0],
            'corners_against':  matches['home_corners_against'].values[0],
            'shots_for':        matches['home_shots_for'].values[0],
            'form':             matches['home_form'].values[0],
        }
    else:
        matches = df[df['AwayTeam'] == team].tail(1)
        if len(matches) == 0:
            print(f"Team {team} not found")
            return None
        return {
            'attack':           matches['away_attack'].values[0],
            'defence':          matches['away_defence'].values[0],
            'corners_for':      matches['away_corners_for'].values[0],
            'corners_against':  matches['away_corners_against'].values[0],
            'shots_for':        matches['away_shots_for'].values[0],
            'form':             matches['away_form'].values[0],
        }

# ── MAIN PREDICTION FUNCTION ──
def predict_match(home_team, away_team, referee_avg_cards=3.75):

    # Get current ratings for both teams
    home = get_team_ratings(home_team, is_home=True)
    away = get_team_ratings(away_team, is_home=False)

    if home is None or away is None:
        return

    # Check if post 2024 transition
    post_2024 = 1

    # ── BUILD FEATURE ROW ──
    result_row = pd.DataFrame([{
        'home_attack':       home['attack'],
        'home_defence':      home['defence'],
        'away_attack':       away['attack'],
        'away_defence':      away['defence'],
        'home_corners_for':  home['corners_for'],
        'away_corners_for':  away['corners_for'],
        'home_form':         home['form'],
        'away_form':         away['form'],
        'attack_diff':       home['attack'] - away['attack'],
        'defence_diff':      home['defence'] - away['defence'],
        'post_2024_transition': post_2024
    }])

    corner_row = pd.DataFrame([{
        'home_corners_for':      home['corners_for'],
        'home_corners_against':  home['corners_against'],
        'away_corners_for':      away['corners_for'],
        'away_corners_against':  away['corners_against'],
        'post_2024_transition':  post_2024
    }])

    goal_row = pd.DataFrame([{
        'home_attack':           home['attack'],
        'home_defence':          home['defence'],
        'away_attack':           away['attack'],
        'away_defence':          away['defence'],
        'home_shots_for':        home['shots_for'],
        'away_shots_for':        away['shots_for'],
        'post_2024_transition':  post_2024
    }])

    # ── PREDICTIONS ──
    # Match result probabilities
    result_probs = xgb_result.predict_proba(result_row)[0]
    prob_home = result_probs[0]
    prob_draw = result_probs[1]
    prob_away = result_probs[2]

    # Goals
    pred_home_goals = poisson_home.predict(goal_row)[0]
    pred_away_goals = poisson_away.predict(goal_row)[0]
    pred_total_goals = pred_home_goals + pred_away_goals

    # Corners
    pred_home_corners = poisson_hc.predict(corner_row)[0]
    pred_away_corners = poisson_ac.predict(corner_row)[0]
    pred_total_corners = pred_home_corners + pred_away_corners

    # Over 2.5 probability (from Poisson distribution)
    import scipy.stats as stats
    over_2_5_prob = 1 - stats.poisson.cdf(2, pred_total_goals)
    btts_prob = (1 - stats.poisson.cdf(0, pred_home_goals)) * \
                (1 - stats.poisson.cdf(0, pred_away_goals))

    # Confidence flag
    max_prob = max(prob_home, prob_away)
    if max_prob >= 0.75:
        confidence = "HIGH"
    elif max_prob >= 0.65:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── OUTPUT ──
    print(f"\n{'='*50}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*50}")
    print(f"\n  MATCH RESULT:")
    print(f"  {home_team} Win:  {prob_home:.1%}")
    print(f"  Draw:           {prob_draw:.1%}")
    print(f"  {away_team} Win: {prob_away:.1%}")
    print(f"  Confidence:     {confidence}")
    print(f"\n  GOALS:")
    print(f"  Expected:       {pred_total_goals:.1f}")
    print(f"  Over 2.5:       {over_2_5_prob:.1%}")
    print(f"  BTTS:           {btts_prob:.1%}")
    print(f"  Score:          {home_team} {pred_home_goals:.1f} — {pred_away_goals:.1f} {away_team}")
    print(f"\n  CORNERS:")
    print(f"  Expected total: {pred_total_corners:.1f}")
    print(f"  {home_team}:    {pred_home_corners:.1f}")
    print(f"  {away_team}:    {pred_away_corners:.1f}")
    print(f"{'='*50}\n")

# ── RUN PREDICTIONS ──
# Check available teams first
print("\nAvailable teams:")
print(sorted(df['HomeTeam'].unique()))

# Predict a match
predict_match('Aston Villa', 'Man City')