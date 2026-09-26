import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
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

# ── TRAIN ON ALL DATA ──
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

xgb_result = XGBClassifier(
    n_estimators=200, max_depth=3,
    learning_rate=0.05, random_state=42,
    eval_metric='mlogloss'
)
xgb_result.fit(
    train[result_features],
    train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
)

poisson_hc = PoissonRegressor(max_iter=300)
poisson_hc.fit(train[corner_features], train['HC'])

poisson_ac = PoissonRegressor(max_iter=300)
poisson_ac.fit(train[corner_features], train['AC'])

poisson_home = PoissonRegressor(max_iter=300)
poisson_home.fit(train[goal_features], train['FTHG'])

poisson_away = PoissonRegressor(max_iter=300)
poisson_away.fit(train[goal_features], train['FTAG'])

print("Models trained successfully")
print(f"Trained on {len(train)} matches")

# ══════════════════════════════════════════
# RANGE PREDICTION FUNCTION
# ══════════════════════════════════════════
def predict_with_range(lambda_val, confidence=0.80):
    """
    Given a Poisson lambda returns:
    - Most likely value (mode)
    - Credible interval at given confidence
    - Probability range covers
    """
    mode      = int(np.floor(lambda_val))
    mode_prob = stats.poisson.pmf(mode, lambda_val)

    lower_tail = (1 - confidence) / 2
    upper_tail = 1 - (1 - confidence) / 2

    lower = int(stats.poisson.ppf(lower_tail, lambda_val))
    upper = int(stats.poisson.ppf(upper_tail, lambda_val))

    range_prob = (
        stats.poisson.cdf(upper, lambda_val) -
        stats.poisson.cdf(lower - 1, lambda_val)
    )

    return mode, mode_prob, lower, upper, range_prob


# ══════════════════════════════════════════
# SCORELINE MATRIX FROM POISSON
# ══════════════════════════════════════════
def predict_result_from_poisson(lambda_home, lambda_away, max_goals=10):
    """
    Build full scoreline matrix and derive
    Win/Draw/Loss probabilities
    """
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i][j] = (
                stats.poisson.pmf(i, lambda_home) *
                stats.poisson.pmf(j, lambda_away)
            )

    prob_home = float(np.sum([
        score_matrix[i][j]
        for i in range(max_goals)
        for j in range(max_goals)
        if i > j
    ]))
    prob_draw = float(np.sum([
        score_matrix[i][i]
        for i in range(max_goals)
    ]))
    prob_away = float(np.sum([
        score_matrix[i][j]
        for i in range(max_goals)
        for j in range(max_goals)
        if j > i
    ]))

    most_likely_idx   = np.unravel_index(
        score_matrix.argmax(), score_matrix.shape
    )
    most_likely_score = f"{most_likely_idx[0]}-{most_likely_idx[1]}"
    most_likely_prob  = round(score_matrix[most_likely_idx], 3)

    return {
        'prob_home':         round(prob_home, 3),
        'prob_draw':         round(prob_draw, 3),
        'prob_away':         round(prob_away, 3),
        'most_likely_score': most_likely_score,
        'most_likely_prob':  most_likely_prob
    }


# ══════════════════════════════════════════
# GET TEAM RATINGS
# ══════════════════════════════════════════
def get_team_ratings(team, is_home):
    if is_home:
        matches = df[df['HomeTeam'] == team].tail(1)
        if len(matches) == 0:
            print(f"Team {team} not found")
            return None
        return {
            'attack':          matches['home_attack'].values[0],
            'defence':         matches['home_defence'].values[0],
            'corners_for':     matches['home_corners_for'].values[0],
            'corners_against': matches['home_corners_against'].values[0],
            'shots_for':       matches['home_shots_for'].values[0],
            'form':            matches['home_form'].values[0],
        }
    else:
        matches = df[df['AwayTeam'] == team].tail(1)
        if len(matches) == 0:
            print(f"Team {team} not found")
            return None
        return {
            'attack':          matches['away_attack'].values[0],
            'defence':         matches['away_defence'].values[0],
            'corners_for':     matches['away_corners_for'].values[0],
            'corners_against': matches['away_corners_against'].values[0],
            'shots_for':       matches['away_shots_for'].values[0],
            'form':            matches['away_form'].values[0],
        }


# ══════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════
def predict_match(home_team, away_team):

    home = get_team_ratings(home_team, is_home=True)
    away = get_team_ratings(away_team, is_home=False)

    if home is None or away is None:
        return

    post_2024 = 1

    # ── FEATURE ROWS ──
    result_row = pd.DataFrame([{
        'home_attack':          home['attack'],
        'home_defence':         home['defence'],
        'away_attack':          away['attack'],
        'away_defence':         away['defence'],
        'home_corners_for':     home['corners_for'],
        'away_corners_for':     away['corners_for'],
        'home_form':            home['form'],
        'away_form':            away['form'],
        'attack_diff':          home['attack'] - away['attack'],
        'defence_diff':         home['defence'] - away['defence'],
        'post_2024_transition': post_2024
    }])

    corner_row = pd.DataFrame([{
        'home_corners_for':     home['corners_for'],
        'home_corners_against': home['corners_against'],
        'away_corners_for':     away['corners_for'],
        'away_corners_against': away['corners_against'],
        'post_2024_transition': post_2024
    }])

    goal_row = pd.DataFrame([{
        'home_attack':          home['attack'],
        'home_defence':         home['defence'],
        'away_attack':          away['attack'],
        'away_defence':         away['defence'],
        'home_shots_for':       home['shots_for'],
        'away_shots_for':       away['shots_for'],
        'post_2024_transition': post_2024
    }])

    # ── PREDICT LAMBDAS ──
    lambda_home_goals   = poisson_home.predict(goal_row)[0]
    lambda_away_goals   = poisson_away.predict(goal_row)[0]
    lambda_total_goals  = lambda_home_goals + lambda_away_goals

    lambda_home_corners  = poisson_hc.predict(corner_row)[0]
    lambda_away_corners  = poisson_ac.predict(corner_row)[0]
    lambda_total_corners = lambda_home_corners + lambda_away_corners

    # ── RANGES FROM POISSON ──
    g_mode, g_mode_prob, g_lower, g_upper, g_range_prob = \
        predict_with_range(lambda_total_goals, 0.80)

    c_mode, c_mode_prob, c_lower, c_upper, c_range_prob = \
        predict_with_range(lambda_total_corners, 0.80)

    # ── MARKET PROBABILITIES ──
    over_2_5  = 1 - stats.poisson.cdf(2, lambda_total_goals)
    over_3_5  = 1 - stats.poisson.cdf(3, lambda_total_goals)
    under_2_5 = stats.poisson.cdf(2, lambda_total_goals)
    under_4_5 = stats.poisson.cdf(4, lambda_total_goals)

    btts = (
        (1 - stats.poisson.cdf(0, lambda_home_goals)) *
        (1 - stats.poisson.cdf(0, lambda_away_goals))
    )

    over_9_5_corners  = 1 - stats.poisson.cdf(9,  lambda_total_corners)
    over_10_5_corners = 1 - stats.poisson.cdf(10, lambda_total_corners)
    under_10_5_corners = stats.poisson.cdf(10, lambda_total_corners)

    # ── RESULT FROM POISSON SCORELINE MATRIX ──
    poisson_result = predict_result_from_poisson(
        lambda_home_goals, lambda_away_goals
    )

    # ── RESULT FROM XGBOOST ──
    xgb_probs = xgb_result.predict_proba(result_row)[0]

    # ── ENSEMBLE — AVERAGE BOTH MODELS ──
    prob_home = round((poisson_result['prob_home'] + xgb_probs[0]) / 2, 3)
    prob_draw = round((poisson_result['prob_draw'] + xgb_probs[1]) / 2, 3)
    prob_away = round((poisson_result['prob_away'] + xgb_probs[2]) / 2, 3)

    # ── CONFIDENCE FLAG ──
    max_prob = max(prob_home, prob_away)
    if max_prob >= 0.75:
        confidence = "HIGH   ✅"
    elif max_prob >= 0.65:
        confidence = "MEDIUM ⚠️"
    else:
        confidence = "LOW    ❌"

    # ── OUTPUT ──
    print(f"\n{'='*55}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*55}")

    print(f"\n  MATCH RESULT (Ensemble: Poisson + XGBoost):")
    print(f"  {home_team} Win:    {prob_home:.1%}")
    print(f"  Draw:               {prob_draw:.1%}")
    print(f"  {away_team} Win:    {prob_away:.1%}")
    print(f"  Confidence:         {confidence}")
    print(f"  Most likely score:  {poisson_result['most_likely_score']}"
          f"  ({poisson_result['most_likely_prob']:.1%})")

    print(f"\n  [Poisson only]  H{poisson_result['prob_home']:.1%}"
          f"  D{poisson_result['prob_draw']:.1%}"
          f"  A{poisson_result['prob_away']:.1%}")
    print(f"  [XGBoost only]  H{xgb_probs[0]:.1%}"
          f"  D{xgb_probs[1]:.1%}"
          f"  A{xgb_probs[2]:.1%}")

    print(f"\n  GOALS:")
    print(f"  Expected (λ):       {lambda_total_goals:.2f}")
    print(f"  Most likely:        {g_mode} goals"
          f"  ({g_mode_prob:.1%} probability)")
    print(f"  80% range:          {g_lower} — {g_upper} goals"
          f"  ({g_range_prob:.1%} confidence)")
    print(f"  Over 2.5:           {over_2_5:.1%}")
    print(f"  Under 2.5:          {under_2_5:.1%}")
    print(f"  Over 3.5:           {over_3_5:.1%}")
    print(f"  Under 4.5:          {under_4_5:.1%}")
    print(f"  BTTS:               {btts:.1%}")
    print(f"  {home_team} goals:  {lambda_home_goals:.2f} expected")
    print(f"  {away_team} goals:  {lambda_away_goals:.2f} expected")

    print(f"\n  CORNERS:")
    print(f"  Expected (λ):       {lambda_total_corners:.2f}")
    print(f"  Most likely:        {c_mode} corners"
          f"  ({c_mode_prob:.1%} probability)")
    print(f"  80% range:          {c_lower} — {c_upper} corners"
          f"  ({c_range_prob:.1%} confidence)")
    print(f"  Over 9.5:           {over_9_5_corners:.1%}")
    print(f"  Over 10.5:          {over_10_5_corners:.1%}")
    print(f"  Under 10.5:         {under_10_5_corners:.1%}")
    print(f"  {home_team}:        {lambda_home_corners:.1f} expected")
    print(f"  {away_team}:        {lambda_away_corners:.1f} expected")

    print(f"{'='*55}\n")


# ── RUN PREDICTIONS ──
print("\nAvailable teams:")
print(sorted(df['HomeTeam'].unique()))

# ── CHANGE TEAMS HERE ──
predict_match('Aston Villa', 'Man City')