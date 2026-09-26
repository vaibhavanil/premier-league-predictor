import pandas as pd

df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\combined.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

df['total_cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']

# ══════════════════════════════════════════
# STEP 1 — RAW EWMA RATINGS (base features)
# ══════════════════════════════════════════

# Goals
df['home_goals_for'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['home_goals_against'] = (
    df.groupby('HomeTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_goals_for'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_goals_against'] = (
    df.groupby('AwayTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# Corners
df['home_corners_for'] = (
    df.groupby('HomeTeam')['HC']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['home_corners_against'] = (
    df.groupby('HomeTeam')['AC']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_corners_for'] = (
    df.groupby('AwayTeam')['AC']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_corners_against'] = (
    df.groupby('AwayTeam')['HC']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# Shots
df['home_shots_for'] = (
    df.groupby('HomeTeam')['HS']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['home_shots_against'] = (
    df.groupby('HomeTeam')['AS']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_shots_for'] = (
    df.groupby('AwayTeam')['AS']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_shots_against'] = (
    df.groupby('AwayTeam')['HS']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# Cards
df['home_cards_for'] = (
    df.groupby('HomeTeam')['total_cards']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)
df['away_cards_for'] = (
    df.groupby('AwayTeam')['total_cards']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# Form
df['home_form'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=3).mean())
)
df['away_form'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=3).mean())
)

# ══════════════════════════════════════════
# STEP 2 — OPPONENT ADJUSTED RATINGS
# ══════════════════════════════════════════
# Formula: team_rating / opponent_defence_rating
# A team scoring 2 goals vs strong defence
# is rated higher than 2 goals vs weak defence

# Goals adjusted
df['home_attack_adj'] = (
    df['home_goals_for'] /
    df['away_goals_against'].replace(0, 0.1)
)
df['away_attack_adj'] = (
    df['away_goals_for'] /
    df['home_goals_against'].replace(0, 0.1)
)
df['home_defence_adj'] = (
    df['home_goals_against'] /
    df['away_goals_for'].replace(0, 0.1)
)
df['away_defence_adj'] = (
    df['away_goals_against'] /
    df['home_goals_for'].replace(0, 0.1)
)

# Corners adjusted
df['home_corners_adj'] = (
    df['home_corners_for'] /
    df['away_corners_against'].replace(0, 0.1)
)
df['away_corners_adj'] = (
    df['away_corners_for'] /
    df['home_corners_against'].replace(0, 0.1)
)

# Shots adjusted
df['home_shots_adj'] = (
    df['home_shots_for'] /
    df['away_shots_against'].replace(0, 0.1)
)
df['away_shots_adj'] = (
    df['away_shots_for'] /
    df['home_shots_against'].replace(0, 0.1)
)

# Cards adjusted
df['home_cards_adj'] = (
    df['home_cards_for'] /
    df['away_cards_for'].replace(0, 0.1)
)
df['away_cards_adj'] = (
    df['away_cards_for'] /
    df['home_cards_for'].replace(0, 0.1)
)

# ══════════════════════════════════════════
# STEP 3 — DIFFERENCE FEATURES
# ══════════════════════════════════════════
df['attack_diff']  = df['home_attack_adj'] - df['away_attack_adj']
df['defence_diff'] = df['home_defence_adj'] - df['away_defence_adj']
df['corner_diff']  = df['home_corners_adj'] - df['away_corners_adj']
df['shot_diff']    = df['home_shots_adj'] - df['away_shots_adj']

# ══════════════════════════════════════════
# STEP 4 — REFEREE FEATURE
# ══════════════════════════════════════════
referee_avg = df.groupby('Referee')['total_cards'].mean().round(2)
referee_avg.name = 'ref_avg_cards'
df = df.merge(referee_avg, on='Referee', how='left')

# ══════════════════════════════════════════
# STEP 5 — TRANSITION FLAG
# ══════════════════════════════════════════
df['post_2024_transition'] = (df['Date'] >= '2024-08-01').astype(int)

# ══════════════════════════════════════════
# STEP 6 — CHECK SAMPLE
# ══════════════════════════════════════════
print("Sample opponent adjusted ratings:")
print(df[['Date', 'HomeTeam', 'AwayTeam',
          'home_attack_adj', 'away_attack_adj',
          'home_corners_adj', 'away_corners_adj',
          'home_shots_adj', 'away_shots_adj']].iloc[60:70])

# ══════════════════════════════════════════
# STEP 7 — DROP NaN AND SAVE
# ══════════════════════════════════════════
df_model = df.dropna(subset=[
    'home_attack_adj', 'away_attack_adj',
    'home_defence_adj', 'away_defence_adj',
    'home_corners_adj', 'away_corners_adj',
    'home_shots_adj', 'away_shots_adj',
    'home_form', 'away_form',
    'ref_avg_cards'
])

print("\nRows before dropping NaN:", len(df))
print("Rows after dropping NaN:", len(df_model))
print("Final dataset shape:", df_model.shape)

df_model.to_csv(
    r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered_adj.csv',
    index=False
)
print("Saved as engineered_adj.csv")