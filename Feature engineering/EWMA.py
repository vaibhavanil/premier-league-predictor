import pandas as pd

df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\combined.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

df['total_cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']

# ── HOME ATTACK RATING ──
df['home_attack'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# ── HOME DEFENCE RATING ──
df['home_defence'] = (
    df.groupby('HomeTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# ── AWAY ATTACK RATING ──
df['away_attack'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# ── AWAY DEFENCE RATING ──
df['away_defence'] = (
    df.groupby('AwayTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# ── CORNER RATINGS ──
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

# ── SHOT RATINGS ──
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
# Shots on target features
df['home_sot_for'] = (
    df.groupby('HomeTeam')['HST']
    .transform(lambda x: x.shift(1)
    .ewm(span=10, min_periods=3).mean())
)

df['home_sot_against'] = (
    df.groupby('HomeTeam')['AST']
    .transform(lambda x: x.shift(1)
    .ewm(span=10, min_periods=3).mean())
)

df['away_sot_for'] = (
    df.groupby('AwayTeam')['AST']
    .transform(lambda x: x.shift(1)
    .ewm(span=10, min_periods=3).mean())
)

df['away_sot_against'] = (
    df.groupby('AwayTeam')['HST']
    .transform(lambda x: x.shift(1)
    .ewm(span=10, min_periods=3).mean())
)

# SOT ratio — shot quality proxy
df['home_sot_ratio'] = (
    df['home_sot_for'] /
    df['home_shots_for'].replace(0, 0.1)
)

df['away_sot_ratio'] = (
    df['away_sot_for'] /
    df['away_shots_for'].replace(0, 0.1)
)

# ── CARD RATINGS ──
df['home_cards'] = (
    df.groupby('HomeTeam')['total_cards']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

df['away_cards'] = (
    df.groupby('AwayTeam')['total_cards']
    .transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
)

# ── FORM FEATURE ──
# span=5 reacts faster to recent form changes
df['home_form'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=3).mean())
)

df['away_form'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=3).mean())
)

# ── REFEREE FEATURE ──
referee_avg = df.groupby('Referee')['total_cards'].mean().round(2)
referee_avg.name = 'ref_avg_cards'
df = df.merge(referee_avg, on='Referee', how='left')

# ── TRANSITION FLAG ──
df['post_2024_transition'] = (df['Date'] >= '2024-08-01').astype(int)

# ── DROP NaN ROWS ──
df_model = df.dropna(subset=[
    'home_attack', 'home_defence',
    'away_attack', 'away_defence',
    'home_corners_for', 'away_corners_for',
    'home_shots_for', 'away_shots_for',
    'home_form', 'away_form',
    'ref_avg_cards'
])

print("Rows before dropping NaN:", len(df))
print("Rows after dropping NaN:", len(df_model))
print("Final dataset shape:", df_model.shape)

# ── SAVE ──
# At the bottom of your EWMA file
df_model.to_csv(
    r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered_ewma.csv',
    index=False
)
print("Saved successfully")