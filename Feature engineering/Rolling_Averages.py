import pandas as pd

df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\combined.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

# ── HOME ATTACK RATING ──
# For each match, average goals scored at home in last 10 home games
df['home_attack'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# ── HOME DEFENCE RATING ──
# For each match, average goals conceded at home in last 10 home games
df['home_defence'] = (
    df.groupby('HomeTeam')['FTAG']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# ── AWAY ATTACK RATING ──
# For each match, average goals scored away in last 10 away games
df['away_attack'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# ── AWAY DEFENCE RATING ──
# For each match, average goals conceded away in last 10 away games
df['away_defence'] = (
    df.groupby('AwayTeam')['FTHG']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

print(df[['Date', 'HomeTeam', 'AwayTeam', 
          'home_attack', 'home_defence', 
          'away_attack', 'away_defence']].head(20))

print(df[['Date', 'HomeTeam', 'AwayTeam',
          'home_attack', 'home_defence',
          'away_attack', 'away_defence']].iloc[30:50])

city_matches = df[(df['HomeTeam'] == 'Man City') | 
                  (df['AwayTeam'] == 'Man City')]

print(city_matches[['Date', 'HomeTeam', 'AwayTeam',
                     'home_attack', 'home_defence',
                     'away_attack', 'away_defence']].head(20))

# Check row 198 — Man City vs Sheffield United Jan 2021
# home_attack should be average of City's last 10 home goals

city_home_games = df[df['HomeTeam'] == 'Man City'].copy()
city_home_games = city_home_games[city_home_games['Date'] < '2021-01-30']

print("City home games before Jan 30 2021:")
print(city_home_games[['Date', 'FTHG', 'home_attack']].tail(12))

# ── CORNER RATINGS ──
df['home_corners_for'] = (
    df.groupby('HomeTeam')['HC']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['home_corners_against'] = (
    df.groupby('HomeTeam')['AC']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['away_corners_for'] = (
    df.groupby('AwayTeam')['AC']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['away_corners_against'] = (
    df.groupby('AwayTeam')['HC']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# ── SHOT RATINGS ──
df['home_shots_for'] = (
    df.groupby('HomeTeam')['HS']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['home_shots_against'] = (
    df.groupby('HomeTeam')['AS']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['away_shots_for'] = (
    df.groupby('AwayTeam')['AS']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['away_shots_against'] = (
    df.groupby('AwayTeam')['HS']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)
 
df['total_cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']

# ── CARD RATINGS ──
df['home_cards'] = (
    df.groupby('HomeTeam')['total_cards']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

df['away_cards'] = (
    df.groupby('AwayTeam')['total_cards']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Check all new ratings
print(df[['Date', 'HomeTeam', 'AwayTeam',
          'home_corners_for', 'away_corners_for',
          'home_shots_for', 'away_shots_for',
          'home_cards', 'away_cards']].iloc[55:65])

# ── FORM FEATURE — points from last 5 games ──
# Win = 3, Draw = 1, Loss = 0

def get_points(result, team, home_col, away_col):
    if result == 'H':
        return 3 if team == home_col else 0
    elif result == 'A':
        return 3 if team == away_col else 0
    else:
        return 1

# Home team points per game rolling 5
df['home_form'] = (
    df.groupby('HomeTeam')['FTHG']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
)

# Away team points per game rolling 5
df['away_form'] = (
    df.groupby('AwayTeam')['FTAG']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
)

# ── REFEREE FEATURE ──
referee_avg = df.groupby('Referee')['total_cards'].mean().round(2)
referee_avg.name = 'ref_avg_cards'
df = df.merge(referee_avg, on='Referee', how='left')

# ── CHECK ──
print(df[['Date', 'HomeTeam', 'AwayTeam',
          'home_form', 'away_form',
          'ref_avg_cards']].iloc[55:65])

# Drop rows where core features are NaN
# These are early season matches with no history
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

# Save engineered dataset
df_model.to_csv(
    r'C:\Users\anilv\Desktop\Football prediction ML\Data\engineered.csv',
    index=False
)

print("Saved successfully")
print("Final dataset shape:", df_model.shape)

