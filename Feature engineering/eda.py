import pandas as pd

# Load combined dataset
df = pd.read_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\combined.csv')

# Basic inspection
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic stats:\n", df.describe())

# Goals
df['total_goals'] = df['FTHG'] + df['FTAG']
df['total_corners'] = df['HC'] + df['AC']
df['total_cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']
df['total_shots'] = df['HS'] + df['AS']
df['total_sot'] = df['HST'] + df['AST']
df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)

# Five key questions
print("Avg total goals:", df['total_goals'].mean())
print("Avg total corners:", df['total_corners'].mean())
print("Avg total cards:", df['total_cards'].mean())
print("Avg total shots:", df['total_shots'].mean())

print("\nHome win rate:\n", df['FTR'].value_counts(normalize=True))
print("\nBTTS rate:", df['btts'].mean())
print("Over 2.5 rate:", df['over_2_5'].mean())

# Convert date and extract season
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Season'] = df['Date'].dt.year

print(df.groupby('Season').agg(
    avg_goals=('total_goals', 'mean'),
    avg_corners=('total_corners', 'mean'),
    avg_cards=('total_cards', 'mean'),
    home_win_rate=('FTR', lambda x: (x == 'H').mean())
).round(2))

# Filter Man City matches
city = df[(df['HomeTeam'] == 'Man City') | 
          (df['AwayTeam'] == 'Man City')]

print("City matches:", len(city))

# City home stats
city_home = df[df['HomeTeam'] == 'Man City']
city_away = df[df['AwayTeam'] == 'Man City']

print("\nCity HOME averages:")
print("Goals scored:", city_home['FTHG'].mean().round(2))
print("Goals conceded:", city_home['FTAG'].mean().round(2))
print("Shots:", city_home['HS'].mean().round(2))
print("Corners:", city_home['HC'].mean().round(2))

print("\nCity AWAY averages:")
print("Goals scored:", city_away['FTAG'].mean().round(2))
print("Goals conceded:", city_away['FTHG'].mean().round(2))
print("Shots:", city_away['AS'].mean().round(2))
print("Corners:", city_away['AC'].mean().round(2))



# Filter Man City matches
city = df[(df['HomeTeam'] == 'Man City') | 
          (df['AwayTeam'] == 'Man City')]

print("City matches:", len(city))

# City home stats
city_home = df[df['HomeTeam'] == 'Man City']
city_away = df[df['AwayTeam'] == 'Man City']

print("\nCity HOME averages:")
print("Goals scored:", city_home['FTHG'].mean().round(2))
print("Goals conceded:", city_home['FTAG'].mean().round(2))
print("Shots:", city_home['HS'].mean().round(2))
print("Corners:", city_home['HC'].mean().round(2))

print("\nCity AWAY averages:")
print("Goals scored:", city_away['FTAG'].mean().round(2))
print("Goals conceded:", city_away['FTHG'].mean().round(2))
print("Shots:", city_away['AS'].mean().round(2))
print("Corners:", city_away['AC'].mean().round(2))

# City home results
print("City HOME results:")
print(city_home['FTR'].value_counts(normalize=True).round(2))

# City away results
print("\nCity AWAY results:")
# When City are away, a win for City = 'A' (Away win)
away_results = city_away['FTR'].value_counts(normalize=True).round(2)
print(away_results)

# City clean sheets
home_cs = (city_home['FTAG'] == 0).mean().round(2)
away_cs = (city_away['FTHG'] == 0).mean().round(2)
print("\nCity HOME clean sheet rate:", home_cs)
print("City AWAY clean sheet rate:", away_cs)

# City over 2.5 goals rate
city_home_o25 = (city_home['total_goals'] > 2.5).mean().round(2)
city_away_o25 = (city_away['total_goals'] > 2.5).mean().round(2)
print("\nCity HOME over 2.5 rate:", city_home_o25)
print("City AWAY over 2.5 rate:", city_away_o25)

# How does City perform against different opponents
# Group by opponent and calculate City's stats

# City as home team vs each opponent
city_home_vs = city_home.groupby('AwayTeam').agg(
    matches=('FTHG', 'count'),
    city_goals=('FTHG', 'mean'),
    opp_goals=('FTAG', 'mean'),
    city_wins=('FTR', lambda x: (x == 'H').mean())
).round(2).sort_values('city_wins')

print("City HOME record vs each opponent:")
print(city_home_vs)

# City AWAY record vs each opponent
city_away_vs = city_away.groupby('HomeTeam').agg(
    matches=('FTAG', 'count'),
    city_goals=('FTAG', 'mean'),
    opp_goals=('FTHG', 'mean'),
    city_wins=('FTR', lambda x: (x == 'A').mean())
).round(2).sort_values('city_wins')

print("City AWAY record vs each opponent:")
print(city_away_vs)


# Add transition period flag
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['post_2024_transition'] = (df['Date'] >= '2024-08-01').astype(int)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['post_2024_transition'] = (df['Date'] >= '2024-08-01').astype(int)

# Filter City matches from the MAIN df, not from city_pre
city_all  = df[(df['HomeTeam'] == 'Man City') | (df['AwayTeam'] == 'Man City')]
city_pre  = city_all[city_all['post_2024_transition'] == 0]
city_post = city_all[city_all['post_2024_transition'] == 1]

print("City BEFORE 2024 — matches:", len(city_pre))
print("City AFTER 2024 — matches:", len(city_post))

print("\nCity BEFORE 2024 goals per game:", round(
    city_pre['FTHG'].where(city_pre['HomeTeam'] == 'Man City', 
    city_pre['FTAG']).mean(), 2))

print("City AFTER 2024 goals per game:", round(
    city_post['FTHG'].where(city_post['HomeTeam'] == 'Man City', 
    city_post['FTAG']).mean(), 2))

print("City BEFORE 2024:")
print("Goals scored:  ", round(city_pre['FTHG'].where(city_pre['HomeTeam'] == 'Man City', city_pre['FTAG']).mean(), 2))
print("Goals conceded:", round(city_pre['FTAG'].where(city_pre['HomeTeam'] == 'Man City', city_pre['FTHG']).mean(), 2))
print("Shots:         ", round(city_pre['HS'].where(city_pre['HomeTeam'] == 'Man City', city_pre['AS']).mean(), 2))
print("Corners:       ", round(city_pre['HC'].where(city_pre['HomeTeam'] == 'Man City', city_pre['AC']).mean(), 2))
print("Win rate:      ", round((city_pre['FTR'].where(city_pre['HomeTeam'] == 'Man City', city_pre['FTR'].map({'H':'A','A':'H','D':'D'})) == 'H').mean(), 2))

print("\nCity AFTER 2024:")
print("Goals scored:  ", round(city_post['FTHG'].where(city_post['HomeTeam'] == 'Man City', city_post['FTAG']).mean(), 2))
print("Goals conceded:", round(city_post['FTAG'].where(city_post['HomeTeam'] == 'Man City', city_post['FTHG']).mean(), 2))
print("Shots:         ", round(city_post['HS'].where(city_post['HomeTeam'] == 'Man City', city_post['AS']).mean(), 2))
print("Corners:       ", round(city_post['HC'].where(city_post['HomeTeam'] == 'Man City', city_post['AC']).mean(), 2))
print("Win rate:      ", round((city_post['FTR'].where(city_post['HomeTeam'] == 'Man City', city_post['FTR'].map({'H':'A','A':'H','D':'D'})) == 'H').mean(), 2))

# Referee impact on cards
referee_stats = df.groupby('Referee').agg(
    matches=('FTR', 'count'),
    avg_cards=('total_cards', 'mean'),
    avg_goals=('total_goals', 'mean'),
    avg_corners=('total_corners', 'mean')
).round(2)

# Only referees with 20+ games to avoid small sample noise
referee_stats = referee_stats[referee_stats['matches'] >= 20]
referee_stats = referee_stats.sort_values('avg_cards', ascending=False)

print(referee_stats)