import pandas as pd
import glob

# Load all seasons at once
files = glob.glob(r'C:\Users\anilv\Desktop\Football prediction ML\Data\*.csv')
dfs = [pd.read_csv(f) for f in files]

# Concatenate into one dataframe
df = pd.concat(dfs, ignore_index=True)

print('All seasons combined:', df.shape)

# Then filter to useful columns
useful_columns = [
    'Date', 'HomeTeam', 'AwayTeam', 'Referee',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY',
    'HR', 'AR', 'HF', 'AF'
]

df = df[useful_columns]

print('After column filter:', df.shape)
print(df.head())
df.to_csv(r'C:\Users\anilv\Desktop\Football prediction ML\Data\combined.csv', index=False)


