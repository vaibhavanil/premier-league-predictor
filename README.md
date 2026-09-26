# Premier League Match Prediction Model

A machine learning pipeline to predict Premier League match outcomes across multiple markets using 6 seasons of historical data (2,238 matches).

## What It Predicts
- Match result (Win/Draw/Loss) with probability outputs
- Total goals — expected value and 80% confidence range
- Total corners — expected value and 80% confidence range
- Total cards — expected value and Over/Under 3.5
- Over/Under 2.5 goals
- Both Teams to Score (BTTS)

## Key Results

| Market | Baseline | Model | Improvement |
|--------|----------|-------|-------------|
| Match Result | 41.3% | 47.6% | +13.7% |
| Match Result (0.65 threshold) | 41.3% | 61.4% | +20.1% |
| Over 3.5 Cards | 51.1% | 58.4% | +14.3% |
| Corners MAE | 2.710 | 2.704 | +0.2% |

## Technical Approach

### Feature Engineering
- EWMA rolling ratings (span=10) for attack, defence, corners and shots for every team
- Referee tendency feature — historical cards per game per referee
- Squad transition period flag for post-2024 Man City decline
- Temporal train/test split to prevent data leakage

### Models
- **Poisson Regression** for count data (goals, corners, cards) — predicts lambda, then derives full probability distribution
- **XGBoost Classifier** for match result classification — captures non-linear feature interactions

### Key Finding
Model accuracy increases significantly with confidence threshold:

| Threshold | Accuracy |
|-----------|----------|
| No threshold | 47.6% |
| 0.55 | 57.7% |
| 0.65 | 61.4% |
| 0.70 | 61.9% |

The model knows when it knows.

## Data
Data sourced from [football-data.co.uk](https://football-data.co.uk) — free CSV downloads covering Premier League seasons. Not included in this repo due to size.

Download the Premier League season files and place them in `data/`.

## Project Structure
```
premier-league-predictor/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          ← data source instructions
├── Feature Engineering/
│   └── eda_ewma.py        ← feature engineering pipeline
└── Model Building/
    ├── model.py           ← model training and evaluation
    └── predict.py         ← live prediction pipeline
```

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Feature engineering
```bash
python "Feature Engineering/eda_ewma.py"
```

### Train and evaluate models
```bash
python "Model Building/model.py"
```

### Live prediction
```bash
python "Model Building/predict.py"
```

## Example Output
```
==================================================
  Arsenal vs Chelsea
==================================================
  MATCH RESULT:
  Arsenal Win:    62.0%
  Draw:           22.0%
  Chelsea Win:    16.0%
  Confidence:     MEDIUM

  GOALS:
  Expected:       2.7
  80% range:      1 — 5 goals
  Over 2.5:       54.2%
  BTTS:           56.1%

  CORNERS:
  Expected:       10.1
  80% range:      7 — 13 corners
==================================================
```

## What I Would Do Next
- Add xG data from understat.com (biggest improvement)
- Implement Dixon-Coles correction for low-scoring matches
- Add more leagues (La Liga, Bundesliga)
- Deploy as a Flask API
- Add opponent strength adjustment to ratings

## Author
Vaibhav Anil — [LinkedIn](https://linkedin.com/in/vaibhav-anil)
