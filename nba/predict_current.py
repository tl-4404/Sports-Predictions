import os
import torch
import torch.nn as nn
import pandas as pd
import joblib
from datetime import date
from fetch_current import build_feature_rows_for_date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model
state = torch.load(os.path.join(BASE_DIR, "models/nba_model.pt"), map_location="cpu")
# check for number of input features
in_features = state["out.0.weight"].shape[1]
print("CHECKPOINT input_dim:", in_features)

class ml_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,1)
        )
    
    def forward(self, x):
        return self.out(x)

# create model with saved weights
model = ml_Model(input_dim=in_features)
model.load_state_dict(state)
# set to eval mode for predictions
model.eval()
# check for same number of features inputted as trained on features
feature_cols = pd.read_csv(os.path.join(BASE_DIR, "models/feature_columns.csv"), header=None)[0].tolist()
print("PREDICT feature count (from CSV):", len(feature_cols))
feature_cols = [c.strip() for c in feature_cols]

if len(feature_cols) != in_features:
    raise RuntimeError(
        f"feature_columns.csv has {len(feature_cols)} columns, "
        f"but checkpoint expects {in_features}. "
        "Delete models/, retrain train_model.py so they match."
    )

# load same normalizing stats
scaler = joblib.load(os.path.join(BASE_DIR, "models/nba_scaler.joblib"))
# get todays teams playing stats
df_today = build_feature_rows_for_date(date.today())
df_today.columns = df_today.columns.str.strip()
# check for no features
if df_today.empty:
    print("No games today or failed to fetch.")
    raise SystemExit
# build differential features to match training (numeric cols only)
numeric_cols = df_today.select_dtypes(include="number").columns
home_stats = {c[len("home_"):]: c for c in numeric_cols if c.startswith("home_")}
away_stats = {c[len("away_"):]: c for c in numeric_cols if c.startswith("away_")}
shared = sorted(home_stats.keys() & away_stats.keys())
diff_df = pd.DataFrame(
    {f"diff_{stat}": df_today[home_stats[stat]].values - df_today[away_stats[stat]].values for stat in shared},
    index=df_today.index
)
df_today = pd.concat([df_today, diff_df], axis=1)

# check for all feature cols present
missing = [c for c in feature_cols if c not in df_today.columns]
if missing:
    raise RuntimeError(f"df_today is missing training columns: {missing[:10]} ...")
# build df
X_today = df_today[feature_cols]
# normalize features
X_today_scaled = scaler.transform(X_today)
# convert pd df -> tensor
X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32)
# run games through model
with torch.no_grad():
    logits = model(X_today_tensor).squeeze(1)
    probs = torch.sigmoid(logits).numpy()

# save and print out probs
df_today["home_win_prob"] = probs

cols_to_show = [c for c in ["GAME_ID", "home_team", "away_team", "home_win_prob"]
                if c in df_today.columns]
print(df_today[cols_to_show].sort_values("home_win_prob", ascending=False))

df_today.to_csv(os.path.join(BASE_DIR, "current_games_with_probs.csv"), index=False)
print("Saved predictions to current_games_with_probs.csv")


