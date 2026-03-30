import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import glob

# load training data (prefer combined multi-season file if available)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
combined = os.path.join(BASE_DIR, "cache", "training_data", "nba_training_combined.csv")
data_path = combined if os.path.exists(combined) else os.path.join(BASE_DIR, "cache", "training_data", "nba_training_202324.csv")

df = pd.read_csv(data_path)
if df.empty:
    raise ValueError("Loaded CSV files, but the combined dataframe is empty.")
print(f"Loaded data from {df} with {len(df)} rows.")

# drop columns not needed for training
columns_to_drop = ["GAME_DATE", "home_team", "away_team", "home_wl", "away_wl"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

numeric_df = df.select_dtypes(include=["number"])
required_cols = ["home_win"]
missing_required = [col for col in required_cols if col not in numeric_df.columns]
if missing_required:
    raise KeyError(f"Missing required training column(s): {missing_required}")

# split into training df, X, and results, y
X = numeric_df.drop(columns=[col for col in ["home_win", "GAME_ID"] if col in numeric_df.columns])
y = numeric_df["home_win"].astype(float)
if X.empty:
    raise ValueError("No numeric feature columns found after dropping target/id columns.")

# drop redundant features
redundant = []
for prefix in ["home_", "away_"]:
    redundant += [
        # rank cols - redundant with the stats themselves
        *[c for c in X.columns if c.startswith(prefix) and "_RANK" in c],
        # duplicate base stats across advanced/misc/fourfactors/opponent tables
        *[c for c in X.columns if re.search(rf"^{prefix}(GP|W|L|W_PCT|MIN)_(advanced|misc|fourfactors|opponent)$", c)],
        # raw counts when percentage is available
        prefix + "W",       # keep W_PCT
        prefix + "L",       # derivable from GP - W
        prefix + "GP",      # keep W_PCT
        prefix + "MIN",     # all games same length
        prefix + "FGM",     prefix + "FGA",    # keep FG_PCT
        prefix + "FG3M",    prefix + "FG3A",   # keep FG3_PCT
        prefix + "FTM",     prefix + "FTA",    # keep FT_PCT
        prefix + "OPP_FGM", prefix + "OPP_FGA",   # keep OPP_FG_PCT
        prefix + "OPP_FG3M",prefix + "OPP_FG3A",  # keep OPP_FG3_PCT
        prefix + "OPP_FTM", prefix + "OPP_FTA",   # keep OPP_FT_PCT
        prefix + "OPP_REB",                        # keep OPP_OREB/OPP_DREB
        # derived stats redundant with components
        prefix + "NET_RATING",      # = OFF_RATING - DEF_RATING
        prefix + "REB_PCT",         # = OREB_PCT + DREB_PCT
        prefix + "AST_TO",          # keep AST_PCT
        prefix + "AST_RATIO",       # keep AST_PCT
        prefix + "EFG_PCT",         # keep TS_PCT (more complete)
        prefix + "POSS",            # keep PACE
        prefix + "E_OFF_RATING", prefix + "E_DEF_RATING", prefix + "E_NET_RATING",
        prefix + "PACE_PER40",
        prefix + "BLKA",
        prefix + "PLUS_MINUS",
        prefix + "PLUS_MINUS_opponent",
        prefix + "EFG_PCT_fourfactors",
        prefix + "TM_TOV_PCT_fourfactors",
        prefix + "OREB_PCT_fourfactors",
    ]

X = X.drop(columns=[c for c in set(redundant) if c in X.columns])

# build differential features: home_stat - away_stat for each matched pair
home_stats = {c[len("home_"):]: c for c in X.columns if c.startswith("home_")}
away_stats = {c[len("away_"):]: c for c in X.columns if c.startswith("away_")}
shared = sorted(home_stats.keys() & away_stats.keys())

diff_df = pd.DataFrame(
    {f"diff_{stat}": X[home_stats[stat]].values - X[away_stats[stat]].values for stat in shared},
    index=X.index
)

# drop paired home/away cols, keep unpaired ones and diffs
paired = set(home_stats[s] for s in shared) | set(away_stats[s] for s in shared)
X = pd.concat([diff_df, X[[c for c in X.columns if c not in paired]].reset_index(drop=True)], axis=1)

print("TRAIN feature count:", X.shape[1])
print("TRAIN columns:", list(X.columns)[:10], "...")
print("TARGET distribution:", y.value_counts(dropna=False).to_dict())

feature_cols = list(X.columns)

# train/test split (80/20)
X_tr, X_val, y_tr, y_val = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# standardize features (fit on train only)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# convert pd df -> tensors
X_train = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_val_scaled, dtype=torch.float32)
y_test = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# training model
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
    
model = ml_Model(input_dim=X_train.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# train model with early stopping
best_val_loss = float("inf")
patience = 10
epochs_no_improve = 0
best_state = None

for epoch in range(200):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test), y_test).item()

    print(f"Epoch {epoch+1}: train={avg_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
            break

model.load_state_dict(best_state)

# evaluate on test set
model.eval()
with torch.no_grad():
    val_preds = model(X_test)
    val_probs = torch.sigmoid(val_preds)
    val_labels = (val_probs >= 0.5).float()
    val_acc = (val_labels == y_test).float().mean().item()
    train_preds = model(X_train)
    train_probs = torch.sigmoid(train_preds)
    train_labels = (train_probs >= 0.5).float()
    train_acc = (train_labels == y_train).float().mean().item()

print(f"\nTrain accuracy: {train_acc:.1%}")
print(f"Test accuracy:  {val_acc:.1%}")

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/nba_model.pt")
# save scaler for prediction (use same normalizing values as model)
joblib.dump(scaler, "models/nba_scaler.joblib")
pd.Series(feature_cols).to_csv("models/feature_columns.csv", index=False, header=False)

print("Saved model, scaler, and feature columns.")
