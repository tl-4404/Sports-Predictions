import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from training_validation import chronological_features
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "cache", "training_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# columns that are identifiers or targets — not features
DROP_COLS  = ["game_pk", "game_date", "pitcher_id", "home_team", "actual_ks", "actual_outs"]


# ── model ────────────────────────────────────────────────────────────────────
# two-headed network: shared layers learn general pitcher/batter patterns,
# then separate heads specialize for Ks vs outs
class KsOutsModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # separate prediction heads
        self.ks_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.outs_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        shared = self.shared(x)
        ks     = self.ks_head(shared)
        outs   = self.outs_head(shared)
        return ks, outs


# ── data loading ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["game_date"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # separate targets
    y_ks   = df["actual_ks"].values.astype(np.float32)
    y_outs = df["actual_outs"].values.astype(np.float32)

    # feature matrix — drop identifiers and targets
    X_df = df.drop(columns=DROP_COLS)

    # fill NaN with column median (batter splits often have gaps);
    # medians are saved so inference fills the same way (see train_model)
    X_df, medians, train_idx, val_idx = chronological_features(X_df, df["game_date"])
    load_data.medians = medians  # stashed for train_model to persist

    feature_cols = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    # time-based split: last 20% of games by date for validation
    # prevents data leakage — model never sees future games during training
    return X, y_ks, y_outs, feature_cols, train_idx, val_idx


# ── training ─────────────────────────────────────────────────────────────────
def train_model(epochs=100, lr=1e-3, batch_size=64):
    X, y_ks, y_outs, feature_cols, train_idx, val_idx = load_data()

    # fit scaler on training data only — no leakage from validation set
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx]   = scaler.transform(X[val_idx])

    # convert to tensors
    X_train = torch.tensor(X[train_idx])
    X_val   = torch.tensor(X[val_idx])
    y_ks_train   = torch.tensor(y_ks[train_idx]).unsqueeze(1)
    y_ks_val     = torch.tensor(y_ks[val_idx]).unsqueeze(1)
    y_outs_train = torch.tensor(y_outs[train_idx]).unsqueeze(1)
    y_outs_val   = torch.tensor(y_outs[val_idx]).unsqueeze(1)

    model     = KsOutsModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")
    patience      = 15
    wait          = 0

    for epoch in range(epochs):
        model.train()

        # shuffle training data each epoch
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb  = X_train[idx]

            pred_ks, pred_outs = model(xb)
            loss = loss_fn(pred_ks, y_ks_train[idx]) + loss_fn(pred_outs, y_outs_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # validation
        model.eval()
        with torch.no_grad():
            vp_ks, vp_outs = model(X_val)
            val_loss = loss_fn(vp_ks, y_ks_val).item() + loss_fn(vp_outs, y_outs_val).item()

        avg_train = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}  train={avg_train:.4f}  val={val_loss:.4f}")

        # early stopping — save best model, stop if no improvement for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # load best weights
    model.load_state_dict(best_state)

    # final validation metrics
    model.eval()
    with torch.no_grad():
        vp_ks, vp_outs = model(X_val)
        mae_ks   = (vp_ks - y_ks_val).abs().mean().item()
        mae_outs = (vp_outs - y_outs_val).abs().mean().item()
    print(f"\nBest validation — Ks MAE: {mae_ks:.2f}, Outs MAE: {mae_outs:.2f}")

    # save everything needed for prediction
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(MODEL_DIR, "ks_outs_model.pt"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(load_data.medians, os.path.join(MODEL_DIR, "medians.pkl"))
    print(f"Saved model, scaler, feature_cols, medians → {MODEL_DIR}")

    return model, scaler, feature_cols


# ── prediction helper ────────────────────────────────────────────────────────
# takes a dict of features (same keys as training), returns predicted Ks and outs
def predict(features_dict, model=None, scaler=None, feature_cols=None, medians=None):
    if model is None:
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        medians_path = os.path.join(MODEL_DIR, "medians.pkl")
        medians      = joblib.load(medians_path) if os.path.exists(medians_path) else None
        state        = torch.load(os.path.join(MODEL_DIR, "ks_outs_model.pt"),
                                  weights_only=True)
        model = KsOutsModel(len(feature_cols))
        model.load_state_dict(state)

    # build feature vector in the correct column order
    x = np.array([features_dict.get(col, np.nan) for col in feature_cols], dtype=np.float32)

    # fill NaN the same way training did — with the training-set medians
    # (a raw 0 scales to an extreme z-score and poisons the prediction)
    if medians is not None:
        fill = np.array([medians.get(col, 0.0) for col in feature_cols], dtype=np.float32)
        x = np.where(np.isnan(x), fill, x)
    x = np.nan_to_num(x, nan=0.0)
    x = scaler.transform(x.reshape(1, -1))
    x = torch.tensor(x)

    model.eval()
    with torch.no_grad():
        pred_ks, pred_outs = model(x)

    return round(pred_ks.item(), 1), round(pred_outs.item(), 1)


if __name__ == "__main__":
    model, scaler, feature_cols = train_model()
    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]} ...")
