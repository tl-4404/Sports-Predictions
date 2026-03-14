import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import glob
import os

# load training data

data_folder = "path/to/your/daily/csv/folder"
csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {data_folder}")

df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
print(f"Loaded {len(csv_files)} files and {len(df)} rows.")

# drop useless columns
df = df.drop(columns=[
    "GAME_DATE",
    "home_team",
    "away_team",
])
# split into training df, X, and results, y
X = df.select_dtypes(include=['number']).drop(columns=['home_win', "GAME_ID"])
y = df["home_win"].astype(float)

print("TRAIN feature count:", X.shape[1])
print("TRAIN columns:", list(X.columns)[:10], "...")

feature_cols = list(X.columns)

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert pd df -> tensors
X_train = torch.tensor(X_scaled, dtype=torch.float32)
y_train = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# training model
class ml_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    
    def forward(self, x):
        return self.out(x)
    
model = ml_Model(input_dim=X_train.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# train model
for epoch in range(100):
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: {avg_loss:.4f}")

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/nba_model.pt")
# save scaler for prediction (use same normalizing values as model)
joblib.dump(scaler, "models/nba_scaler.joblib")
pd.Series(feature_cols).to_csv("models/feature_columns.csv", index=False)

print("Saved model, scaler, and feature columns.")
