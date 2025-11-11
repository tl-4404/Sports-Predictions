import torch
import torch.nn as nn
import torch.optim as optim

# data points (X_train)
# Aof_rating, Adef_rating, Bof_rating, Bdef_rating, AH(1) v. BA(0), this seasons win%

# outcomes (y_train)
# A win (1), B win(0)

X_train = torch.tensor([
    [112.8, 112.6, 113.3, 118.3, 0.0, 1.0, 0.500, 0.333],
    [107.6, 120.5, 115.5, 114.1, 0.0, 1.0, 0.2220, 0.500],
    [120.0, 115.2, 109.4, 117.2, 0.0, 1.0, 0.600, 0.333],
    [104.2, 110.0, 117.0, 115.5, 1.0, 0.0, 0.300, 0.600],
    [115.4, 114.8, 117.9, 110.9, 1.0, 0.0, 0.555, 0.777],
    [117.7, 113.2, 116.4, 111.6, 0.0, 1.0, 0.700, 0.600]
], dtype=torch.float32)

y_train = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)

class ml_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.out(x)
    
model = ml_Model()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/500 - Loss: {loss.item():.4f}")

X_test = torch.tensor([
    [116.4, 114.2, 111.3, 126.6, 0.0, 1.0, 0.500, 0.100]
])

y_test = model(X_test).item()
print("Predicted probability Team A wins:", round(y_test, 3))
print("Predicted winner:", "Team A" if y_test > 0.5 else "Team B")

# lakers v hornets 11/10/25 [117.2, 116.2, 115.8, 117.9, 0.0, 1.0, 0.700, 0.333]
# wizards v pistons 11/10/25 [106.8, 121.8, 115.5, 110.2, 0.0, 1.0, 0.100, 0.900]
# blazers v magic 11/10/25 [114.9, 112.9, 114.1, 114.2, 0.0, 1.0, 0.555, 0.400]
# warriors v. thunder [114.0, 111.4, 118.0, 104.5, 0.0, 1.0, 0.545, 0.909]
# 