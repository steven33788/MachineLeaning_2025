import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --- 1) Data ---
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

np.random.seed(0)
torch.manual_seed(0)

N_train, N_val, N_test = 1000, 200, 200
xs = np.random.uniform(-1, 1, size=(N_train + N_val + N_test, 1))
ys = runge(xs)

# shuffle and split
perm = np.random.permutation(len(xs))
xs, ys = xs[perm], ys[perm]

x_train, y_train = xs[:N_train], ys[:N_train]
x_val, y_val = xs[N_train:N_train+N_val], ys[N_train:N_train+N_val]
x_test, y_test = xs[N_train+N_val:], ys[N_train+N_val:]

# convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=128, shuffle=False)

# --- 2) Model ---
class MLP(nn.Module):
    def __init__(self, hidden_sizes=[50,50], activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = MLP(hidden_sizes=[50,50], activation=nn.Tanh).to(device)

# --- 3) Training setup ---
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 2000
best_val = float('inf')
patience = 100
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(1, n_epochs+1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # early stopping
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# load best
model.load_state_dict(best_state)
model.to(device)

# --- 4) Evaluation ---
model.eval()
with torch.no_grad():
    pred_test = model(x_test_t).cpu().numpy().ravel()
    mse_test = np.mean((pred_test - y_test.ravel())**2)
    max_err = np.max(np.abs(pred_test - y_test.ravel()))

print(f"Test MSE: {mse_test:.6e}, Test max abs error: {max_err:.6e}")

# dense plot on [-1,1]
x_dense = np.linspace(-1, 1, 400).reshape(-1,1).astype(np.float32)
with torch.no_grad():
    pred_dense = model(torch.tensor(x_dense).to(device)).cpu().numpy().ravel()
y_dense = runge(x_dense).ravel()

#draw graphs
plt.figure(figsize=(8,4))
plt.plot(x_dense, y_dense, label='True f(x)')
plt.plot(x_dense, pred_dense, '--', label='NN prediction')
plt.scatter(x_test, pred_test, s=10, alpha=0.5, label='Test preds')
plt.legend()
plt.title('Runge function and NN prediction')
plt.tight_layout()
plt.show()

# loss curves
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE (log scale)')
plt.legend()
plt.title('Train/Val loss')
plt.tight_layout()
plt.show()






