import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --- 1) Data ---
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_prime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

np.random.seed(0)
torch.manual_seed(0)

N_train, N_val, N_test = 1000, 200, 200
xs = np.random.uniform(-1, 1, size=(N_train + N_val + N_test, 1))
ys = runge(xs)
ys_prime = runge_prime(xs)

# shuffle and split
perm = np.random.permutation(len(xs))
xs, ys, ys_prime = xs[perm], ys[perm], ys_prime[perm]

x_train, y_train, y_train_prime = xs[:N_train], ys[:N_train], ys_prime[:N_train]
x_val, y_val, y_val_prime = xs[N_train:N_train+N_val], ys[N_train:N_train+N_val], ys_prime[N_train:N_train+N_val]
x_test, y_test, y_test_prime = xs[N_train+N_val:], ys[N_train+N_val:], ys_prime[N_train+N_val:]

# convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train_t = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
y_train_prime_t = torch.tensor(y_train_prime, dtype=torch.float32).to(device)
x_val_t = torch.tensor(x_val, dtype=torch.float32, requires_grad=True).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
y_val_prime_t = torch.tensor(y_val_prime, dtype=torch.float32).to(device)
x_test_t = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
y_test_prime_t = torch.tensor(y_test_prime, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t, y_train_prime_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_t, y_val_t, y_val_prime_t), batch_size=128, shuffle=False)

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
    for xb, yb, yb_prime in train_loader:
        optimizer.zero_grad()
        
        # Function loss
        pred_y = model(xb)
        loss_y = loss_fn(pred_y, yb)
        
        # Derivative loss
        pred_y_prime = torch.autograd.grad(
            outputs=pred_y,
            inputs=xb,
            grad_outputs=torch.ones_like(pred_y),
            create_graph=True,
            retain_graph=True
        )[0]
        loss_y_prime = loss_fn(pred_y_prime, yb_prime)
        
        # Total loss
        loss = loss_y + loss_y_prime
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    for xb, yb, yb_prime in val_loader:
        # Re-enable gradient tracking for the input tensor
        xb.requires_grad_(True)
        
        # Forward pass
        pred_y_val = model(xb)
        
        # Calculate derivative
        pred_y_prime_val = torch.autograd.grad(
            outputs=pred_y_val,
            inputs=xb,
            grad_outputs=torch.ones_like(pred_y_val),
            create_graph=False,
            retain_graph=False
        )[0]

        # Calculate losses outside the no_grad block
        loss_y = loss_fn(pred_y_val, yb)
        loss_y_prime = loss_fn(pred_y_prime_val, yb_prime)
        
        # Combine losses
        loss = loss_y + loss_y_prime
        val_loss += loss.item() * xb.size(0)

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

# Temporarily enable gradients for the test set input to compute derivatives
x_test_t.requires_grad_(True)

# Test set evaluation - Corrected
pred_test_t = model(x_test_t)
pred_test = pred_test_t.detach().cpu().numpy().ravel()
mse_test = np.mean((pred_test - y_test.ravel())**2)
max_err = np.max(np.abs(pred_test - y_test.ravel()))

# Derivative test set evaluation
pred_test_prime_t = torch.autograd.grad(
    outputs=pred_test_t,
    inputs=x_test_t,
    grad_outputs=torch.ones_like(pred_test_t),
    create_graph=False,
    retain_graph=False
)[0]
pred_test_prime = pred_test_prime_t.detach().cpu().numpy().ravel()
mse_test_prime = np.mean((pred_test_prime - y_test_prime.ravel())**2)
max_err_prime = np.max(np.abs(pred_test_prime - y_test_prime.ravel()))

# Reset requires_grad to its original state (optional but good practice)
x_test_t.requires_grad_(False)

print(f"Test Function MSE: {mse_test:.6e}, Test Function max abs error: {max_err:.6e}")
print(f"Test Derivative MSE: {mse_test_prime:.6e}, Test Derivative max abs error: {max_err_prime:.6e}")

# dense plot on [-1,1]
x_dense = np.linspace(-1, 1, 400).reshape(-1,1).astype(np.float32)
# Create the tensor with requires_grad=True outside the no_grad block
x_dense_t = torch.tensor(x_dense, dtype=torch.float32, requires_grad=True).to(device)

# Function prediction
# The with torch.no_grad() block is good practice for the main prediction to save memory
with torch.no_grad():
    pred_dense = model(x_dense_t).detach().cpu().numpy().ravel()

# Derivative prediction
# This needs to be done outside the no_grad block
# The computational graph must be built here
pred_dense_prime_t = torch.autograd.grad(
    outputs=model(x_dense_t),
    inputs=x_dense_t,
    grad_outputs=torch.ones_like(model(x_dense_t)),
    create_graph=False,
    retain_graph=False
)[0]
pred_dense_prime = pred_dense_prime_t.detach().cpu().numpy().ravel()

y_dense = runge(x_dense).ravel()
y_dense_prime = runge_prime(x_dense).ravel()

# draw graphs for function and derivative
plt.figure(figsize=(12, 5))

# Plot for the function
plt.subplot(1, 2, 1)
plt.plot(x_dense, y_dense, label='True $f(x)$')
plt.plot(x_dense, pred_dense, '--', label='NN prediction')
plt.scatter(x_test, pred_test, s=10, alpha=0.5, label='Test preds')
plt.legend()
plt.title('Runge Function and NN Prediction')
plt.xlabel('x')
plt.ylabel('f(x)')

# Plot for the derivative
plt.subplot(1, 2, 2)
plt.plot(x_dense, y_dense_prime, label='True $f\'(x)$')
plt.plot(x_dense, pred_dense_prime, '--', label='NN prediction')
plt.scatter(x_test, pred_test_prime, s=10, alpha=0.5, label='Test preds')
plt.legend()
plt.title('Runge Function Derivative and NN Prediction')
plt.xlabel('x')
plt.ylabel('f\'(x)')

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
plt.title('Train/Val Loss')
plt.tight_layout()
plt.show()