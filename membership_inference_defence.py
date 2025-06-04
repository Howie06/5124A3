# Appendix: Python code for training and evaluation with output encryption defense.
import torch
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1) Load MNIST and split into target/shadow member/non-member sets
transform = transforms.ToTensor()
mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
target_indices = np.arange(0, 5000)
target_non_indices = np.arange(5000, 10000)
shadow_indices = np.arange(10000, 15000)
shadow_non_indices = np.arange(15000, 20000)
target_set = Subset(mnist, target_indices)
target_non_set = Subset(mnist, target_non_indices)
shadow_set = Subset(mnist, shadow_indices)
shadow_non_set = Subset(mnist, shadow_non_indices)
target_loader = DataLoader(target_set, batch_size=128, shuffle=True)
target_non_loader = DataLoader(target_non_set, batch_size=128, shuffle=False)
shadow_loader = DataLoader(shadow_set, batch_size=128, shuffle=True)
shadow_non_loader = DataLoader(shadow_non_set, batch_size=128, shuffle=False)

# 2) Define a simple neural network (e.g. MLP) for MNIST
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 3) Train target model on its member data
device = torch.device("cpu")
target_model = SimpleNet().to(device)
optimizer = optim.Adam(target_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
target_model.train()
for epoch in range(5):
    for x, y in target_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = target_model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()

# 4) Train shadow model similarly
shadow_model = SimpleNet().to(device)
optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
shadow_model.train()
for epoch in range(5):
    for x, y in shadow_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(shadow_model(x), y)
        loss.backward(); optimizer.step()

# 5) Collect outputs for shadow data (for attack training)
def get_outputs(model, loader):
    model.eval()
    outs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            outs.append(probs)
            labels.append(y.numpy())
    return np.vstack(outs), np.concatenate(labels)

shadow_member_out, shadow_member_lbl = get_outputs(shadow_model, shadow_loader)
shadow_non_out, shadow_non_lbl = get_outputs(shadow_model, shadow_non_loader)
# 'shadow_member_out' are outputs on data shadow_model was trained on (members)
# 'shadow_non_out' on held-out data (non-members)
# We'll train per-class attack MLPs below.

# 6) Encrypt the target outputs at inference time
# Prepare encryption key (secret): one positive weight per class
key = torch.rand(10) + 0.5  # example: values in [0.5,1.5]
key = key / key.sum()      # normalize (optional; not strictly needed)

# Function to apply encryption to a batch of softmax outputs
def encrypt_outputs(probs, key):
    # probs: numpy array shape (N,10)
    q = probs * key.numpy()           # multiply each column
    q = q / q.sum(axis=1, keepdims=True)  # renormalize rows
    return q

# 7) Train per-class attack models on shadow outputs (unencrypted)
from sklearn.neural_network import MLPClassifier

attack_mlps = {}
for cls in range(10):
    # Prepare training data for class 'cls'
    idx_m = (shadow_member_lbl == cls)
    idx_n = (shadow_non_lbl == cls)
    X_train = np.vstack([shadow_member_out[idx_m], shadow_non_out[idx_n]])
    y_train = np.hstack([np.ones(idx_m.sum()), np.zeros(idx_n.sum())])
    if len(X_train) == 0:
        continue

    # Increase max_iter and enable early stopping to avoid convergence warnings
    attack_mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,           # give more epochs
        tol=1e-4,               # default is 1e-4; for stricter convergence, you could use 1e-5
        early_stopping=True,    # stop if validation score does not improve for 10 epochs
        validation_fraction=0.1 # 10% of data used as a small hold‐out set
    )

    attack_mlp.fit(X_train, y_train)
    attack_mlps[cls] = attack_mlp


# 8) Evaluate attack on the target model
target_model.eval()
# Get target outputs for its members and non-members
target_member_out, target_member_lbl = get_outputs(target_model, target_loader)
target_non_out, target_non_lbl = get_outputs(target_model, target_non_loader)
# Baseline: use raw softmax (no defense)
X_test_base = np.vstack([
    target_member_out[target_member_lbl == cls] for cls in range(10)
] + [
    target_non_out[target_non_lbl == cls] for cls in range(10)
])
y_test = np.hstack([
    np.ones((target_member_lbl == cls).sum()) for cls in range(10)
] + [
    np.zeros((target_non_lbl == cls).sum()) for cls in range(10)
])
# For simplicity, we test per-class: compute accuracy per class and average
acc_base = []
acc_enc = []
for cls, mlp in attack_mlps.items():
    # Subset test data for this class
    idx_m = (target_member_lbl == cls)
    idx_n = (target_non_lbl == cls)
    if not np.any(idx_m) or not np.any(idx_n):
        continue
    Xm = target_member_out[idx_m]; Xn = target_non_out[idx_n]
    y_m = np.ones(Xm.shape[0]); y_n = np.zeros(Xn.shape[0])
    X_base = np.vstack([Xm, Xn]); y_true = np.hstack([y_m, y_n])
    # Baseline attack accuracy
    y_pred = mlp.predict(X_base)
    acc_base.append((y_pred==y_true).mean())
    # Defense: encrypt outputs before attack
    Xm_enc = encrypt_outputs(Xm, key)
    Xn_enc = encrypt_outputs(Xn, key)
    X_enc = np.vstack([Xm_enc, Xn_enc])
    y_pred_enc = mlp.predict(X_enc)
    acc_enc.append((y_pred_enc==y_true).mean())

print(f"Attack accuracy per class (baseline) = {np.mean(acc_base):.3f}")
print(f"Attack accuracy per class (encrypted) = {np.mean(acc_enc): .3f}")
