import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import random

# 1. Define CNN architecture for target and shadow
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64*14*14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2. Load MNIST and create splits
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

indices = list(range(len(full_train)))
random.shuffle(indices)

# 10k for target, 20k for shadow, rest unused
target_train_indices = indices[:10000]
shadow_indices       = indices[10000:30000]
shadow_train_indices = shadow_indices[:10000]
shadow_test_indices  = shadow_indices[10000:20000]

target_train_ds = Subset(full_train, target_train_indices)
shadow_train_ds = Subset(full_train, shadow_train_indices)
shadow_test_ds  = Subset(full_train, shadow_test_indices)

target_train_loader = DataLoader(target_train_ds, batch_size=64, shuffle=True, num_workers=0)
shadow_train_loader = DataLoader(shadow_train_ds, batch_size=64, shuffle=True, num_workers=0)
shadow_test_loader  = DataLoader(shadow_test_ds,  batch_size=64, shuffle=False, num_workers=0)
target_test_loader  = DataLoader(test_ds,         batch_size=1000, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Train Target model on 10k (overfitting)
target_model = CNNModel().to(device)
opt_t = optim.Adam(target_model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    target_model.train()
    for x, y in target_train_loader:
        x, y = x.to(device), y.to(device)
        opt_t.zero_grad()
        out = target_model(x)
        loss = crit(out, y)
        loss.backward()
        opt_t.step()

    target_model.eval()
    cm, cn = 0, 0
    with torch.no_grad():
        for x, y in target_train_loader:
            x, y = x.to(device), y.to(device)
            preds = target_model(x).argmax(dim=1)
            cm += preds.eq(y).sum().item()
        for x, y in target_test_loader:
            x, y = x.to(device), y.to(device)
            preds = target_model(x).argmax(dim=1)
            cn += preds.eq(y).sum().item()
    mem_acc = 100 * cm / len(target_train_ds)
    nonmem_acc = 100 * cn / len(test_ds)
    print(f"Target Epoch {epoch:2d}  MemAcc={mem_acc:.2f}%  NonMemAcc={nonmem_acc:.2f}%")

# 4. Train Shadow model on 10k/10k
shadow_model = CNNModel().to(device)
opt_s = optim.Adam(shadow_model.parameters(), lr=0.001)

for epoch in range(1, 11):
    shadow_model.train()
    for x, y in shadow_train_loader:
        x, y = x.to(device), y.to(device)
        opt_s.zero_grad()
        out = shadow_model(x)
        loss = crit(out, y)
        loss.backward()
        opt_s.step()

    shadow_model.eval()
    cm, cn = 0, 0
    with torch.no_grad():
        for x, y in shadow_train_loader:
            x, y = x.to(device), y.to(device)
            preds = shadow_model(x).argmax(dim=1)
            cm += preds.eq(y).sum().item()
        for x, y in shadow_test_loader:
            x, y = x.to(device), y.to(device)
            preds = shadow_model(x).argmax(dim=1)
            cn += preds.eq(y).sum().item()
    mem_acc_s = 100 * cm / len(shadow_train_ds)
    nonmem_acc_s = 100 * cn / len(shadow_test_ds)
    print(f"Shadow Epoch {epoch:2d}  MemAcc={mem_acc_s:.2f}%  NonMemAcc={nonmem_acc_s:.2f}%")

# 5. Build attack dataset from Shadow outputs
shadow_model.eval()
member_outs, nonmember_outs = [], []
with torch.no_grad():
    for x, _ in shadow_train_loader:
        x = x.to(device)
        out = shadow_model(x)
        member_outs.append(out.cpu())
    for x, _ in shadow_test_loader:
        x = x.to(device)
        out = shadow_model(x)
        nonmember_outs.append(out.cpu())

member_outs = torch.cat(member_outs, dim=0)
nonmember_outs = torch.cat(nonmember_outs, dim=0)

X_attack = torch.cat([member_outs, nonmember_outs], dim=0)
y_attack = torch.cat([torch.ones(len(member_outs)), torch.zeros(len(nonmember_outs))], dim=0)

perm = torch.randperm(len(X_attack))
X_attack = X_attack[perm]
y_attack = y_attack[perm]

split = int(0.7 * len(X_attack))
X_att_train, X_att_test = X_attack[:split], X_attack[split:]
y_att_train, y_att_test = y_attack[:split].long(), y_attack[split:].long()

att_train_ds = TensorDataset(X_att_train, y_att_train)
att_test_ds  = TensorDataset(X_att_test,  y_att_test)
att_train_loader = DataLoader(att_train_ds, batch_size=64, shuffle=True, num_workers=0)
att_test_loader  = DataLoader(att_test_ds,  batch_size=64, shuffle=False, num_workers=0)

# 6. Define and train Attack model
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

attack_model = AttackModel().to(device)
opt_a = optim.Adam(attack_model.parameters(), lr=0.001)
crit_a = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    attack_model.train()
    total_loss = 0.0
    for feats, lbls in att_train_loader:
        feats, lbls = feats.to(device), lbls.to(device)
        opt_a.zero_grad()
        logits = attack_model(feats)
        loss = crit_a(logits, lbls)
        loss.backward()
        opt_a.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(att_train_loader)

    attack_model.eval()
    corr = 0
    with torch.no_grad():
        for feats, lbls in att_test_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            preds = attack_model(feats).argmax(dim=1)
            corr += preds.eq(lbls).sum().item()
    acc = 100 * corr / len(att_test_ds)
    print(f"Attack Epoch {epoch:2d}  Loss={avg_loss:.4f}  Acc={acc:.2f}%")

# 7. Evaluate Attack on Target model
# Prepare target members vs non-members
member_feats_t, nonmember_feats_t = [], []
target_model.eval()
with torch.no_grad():
    for x, _ in target_train_loader:
        x = x.to(device)
        member_feats_t.append(target_model(x).cpu())
    cnt = 0
    for x, _ in target_test_loader:
        if cnt >= len(target_train_ds):
            break
        x = x.to(device)
        nonmember_feats_t.append(target_model(x).cpu())
        cnt += x.size(0)

member_feats_t = torch.cat(member_feats_t, dim=0)[:len(target_train_ds)]
nonmember_feats_t = torch.cat(nonmember_feats_t, dim=0)[:len(target_train_ds)]

X_tgt = torch.cat([member_feats_t, nonmember_feats_t], dim=0)
y_tgt = torch.cat([torch.ones(len(member_feats_t)), torch.zeros(len(nonmember_feats_t))], dim=0)

attack_model.eval()
corr_tgt = 0
with torch.no_grad():
    for i in range(len(X_tgt)):
        feat = X_tgt[i].unsqueeze(0).to(device)
        lbl = y_tgt[i].view(1).long().to(device)
        pred = attack_model(feat).argmax(dim=1)
        corr_tgt += pred.eq(lbl).sum().item()
attack_acc = 100 * corr_tgt / len(X_tgt)

print(f"\nMembership Inference Attack Accuracy on Target: {attack_acc:.2f}%")
