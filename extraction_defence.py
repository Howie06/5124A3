import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# Adjust path so that target_model.py and Sub_model.py (in /mnt/data) can be imported
sys.path.append('/mnt/data')
from target_model import MNIST as TargetModel
from extraction_attack import SubstituteModel

# Device & MNIST test loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

# Load and evaluate the clean target model
target = TargetModel().to(device)
target.load_state_dict(torch.load('target_model.pth', map_location=device))
target.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = target(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
target_acc = 100 * correct / len(test_ds)

# Defence: randomly flip 5% of labels
def defended_predict(model, x, noise_rate=0.05):
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        mask = torch.rand_like(preds, dtype=torch.float) < noise_rate
        rnd = torch.randint(0, 10, preds.shape, dtype=torch.long)
        rnd = torch.where(rnd == preds, (rnd + 1) % 10, rnd)
        preds[mask] = rnd[mask]
        return preds

# Simulate oracle queries with defence to build the attack dataset
all_x, all_y = [], []
for x, _ in test_loader:
    x = x.to(device)
    y_def = defended_predict(target, x, noise_rate=0.05)
    all_x.append(x.cpu())
    all_y.append(y_def.cpu())
sub_inputs = torch.cat(all_x, dim=0)
sub_labels = torch.cat(all_y, dim=0)

# Prepare substitute DataLoader
attack_ds = TensorDataset(sub_inputs, sub_labels)
attack_loader = DataLoader(attack_ds, batch_size=64, shuffle=True, num_workers=0)

# Train the substitute model on the defended labels
sub = SubstituteModel().to(device)
opt = optim.Adam(sub.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    sub.train()
    for x_s, y_s in attack_loader:
        x_s, y_s = x_s.to(device), y_s.to(device)
        opt.zero_grad()
        loss_fn(sub(x_s), y_s).backward()
        opt.step()

# Evaluate substitute’s utility on true test labels
sub.eval()
correct_sub = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = sub(x).argmax(dim=1)
        correct_sub += preds.eq(y).sum().item()
sub_acc = 100 * correct_sub / len(test_ds)

# Evaluate fidelity (agreement) between substitute and undefended target
agree = 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        t_pred = target(x).argmax(dim=1)
        s_pred = sub(x).argmax(dim=1)
        agree += (t_pred == s_pred).sum().item()
fidelity = 100 * agree / len(test_ds)

# Print results
print(f"Target accuracy (clean):            {target_acc:6.2f}%")
print(f"Substitute accuracy (under defence):{sub_acc:6.2f}%")
print(f"Fidelity (agreement) under defence:  {fidelity:6.2f}%")
