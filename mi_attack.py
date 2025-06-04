# attack_mnist.py

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

# Import the MNIST model architecture from mnist.py
from mnist import MNIST


# ───────────────────────────────────────────────────────────────────────────
# 1. Utility functions: train & test loops for classifier (target/shadow)
# ───────────────────────────────────────────────────────────────────────────

def train_classifier(model, device, train_loader, optimizer, criterion, epoch):
    """
    Standard training loop for a classifier (target or shadow).
    """
    model.train()
    running_loss = 0.0
    total_samples = len(train_loader.dataset)
    batches_per_epoch = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Print status every 200 mini‐batches
        if batch_idx % 200 == 0:
            avg_loss = running_loss / 200
            samples_processed = batch_idx * data.size(0)
            percent = 100. * batch_idx / batches_per_epoch
            print(
                f"[Epoch {epoch:2d}] "
                f"[{samples_processed:5d}/{total_samples}] "
                f"{percent:3.0f}%  Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0


def evaluate_classifier(model, device, test_loader, criterion=None):
    """
    Evaluate model on test_loader. If criterion is provided, compute loss too.
    Returns (avg_loss, accuracy). If criterion is None, avg_loss is None.
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if criterion is not None:
                test_loss += criterion(output, target).item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()

    if criterion is not None:
        test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return (test_loss, accuracy)


# ───────────────────────────────────────────────────────────────────────────
# 2. Define the “attack model” (small MLP: 10‐dim → binary)
# ───────────────────────────────────────────────────────────────────────────

class AttackMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)  # two outputs: [non‐member, member]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits, to be used with CrossEntropyLoss


def train_attack_model(model, device, attack_loader, optimizer, criterion, epochs):
    """
    Training loop for the attack model.
    """
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (prob_vecs, member_labels) in enumerate(attack_loader, start=1):
            prob_vecs = prob_vecs.to(device)
            labels = member_labels.to(device)

            optimizer.zero_grad()
            logits = model(prob_vecs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        avg_loss = running_loss / len(attack_loader)
        acc = 100. * correct / total
        print(f"[Attack] Epoch {epoch:2d}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")
    print("Finished training attack model.\n")


def evaluate_attack_model(model, device, attack_loader):
    """
    Evaluate the attack model on a given attack_loader.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for prob_vecs, member_labels in attack_loader:
            prob_vecs = prob_vecs.to(device)
            labels = member_labels.to(device)
            logits = model(prob_vecs)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100. * correct / total
    return acc


# ───────────────────────────────────────────────────────────────────────────
# 3. Main pipeline
# ───────────────────────────────────────────────────────────────────────────

def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    # MNIST transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full MNIST train & test sets
    full_train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    full_test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # ─────────────────────────────
    # 3.a) Ensure target model is trained
    # ─────────────────────────────
    target_model_path = "target_model.pth"
    if not os.path.exists(target_model_path):
        print("Target model not found. Training it now (5 epochs on full train set)...")
        train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(full_test_ds, batch_size=1000, shuffle=False, num_workers=2)

        target_model = MNIST().to(device)
        optimizer = optim.SGD(target_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 6):
            train_classifier(target_model, device, train_loader, optimizer, criterion, epoch)
            t_loss, t_acc = evaluate_classifier(target_model, device, test_loader, criterion)
            print(f"[Target] Epoch {epoch:2d}  Test loss: {t_loss:.4f}  Test Acc: {t_acc:.2f}%\n")

        torch.save(target_model.state_dict(), target_model_path)
        print("Target model saved to", target_model_path)

    # Load the trained target model
    target_model = MNIST().to(device)
    target_model.load_state_dict(torch.load(target_model_path, map_location=device))
    target_model.eval()
    print("Loaded target model.\n")

    # ───────────────────────────────────────────────────────────────────────────
    # 3.b) Train S shadow models (with random sampling per shadow)
    # ───────────────────────────────────────────────────────────────────────────
    S = args.num_shadows
    shadow_epochs = args.shadow_epochs
    shadow_train_size = args.shadow_train_size
    shadow_holdout_size = args.shadow_holdout_size

    shadow_models = []  # will store tuples (shadow_model, train_idx, holdout_idx)

    all_indices = list(range(len(full_train_ds)))
    random.shuffle(all_indices)

    for i in range(S):
        # Randomly sample (shadow_train_size + shadow_holdout_size) indices from the 60k pool
        subset = random.sample(all_indices, shadow_train_size + shadow_holdout_size)
        train_idx = subset[:shadow_train_size]
        holdout_idx = subset[shadow_train_size:]

        shadow_train_ds = Subset(full_train_ds, train_idx)
        shadow_holdout_ds = Subset(full_train_ds, holdout_idx)

        shadow_train_loader = DataLoader(shadow_train_ds, batch_size=64, shuffle=True, num_workers=2)
        shadow_holdout_loader = DataLoader(shadow_holdout_ds, batch_size=1000, shuffle=False, num_workers=2)

        print(f"Training shadow model [{i + 1}/{S}] on {len(train_idx)} samples ...")
        shadow_model = MNIST().to(device)
        optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, shadow_epochs + 1):
            train_classifier(shadow_model, device, shadow_train_loader, optimizer, criterion, epoch)
        print(f"  --> Finished training shadow model {i + 1}.\n")

        shadow_models.append((shadow_model, train_idx, holdout_idx))

    # ───────────────────────────────────────────────────────────────────────────
    # 3.c) Build the attack dataset from shadow‐model outputs
    # ───────────────────────────────────────────────────────────────────────────
    attack_inputs = []  # will store probability vectors [10]
    attack_labels = []  # 1 if member, 0 if non‐member

    softmax = nn.Softmax(dim=1)

    for i, (shadow_model, train_idx, holdout_idx) in enumerate(shadow_models):
        shadow_model.eval()

        # 1) Members: shadow_train_idx
        member_subset = Subset(full_train_ds, train_idx)
        member_loader = DataLoader(member_subset, batch_size=1000, shuffle=False, num_workers=2)

        with torch.no_grad():
            for data, _ in member_loader:
                data = data.to(device)
                outputs = shadow_model(data)
                probs = softmax(outputs).cpu()
                for vec in probs:
                    attack_inputs.append(vec.unsqueeze(0))
                    attack_labels.append(1)

        # 2) Non‐members: shadow_holdout_idx
        holdout_subset = Subset(full_train_ds, holdout_idx)
        holdout_loader = DataLoader(holdout_subset, batch_size=1000, shuffle=False, num_workers=2)

        with torch.no_grad():
            for data, _ in holdout_loader:
                data = data.to(device)
                outputs = shadow_model(data)
                probs = softmax(outputs).cpu()
                for vec in probs:
                    attack_inputs.append(vec.unsqueeze(0))
                    attack_labels.append(0)

        print(f"Collected attack data from shadow model {i + 1}: "
              f"{len(train_idx)} member samples, {len(holdout_idx)} non‐member samples.")

    attack_X = torch.cat(attack_inputs, dim=0)  # shape: [N_attack, 10]
    attack_Y = torch.tensor(attack_labels, dtype=torch.long)  # shape: [N_attack]

    print(f"\nTotal attack‐model dataset size: {attack_X.size(0)} samples "
          f"({attack_Y.sum().item()} members, {attack_X.size(0) - attack_Y.sum().item()} non‐members)\n")

    # Split into train/test for attack model
    attack_dataset = TensorDataset(attack_X, attack_Y)
    train_size = int(0.8 * len(attack_dataset))
    test_size = len(attack_dataset) - train_size
    attack_train_ds, attack_test_ds = torch.utils.data.random_split(
        attack_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )
    attack_train_loader = DataLoader(attack_train_ds, batch_size=256, shuffle=True, num_workers=2)
    attack_test_loader = DataLoader(attack_test_ds, batch_size=256, shuffle=False, num_workers=2)

    # ───────────────────────────────────────────────────────────────────────────
    # 3.d) Train the attack model
    # ───────────────────────────────────────────────────────────────────────────
    attack_model = AttackMLP().to(device)
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training attack model on shadow data ...\n")
    train_attack_model(attack_model, device, attack_train_loader, optimizer, criterion, epochs=args.attack_epochs)

    train_acc = evaluate_attack_model(attack_model, device, attack_train_loader)
    test_acc = evaluate_attack_model(attack_model, device, attack_test_loader)
    print(f"[Attack Model] On shadow‐data TRAIN split: {train_acc:.2f}%  |  TEST split: {test_acc:.2f}%\n")

    # ───────────────────────────────────────────────────────────────────────────
    # 3.e) Evaluate the attack against the target model
    # ───────────────────────────────────────────────────────────────────────────

    # 1) Target members: entire MNIST training set (60k)
    target_train_loader = DataLoader(full_train_ds, batch_size=1000, shuffle=False, num_workers=2)
    target_member_probs = []

    with torch.no_grad():
        for data, _ in target_train_loader:
            data = data.to(device)
            outputs = target_model(data)
            probs = softmax(outputs).cpu()
            target_member_probs.append(probs)
    target_member_probs = torch.cat(target_member_probs, dim=0)  # [60000, 10]
    target_member_labels = torch.ones(target_member_probs.size(0), dtype=torch.long)

    # 2) Target non‐members: MNIST test set (10k)
    target_test_loader = DataLoader(full_test_ds, batch_size=1000, shuffle=False, num_workers=2)
    target_nonmember_probs = []

    with torch.no_grad():
        for data, _ in target_test_loader:
            data = data.to(device)
            outputs = target_model(data)
            probs = softmax(outputs).cpu()
            target_nonmember_probs.append(probs)
    target_nonmember_probs = torch.cat(target_nonmember_probs, dim=0)  # [10000, 10]
    target_nonmember_labels = torch.zeros(target_nonmember_probs.size(0), dtype=torch.long)

    # Combine into a single evaluation dataset
    X_target_attack = torch.cat([target_member_probs, target_nonmember_probs], dim=0)
    Y_target_attack = torch.cat([target_member_labels, target_nonmember_labels], dim=0)
    target_attack_ds = TensorDataset(X_target_attack, Y_target_attack)
    target_attack_loader = DataLoader(target_attack_ds, batch_size=256, shuffle=False, num_workers=2)

    # Run the trained attack model on the target dataset
    attack_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for probs, labels in target_attack_loader:
            probs = probs.to(device)
            labels = labels.to(device)
            logits = attack_model(probs)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    final_acc = 100. * correct / total
    print("========================================================")
    print(f"Membership Inference Attack Results on Target Model:")
    print(f"  --> Total eval samples: {total}")
    print(f"  --> Attack accuracy (members vs non‐members): {final_acc:.2f}%")
    print("========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membership Inference Attack on MNIST Target Model")
    parser.add_argument("--num_shadows", type=int, default=5,
                        help="Number of shadow models to train (default: 5)")
    parser.add_argument("--shadow_epochs", type=int, default=5,
                        help="Epochs to train each shadow model (default: 5)")
    parser.add_argument("--shadow_train_size", type=int, default=10000,
                        help="Number of samples for each shadow’s training set (default: 10000)")
    parser.add_argument("--shadow_holdout_size", type=int, default=10000,
                        help="Number of samples for each shadow’s hold‐out set (default: 10000)")
    parser.add_argument("--attack_epochs", type=int, default=20,
                        help="Epochs to train the attack model (default: 20)")
    args = parser.parse_args()

    main(args)




