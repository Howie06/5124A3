import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from target_model import MNIST

# ---------------------------------------
# Utility: Train a model on a DataLoader
# ---------------------------------------
def train_model(model, dataloader, epochs, lr, device):
    """
    Train `model` on the data provided by `dataloader` for `epochs` epochs,
    using Adam optimizer with learning rate `lr`. Training happens on `device`.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# ----------------------------------------------------
# Utility: Compute softmax posteriors for each sample
# ----------------------------------------------------
def get_posteriors(model, loader, device):
    """
    Given a trained `model` and a `loader` providing (data, label) batches,
    return two NumPy arrays:
      - posteriors: shape (num_samples, 10), softmax probabilities
      - true_labels: shape (num_samples,), the ground-truth digit labels
    """
    model.eval()
    all_posteriors = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            logits = model(data)                 # [batch_size, 10]
            probs = F.softmax(logits, dim=1)     # convert logits → probabilities
            all_posteriors.append(probs.cpu().numpy())
            all_labels.append(target.numpy())
    all_posteriors = np.concatenate(all_posteriors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_posteriors, all_labels

# ----------------------------------------
# Train multiple shadow models & gather data
# ----------------------------------------
def train_shadow_models(train_dataset_full, num_shadows, shadow_epochs, device):
    """
    Given the full MNIST training dataset `train_dataset_full`, split it into
    `num_shadows` disjoint subsets to train `num_shadows` shadow models.
    Each shadow model is trained for `shadow_epochs` epochs. Returns:
      - all_shadow_posteriors: (N, 10) array of softmax vectors from shadow models
      - all_shadow_labels:     (N,)  array of the true digit labels
      - all_shadow_membership: (N,)  array of membership labels (1=member, 0=non-member)
    """
    shadow_posteriors = []
    shadow_labels = []
    shadow_membership = []

    n_total = len(train_dataset_full)
    subset_size = n_total // num_shadows

    for i in range(num_shadows):
        # Determine indices for shadow’s own training portion vs. the rest
        start_idx = i * subset_size
        end_idx = start_idx + subset_size if i < num_shadows - 1 else n_total

        shadow_train_indices = list(range(start_idx, end_idx))
        shadow_test_indices = list(range(0, start_idx)) + list(range(end_idx, n_total))

        shadow_train_subset = Subset(train_dataset_full, shadow_train_indices)
        shadow_test_subset  = Subset(train_dataset_full, shadow_test_indices)

        shadow_train_loader = DataLoader(shadow_train_subset, batch_size=128, shuffle=True, num_workers=2)
        shadow_test_loader  = DataLoader(shadow_test_subset, batch_size=128, shuffle=False, num_workers=2)

        # Initialize and train one shadow model
        shadow_model = MNIST().to(device)
        print(f"[INFO] Training shadow model {i+1}/{num_shadows}...")
        train_model(shadow_model, shadow_train_loader, shadow_epochs, lr=0.001, device=device)

        # Gather “member” posteriors (shadow’s own training set)
        post_train, labels_train = get_posteriors(shadow_model, shadow_train_loader, device)
        shadow_posteriors.append(post_train)
        shadow_labels.append(labels_train)
        shadow_membership.append(np.ones_like(labels_train))  # 1 = member

        # Gather “non‐member” posteriors (the rest of the training data)
        post_test, labels_test = get_posteriors(shadow_model, shadow_test_loader, device)
        shadow_posteriors.append(post_test)
        shadow_labels.append(labels_test)
        shadow_membership.append(np.zeros_like(labels_test))  # 0 = non‐member

        print(f"[INFO] Shadow {i+1} collected: members={len(labels_train)}, non-members={len(labels_test)}\n")

    # Concatenate all shadow data
    all_shadow_posteriors = np.concatenate(shadow_posteriors, axis=0)
    all_shadow_labels     = np.concatenate(shadow_labels, axis=0)
    all_shadow_membership = np.concatenate(shadow_membership, axis=0)

    total_samples = all_shadow_posteriors.shape[0]
    print(f"[INFO] Total shadow dataset size: {total_samples} samples.\n")
    return all_shadow_posteriors, all_shadow_labels, all_shadow_membership

# ----------------------------------------
# Define the per-class Attack MLP
# ----------------------------------------
class AttackMLP(nn.Module):
    def __init__(self, input_dim=10):
        super(AttackMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # binary output: [non-member, member]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Main execution under guard
# -----------------------------
def main():
    # ----------------------------
    # Device & File Path Setup
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}\n")

    model = MNIST()
    model.load_state_dict(torch.load('target_model.pth'))
    model.eval()

    # ----------------------------
    # Prepare MNIST Datasets
    # ----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Full MNIST training set (for target & shadows)
    train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # MNIST test set (for final evaluation’s non-members)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # ----------------------------
    # Load or Train the Target Model
    # ----------------------------
    target_model = MNIST().to(device)
    loaded_ok = False

    if os.path.exists('target_model.pth'):
        try:
            state_dict = torch.load('target_model.pth', map_location=device)
            target_model.load_state_dict(state_dict)
            print("[INFO] Successfully loaded provided target model.\n")
            loaded_ok = True
        except Exception as e:
            print("[WARNING] Failed to load provided target model. Will train a new one.")
            print(f"Error: {e}\n")

    if not loaded_ok:
        # Train the target model on the entire MNIST training set
        full_train_loader = DataLoader(train_dataset_full, batch_size=128, shuffle=True, num_workers=2)
        print("[INFO] Training a new target model on full MNIST training set...")
        train_model(target_model, full_train_loader, epochs=5, lr=0.001, device=device)
        torch.save(target_model.state_dict(), 'target_model.pth')
        print("[INFO] Saved newly trained target model.\n")

    target_model.eval()

    # ---------------------------------------------------
    # Train Shadow Models & Collect Shadow Posteriors
    # ---------------------------------------------------
    num_shadows = 2
    shadow_epochs = 3

    (shadow_posteriors,
     shadow_labels,
     shadow_membership) = train_shadow_models(train_dataset_full,
                                              num_shadows,
                                              shadow_epochs,
                                              device)

    # ----------------------------------------------------
    # Train One Attack Model per Class (0 through 9)
    # ----------------------------------------------------
    attack_models = {}
    attack_criterions = {}
    attack_optimizers = {}

    for digit in range(10):
        # Filter shadow data for this true class
        class_indices = np.where(shadow_labels == digit)[0]
        class_posteriors = shadow_posteriors[class_indices]    # shape (M, 10)
        class_membership = shadow_membership[class_indices]    # shape (M,)

        # Convert to torch tensors
        X = torch.tensor(class_posteriors, dtype=torch.float32)
        y = torch.tensor(class_membership, dtype=torch.long)

        # Shuffle & split 80/20 for training/testing the attack model
        num_samples = len(y)
        indices = list(range(num_samples))
        random.shuffle(indices)
        split = int(0.8 * num_samples)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data  = torch.utils.data.TensorDataset(X_test,  y_test)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

        # Initialize attack model, optimizer, loss
        attack_model = AttackMLP().to(device)
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train attack model for this digit
        epochs_attack = 5
        print(f"[INFO] Training AttackMLP for digit {digit} with {len(y_train)} samples...")
        for epoch in range(epochs_attack):
            attack_model.train()
            total_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = attack_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_data.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"  Epoch {epoch+1}/{epochs_attack} - Loss: {avg_loss:.4f}")
        print()

        # Evaluate on hold-out portion
        attack_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                logits = attack_model(batch_data)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)
        acc = correct / total * 100
        print(f"[INFO] AttackMLP digit {digit} test accuracy: {acc:.2f}%\n")

        # Store the trained attack model
        attack_models[digit] = attack_model
        attack_optimizers[digit] = optimizer
        attack_criterions[digit] = criterion

    # -------------------------------------------------
    # Final Evaluation on Target Model (10,000 Samples)
    # -------------------------------------------------
    # We'll pick 5,000 from the target’s training set (members)
    # and 5,000 from the target’s test set (non-members).
    all_train_indices = list(range(len(train_dataset_full)))
    random.shuffle(all_train_indices)
    member_indices = all_train_indices[:5000]
    member_subset = Subset(train_dataset_full, member_indices)
    member_loader = DataLoader(member_subset, batch_size=128, shuffle=False)

    all_test_indices = list(range(len(test_dataset)))
    random.shuffle(all_test_indices)
    nonmember_indices = all_test_indices[:5000]
    nonmember_subset = Subset(test_dataset, nonmember_indices)
    nonmember_loader = DataLoader(nonmember_subset, batch_size=128, shuffle=False)

    # Gather posteriors & true labels for members
    post_mem, labels_mem = get_posteriors(target_model, member_loader, device)
    # Gather posteriors & true labels for non-members
    post_nonmem, labels_nonmem = get_posteriors(target_model, nonmember_loader, device)

    eval_posteriors     = np.concatenate([post_mem,      post_nonmem], axis=0)  # shape (10000, 10)
    eval_labels         = np.concatenate([labels_mem,   labels_nonmem], axis=0) # shape (10000,)
    eval_true_membership= np.concatenate([np.ones_like(labels_mem),
                                           np.zeros_like(labels_nonmem)], axis=0)

    # Attack: route each sample’s posterior through the per-class attack model
    attacker_preds = []
    for idx in range(len(eval_labels)):
        true_digit = int(eval_labels[idx])
        posterior_vec = torch.tensor(eval_posteriors[idx], dtype=torch.float32).unsqueeze(0).to(device)
        attack_model = attack_models[true_digit]
        attack_model.eval()
        with torch.no_grad():
            logits = attack_model(posterior_vec)
            pred = torch.argmax(logits, dim=1).item()  # 1 = member, 0 = non-member
        attacker_preds.append(pred)

    attacker_preds = np.array(attacker_preds)
    total_correct = (attacker_preds == eval_true_membership).sum()
    overall_acc = total_correct / len(attacker_preds) * 100

    # Compute TPR and TNR
    tp = ((attacker_preds == 1) & (eval_true_membership == 1)).sum()
    tn = ((attacker_preds == 0) & (eval_true_membership == 0)).sum()
    tpr = tp / len(labels_mem) * 100
    tnr = tn / len(labels_nonmem) * 100

    print("=== Final Membership Inference Attack Results ===")
    print(f"Attack Accuracy (10000 samples): {overall_acc:.2f}%")
    print(f"True Positive Rate (Members detected): {tp}/{len(labels_mem)} = {tpr:.2f}%")
    print(f"True Negative Rate (Non-members detected): {tn}/{len(labels_nonmem)} = {tnr:.2f}%")

if __name__ == '__main__':
    main()