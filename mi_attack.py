import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import os
import random
from mnist import MNIST

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple CNN architecture for MNIST (common LeNet-like)

# Paths
TARGET_MODEL_PATH = '/data/target_model.pth'

# Prepare MNIST data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download MNIST dataset
train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Function to train a model on given DataLoader
def train_model(model, dataloader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
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

# Function to compute posteriors (softmax probabilities) for all samples in a DataLoader
def get_posteriors(model, loader):
    model.eval()
    all_posteriors = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)  # [batch, 10]
            all_posteriors.append(probs.cpu().numpy())
            all_labels.append(target.numpy())
    all_posteriors = np.concatenate(all_posteriors, axis=0)  # [num_samples, 10]
    all_labels = np.concatenate(all_labels, axis=0)          # [num_samples]
    return all_posteriors, all_labels

# 1. Load (or train) the Target Model
model = MNIST()
model.load_state_dict(torch.load('target_model.pth'))
model.eval()
loaded_correctly = False

# Attempt to load provided target model
if os.path.exists(TARGET_MODEL_PATH):
    try:
        state_dict = torch.load(TARGET_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("[INFO] Successfully loaded provided target model.")
        loaded_correctly = True
    except Exception as e:
        print("[WARNING] Failed to load provided target model. Proceeding to train a new one.")
        print(f"Error: {e}")

# If loading failed or file not found, train a new target model on full MNIST train set
if not loaded_correctly:
    full_train_loader = DataLoader(train_dataset_full, batch_size=128, shuffle=True, num_workers=2)
    print("[INFO] Training a new target model on full MNIST training set...")
    train_model(model, full_train_loader, epochs=5, lr=0.001)
    torch.save(model.state_dict(), TARGET_MODEL_PATH)
    print("[INFO] Saved newly trained target model.")

# 2. Construct Shadow Models and Gather Posteriors
def train_shadow_models(num_shadows=2, shadow_epochs=3):
    shadow_posteriors = []
    shadow_labels = []
    shadow_membership = []  # 1 for member, 0 for non-member

    # Split full training set into equal disjoint chunks for shadows
    n_total = len(train_dataset_full)
    subset_size = n_total // num_shadows

    for i in range(num_shadows):
        # Determine indices for shadow training and testing data
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        if i == num_shadows - 1:
            end_idx = n_total

        # Shadow dataset split
        shadow_train_indices = list(range(start_idx, end_idx))
        shadow_test_indices = list(range(0, start_idx)) + list(range(end_idx, n_total))

        shadow_train_subset = Subset(train_dataset_full, shadow_train_indices)
        shadow_test_subset = Subset(train_dataset_full, shadow_test_indices)

        shadow_train_loader = DataLoader(shadow_train_subset, batch_size=128, shuffle=True, num_workers=2)
        shadow_test_loader = DataLoader(shadow_test_subset, batch_size=128, shuffle=False, num_workers=2)

        # Initialize and train shadow model
        shadow_model = MNIST().to(device)
        print(f"[INFO] Training shadow model {i+1}/{num_shadows}...")
        train_model(shadow_model, shadow_train_loader, epochs=shadow_epochs, lr=0.001)

        # Gather posteriors for shadow "members"
        post_train, labels_train = get_posteriors(shadow_model, shadow_train_loader)
        shadow_posteriors.append(post_train)
        shadow_labels.append(labels_train)
        shadow_membership.append(np.ones_like(labels_train))  # all 1s for members

        # Gather posteriors for shadow "non-members"
        post_test, labels_test = get_posteriors(shadow_model, shadow_test_loader)
        shadow_posteriors.append(post_test)
        shadow_labels.append(labels_test)
        shadow_membership.append(np.zeros_like(labels_test))  # all 0s for non-members

        print(f"[INFO] Shadow model {i+1} posteriors collected: members({len(labels_train)}), non-members({len(labels_test)})\n")

    # Concatenate all shadow data
    all_shadow_posteriors = np.concatenate(shadow_posteriors, axis=0)
    all_shadow_labels = np.concatenate(shadow_labels, axis=0)
    all_shadow_membership = np.concatenate(shadow_membership, axis=0)

    print(f"[INFO] Total shadow dataset size: {all_shadow_posteriors.shape[0]} samples.")
    return all_shadow_posteriors, all_shadow_labels, all_shadow_membership

# Train shadow models and collect data
shadow_posteriors, shadow_labels, shadow_membership = train_shadow_models(num_shadows=2, shadow_epochs=3)

# 3. Train Class-Specific Attack Models
# We'll train one attack model per class (0-9). Each attack model is a simple MLP.
class AttackMLP(nn.Module):
    def __init__(self, input_dim=10):
        super(AttackMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification: member (1) vs non-member (0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare data for each class
attack_models = {}
attack_optimizers = {}
attack_criterions = {}

for label in range(10):
    # Select shadow samples of a specific true class
    class_indices = np.where(shadow_labels == label)[0]
    class_posteriors = shadow_posteriors[class_indices]
    class_membership = shadow_membership[class_indices]

    # Convert to tensors
    X = torch.tensor(class_posteriors, dtype=torch.float32)
    y = torch.tensor(class_membership, dtype=torch.long)

    # Shuffle and split into train/test for attack model
    dataset_size = len(y)
    split_point = int(0.8 * dataset_size)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Create DataLoaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Initialize attack model, optimizer, and loss
    model_attack = AttackMLP().to(device)
    optimizer_attack = optim.Adam(model_attack.parameters(), lr=0.001)
    criterion_attack = nn.CrossEntropyLoss()

    # Train attack model
    epochs_attack = 5
    print(f"[INFO] Training attack model for class {label} with {len(y_train)} training samples...")
    for epoch in range(epochs_attack):
        model_attack.train()
        total_loss = 0
        for data_batch, label_batch in train_loader:
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            optimizer_attack.zero_grad()
            outputs = model_attack(data_batch)
            loss = criterion_attack(outputs, label_batch)
            loss.backward()
            optimizer_attack.step()
            total_loss += loss.item() * data_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1}/{epochs_attack} - Loss: {avg_loss:.4f}")

    # Evaluate attack model
    model_attack.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, label_batch in test_loader:
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            outputs = model_attack(data_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)
    accuracy = correct / total * 100
    print(f"  Attack Model Class {label} Test Accuracy: {accuracy:.2f}%\n")

    # Save trained attack model and details
    attack_models[label] = model_attack
    attack_optimizers[label] = optimizer_attack
    attack_criterions[label] = criterion_attack

# 4. EvaluateAttack on Target Model Using 10000 Samples
#    We select 5000 members and 5000 non-members from the target's MNIST train/test.

# Sample 5000 from target model's training data (members)
all_train_indices = list(range(len(train_dataset_full)))
random.shuffle(all_train_indices)
member_indices = all_train_indices[:5000]
member_subset = Subset(train_dataset_full, member_indices)
member_loader = DataLoader(member_subset, batch_size=128, shuffle=False)

# Sample 5000 from target model's test data (non-members)
all_test_indices = list(range(len(test_dataset)))
random.shuffle(all_test_indices)
nonmember_indices = all_test_indices[:5000]
nonmember_subset = Subset(test_dataset, nonmember_indices)
nonmember_loader = DataLoader(nonmember_subset, batch_size=128, shuffle=False)

# Get posteriors and true labels from target model
post_mem, labels_mem = get_posteriors(model, member_loader)
post_nonmem, labels_nonmem = get_posteriors(model, nonmember_loader)

# Combine for evaluation
eval_posteriors = np.concatenate([post_mem, post_nonmem], axis=0)  # [10000, 10]
eval_labels = np.concatenate([labels_mem, labels_nonmem], axis=0)  # [10000]
eval_true_membership = np.concatenate([np.ones_like(labels_mem), np.zeros_like(labels_nonmem)], axis=0)  # [10000]

# Use the attack models to predict membership
attacker_preds = []
for i in range(len(eval_labels)):
    true_class = eval_labels[i]
    posterior = torch.tensor(eval_posteriors[i], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 10]
    attack_model = attack_models[int(true_class)]
    attack_model.eval()
    with torch.no_grad():
        logits = attack_model(posterior)
        pred = torch.argmax(logits, dim=1).item()  # 1 for member, 0 for non-member
    attacker_preds.append(pred)

attacker_preds = np.array(attacker_preds)

# Compute overall attack accuracy
correct_preds = (attacker_preds == eval_true_membership).sum()
attack_accuracy = correct_preds / len(attacker_preds) * 100
print(f"=== Final Membership Inference Attack Accuracy on 10000 samples ===")
print(f"Attack Accuracy: {attack_accuracy:.2f}%")

# Compute TPR and TNR
tp = ((attacker_preds == 1) & (eval_true_membership == 1)).sum()
tn = ((attacker_preds == 0) & (eval_true_membership == 0)).sum()
tpr = tp / len(labels_mem) * 100
tnr = tn / len(labels_nonmem) * 100
print(f"True Positive Rate (Members correctly detected): {tp}/{len(labels_mem)} = {tpr:.2f}%")
print(f"True Negative Rate (Non-members correctly detected): {tn}/{len(labels_nonmem)} = {tnr:.2f}%")






