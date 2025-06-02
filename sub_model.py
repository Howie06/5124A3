import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from mnist import MNIST
import random

# Define the substitute model
class SubstituteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*14*14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load full MNIST test set, then sample a smaller subset of 10000 for querying
    full_test = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)
    all_indices = list(range(len(full_test)))
    random.shuffle(all_indices)
    subset_size = 5000
    query_indices = all_indices[:subset_size]
    query_ds = Subset(full_test, query_indices)

    query_loader = DataLoader(query_ds, batch_size=200, shuffle=False, num_workers=2)


    # Load pretrained target model
    target = MNIST().to(device)
    target.load_state_dict(torch.load("target_model.pth"))
    target.eval()

    # Collect “noisy” labels from the target model’s predictions
    all_inputs, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(query_loader, start=1):
            data = data.to(device)
            out = target(data)
            preds = out.argmax(dim=1)
            all_inputs.append(data.cpu())
            all_labels.append(preds.cpu())

    sub_inputs = torch.cat(all_inputs, dim=0)   # [5000,1,28,28]
    sub_labels = torch.cat(all_labels, dim=0)   # [5000]
    print(f"Collected {sub_inputs.shape[0]} samples for substitute training.")

    # Prepare substitute dataset and loader
    sub_ds = TensorDataset(sub_inputs, sub_labels)
    batch_size = 64
    sub_loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # Calculate iterations per epoch
    iterations_per_epoch = len(sub_ds) // batch_size + (1 if len(sub_ds) % batch_size else 0)
    num_epochs = 5
    batches_per_epoch = len(sub_loader)
    total_iterations = iterations_per_epoch * num_epochs
    print(f"Substitute training: {num_epochs} epochs, batch size {batch_size}, "
          f"{iterations_per_epoch} iterations/epoch, {total_iterations} total iterations.")

    # Instantiate substitute model
    sub_model = SubstituteModel().to(device)
    sub_opt   = optim.Adam(sub_model.parameters(), lr=0.001)
    sub_crit  = nn.CrossEntropyLoss()

    # Train the substitute model on the extracted labels
    for epoch in range(1, num_epochs + 1):
        sub_model.train()
        run_loss = 0.0
        for bidx, (data, target_label) in enumerate(sub_loader, start=1):
            data, target_label = data.to(device), target_label.to(device)
            sub_opt.zero_grad()
            out = sub_model(data)
            loss = sub_crit(out, target_label)
            loss.backward()
            sub_opt.step()

            run_loss += loss.item()
            if batch_idx % 200 == 0:
                avg = running_loss / 200
                samples_processed = batch_idx * data.size(0)
                percent = 100. * batch_idx / batches_per_epoch
                print(f"[Target] Epoch {epoch} Batch {batch_idx} "
                      f"[{samples_processed}/{subset_size} ({percent:.0f}%)]  Loss: {avg:.4f}")
                running_loss = 0.0

    # Evaluate substitute on the full MNIST test set (5k samples) for utility
    full_test_loader = DataLoader(full_test, batch_size=1000, shuffle=False, num_workers=0)
    sub_model.eval()
    correct_sub = 0
    total_sub = 0
    with torch.no_grad():
        for x, y in full_test_loader:
            x, y = x.to(device), y.to(device)
            preds = sub_model(x).argmax(dim=1)
            correct_sub += preds.eq(y).sum().item()
            total_sub += y.size(0)
    sub_accuracy = 100 * correct_sub / total_sub

    # Evaluate fidelity against clean target predictions on full test set
    fidelity_count = 0
    with torch.no_grad():
        for x, _ in full_test_loader:
            x = x.to(device)
            t_preds = target(x).argmax(dim=1)
            s_preds = sub_model(x).argmax(dim=1)
            fidelity_count += (t_preds == s_preds).sum().item()
    fidelity = 100 * fidelity_count / total_sub

    print("\n=== Final Results ===")
    print(f"Substitute Accuracy on True Test (10k): {sub_accuracy:.2f}%")
    print(f"Fidelity with Target on Full Test:      {fidelity:.2f}%")


    # Save the substitute model
    torch.save(sub_model.state_dict(), "substitute_model.pth")


if __name__ == "__main__":
    main()

