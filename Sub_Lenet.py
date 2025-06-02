import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import random

# 1. Define the substitute network architecture, following Lenet style
class SubstituteNet(nn.Module):
    def __init__(self):
        super(SubstituteNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels, 5×5 spatial after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # [B,6,14,14]
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # [B,16,5,5]
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)   # flatten to [B,16*5*5]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Normalize and load MNIST training set; take a random 10k subset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = list(range(len(full_train)))
    random.shuffle(indices)

    # Use 10,000 samples for substitute training
    subset_size = 10000
    subset_indices = indices[:subset_size]
    sub_train_ds = Subset(full_train, subset_indices)
    sub_train_loader = DataLoader(sub_train_ds, batch_size=64, shuffle=True, num_workers=2)

    print(f"Substitute will train on {subset_size} MNIST samples over {len(sub_train_loader)} batches per epoch.")

    # 3. Instantiate substitute model, optimizer, scheduler, loss
    model = SubstituteNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.NLLLoss()

    # 4. Train substitute for 6 epochs (matching style in a3_mnist.py)
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(sub_train_loader, start=1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                processed = batch_idx * len(data)
                total = subset_size
                percent = 100. * batch_idx / len(sub_train_loader)
                print(f"Sub Epoch: {epoch} [{processed}/{total} ({percent:.0f}%)]\tLoss: {loss.item():.6f}")

        scheduler.step()

    # Evaluate on full MNIST test set after each epoch
    model.eval()
    test_loss = 0.0
    correct = 0
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(full_test, batch_size=1000, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(full_test)
    accuracy = 100. * correct / len(full_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {correct}/{len(full_test)} ({accuracy:.2f}%)\n")

    # 5. Save substitute model parameters
    torch.save(model.state_dict(), "substitute_model.pth")


if __name__ == "__main__":
    main()
