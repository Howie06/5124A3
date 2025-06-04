import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the target model
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*14*14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))              # [B,32,28,28]
        x = self.pool(F.relu(self.conv2(x)))    # [B,64,14,14]
        x = x.view(x.size(0), -1)               # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)                      # logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load the full MNIST training set (60,000 samples) and test set
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False, num_workers=2)

    model     = MNIST().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train the target model
    num_epochs = 5
    total_samples = len(train_ds)  # 60000
    batches_per_epoch = len(train_loader)

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Print every 200 mini-batches, indicating how many samples processed so far
            if batch_idx % 200 == 0:
                avg = running_loss / 200
                samples_processed = batch_idx * data.size(0)
                percent = 100. * batch_idx / batches_per_epoch
                print(f"[Target] Epoch {epoch} Batch {batch_idx} "
                      f"[{samples_processed}/{total_samples} ({percent:.0f}%)]  Loss: {avg:.4f}")
                running_loss = 0.0

        # evaluate on true test set


    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += criterion(out, target).item() * data.size(0)
            preds = out.argmax(dim=1)
            correct += preds.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

    # Simulate black-box querying of the target model using all 60,000 training samples
    model.eval()
    all_inputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader, start=1):
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_inputs.append(data.cpu())
            all_labels.append(preds.cpu())

    # Save the model
    torch.save(model.state_dict(), 'target_model.pth')

if __name__ == "__main__":
    main()