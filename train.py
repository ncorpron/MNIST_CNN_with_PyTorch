import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN
from utils import set_seed

# --- Setup
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# --- Model, loss, optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop
best_acc = 0.0
history = {"loss": [], "acc": [], "val_acc": []}

for epoch in range(10):  # can adjust epochs
    model.train()
    total_loss, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    train_loss = total_loss / len(train_ds)
    train_acc = correct / len(train_ds)

    # Validate
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    val_acc = correct / len(test_ds)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/mnist_best.pth")
        print(f"✅ Saved new best model at epoch {epoch+1} with val_acc={val_acc:.4f}")

    history["loss"].append(train_loss)
    history["acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")

