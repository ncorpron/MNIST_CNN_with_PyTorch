import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from models import SimpleCNN
from utils import set_seed

# --- Setup
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_ds = datasets.MNIST(root=".", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

# --- Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_best.pth", map_location=device))
model.eval()

# --- Evaluate accuracy
correct = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(f"Test accuracy: {correct/len(test_ds):.4f}")

# --- Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- Sample predictions
fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(test_ds))
    img, label = test_ds[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"True:{label}, Pred:{pred}")
    ax.axis("off")
plt.show()
