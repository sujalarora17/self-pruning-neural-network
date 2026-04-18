import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DeeperNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=False, num_workers=2)


def sparsity_loss(model):
    total = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.sum()
    return total


def test_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def calculate_sparsity(model):
    total = 0
    pruned = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < 1e-2).sum().item()
    return pruned / total if total > 0 else 0


def get_all_gates(model):
    gates_list = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            gates_list.extend(gates.flatten().tolist())
    return gates_list


def train_model(model_class, lambda_val, epochs=10, lr=0.001):
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'loss': [], 'acc': [], 'sparsity': []}

    print(f"\n  Model: {model_class.__name__} | Lambda: {lambda_val}")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target) + lambda_val * sparsity_loss(model)
            loss.backward()
            optimizer.step()

            total_loss += criterion(output, target).item()
            batches += 1

        avg_loss = total_loss / batches
        acc = test_accuracy(model)
        sp = calculate_sparsity(model)

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['sparsity'].append(sp)

        print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}% | Sparsity: {sp*100:.1f}%")

    return model, history


lambdas = [0.0001, 0.001, 0.01]
all_results = []
baseline_results = []
best_model = None
best_acc = 0
best_label = ""

print("=" * 60)
print("PHASE 1: Lambda comparison on BaselineNet")
print("=" * 60)

for lam in lambdas:
    model, history = train_model(BaselineNet, lam, epochs=10)
    acc = test_accuracy(model)
    sparsity = calculate_sparsity(model)

    entry = {
        'lambda': lam, 'acc': acc, 'sparsity': sparsity,
        'model': model, 'history': history,
        'label': f"Baseline_λ{lam}"
    }
    baseline_results.append(entry)
    all_results.append(entry)

    if acc > best_acc:
        best_acc, best_model, best_label = acc, model, f"BaselineNet λ={lam}"

    print(f"\n✅ Lambda: {lam} | Acc: {acc*100:.2f}% | Sparsity: {sparsity*100:.2f}%\n")


best_lam = baseline_results[
    max(range(len(baseline_results)), key=lambda i: baseline_results[i]['acc'])
]['lambda']

print("=" * 60)
print(f"PHASE 2: Architecture comparison at Lambda = {best_lam}")
print("=" * 60)

for arch in [DeeperNet, BatchNormNet]:
    model, history = train_model(arch, best_lam, epochs=10)
    acc = test_accuracy(model)
    sparsity = calculate_sparsity(model)

    entry = {
        'lambda': best_lam, 'acc': acc, 'sparsity': sparsity,
        'model': model, 'history': history,
        'label': f"{arch.__name__}_λ{best_lam}"
    }
    all_results.append(entry)

    if acc > best_acc:
        best_acc, best_model, best_label = acc, model, f"{arch.__name__} λ={best_lam}"

    print(f"\n✅ {arch.__name__} | Acc: {acc*100:.2f}% | Sparsity: {sparsity*100:.2f}%\n")


print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(f"{'Model':<30} {'Lambda':<10} {'Accuracy':<12} {'Sparsity'}")
print("-" * 65)
for r in all_results:
    print(f"{r['label']:<30} {r['lambda']:<10} {r['acc']*100:.2f}%{'':6} {r['sparsity']*100:.2f}%")
print(f"\nBest: {best_label} | Accuracy: {best_acc*100:.2f}%")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Self-Pruning Neural Network — Analysis", fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
gates = get_all_gates(best_model)
ax1.hist(gates, bins=100, color='steelblue', edgecolor='black', alpha=0.8)
ax1.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='Prune threshold')
ax1.set_title(f"Gate Distribution — {best_label}", fontsize=12)
ax1.set_xlabel("Gate Value")
ax1.set_ylabel("Count")
ax1.legend()

ax2 = axes[0, 1]
lam_vals  = [r['lambda']    for r in baseline_results]
acc_vals  = [r['acc'] * 100  for r in baseline_results]
spar_vals = [r['sparsity'] * 100 for r in baseline_results]
ax2.plot(lam_vals, acc_vals,  'bo-', linewidth=2, markersize=8, label='Accuracy (%)')
ax2.plot(lam_vals, spar_vals, 'rs-', linewidth=2, markersize=8, label='Sparsity (%)')
ax2.set_xscale('log')
ax2.set_title("Accuracy vs Sparsity — Lambda Tradeoff", fontsize=12)
ax2.set_xlabel("Lambda (log scale)")
ax2.set_ylabel("Percentage (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
for r in baseline_results:
    ax3.plot(r['history']['epoch'], r['history']['loss'],
             marker='o', linewidth=2, label=f"λ={r['lambda']}")
ax3.set_title("Training Loss Over Epochs", fontsize=12)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
for r in baseline_results:
    ax4.plot(r['history']['epoch'],
             [s * 100 for s in r['history']['sparsity']],
             marker='s', linewidth=2, label=f"λ={r['lambda']}")
ax4.set_title("Sparsity Growth During Training", fontsize=12)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Sparsity (%)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis.png")
