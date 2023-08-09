import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"


N = 28
CLASSES = 10

images = []
with open("./train-images-idx3-ubyte", "rb") as f:
    inp = f.read()
    inp = inp[16:]
    images = (
        torch.tensor(list(inp), device=device, dtype=torch.float).view(-1, N * N) / 255
    )

with open("./train-labels-idx1-ubyte", "rb") as f:
    inp = f.read()
    inp = inp[8:]
    labels = F.one_hot(
        torch.tensor(list(inp), device=device), num_classes=CLASSES
    ).type(torch.float)

test_images = []
with open("./t10k-images-idx3-ubyte", "rb") as f:
    inp = f.read()
    inp = inp[16:]
    test_images = (
        torch.tensor(list(inp), device=device, dtype=torch.float).view(-1, N * N) / 255
    )

with open("./t10k-labels-idx1-ubyte", "rb") as f:
    inp = f.read()
    inp = inp[8:]
    test_labels = torch.tensor(list(inp), device=device)


class MNIST(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(N * N, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, CLASSES),
        )

    def forward(self, x):
        out = F.softmax(self.nn(x))
        return out


model = MNIST(hidden_size=128).to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 1000000
pbar = tqdm(range(epochs))
losses = []
for step in pbar:
    ix = torch.randint(0, len(images), (512,))
    inp, tar = images[ix], labels[ix]
    out = model(images[ix])

    optim.zero_grad()
    loss = F.cross_entropy(out, tar)
    pbar.set_description(f"loss = {loss.item():.2f}")
    losses.append(loss.item())
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        model.eval()
        total = test_images.shape[0]
        out = torch.argmax(model(test_images), dim=1, keepdim=False)
        correct = torch.eq(out, test_labels).sum()
        print("Accuracy: {}%".format(100 * correct / total))
        model.train()

plt.plot(losses)
plt.show()
