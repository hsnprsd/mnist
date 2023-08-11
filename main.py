import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dnn import MNIST
from reader import read_test, read_train

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, test_images, test_labels):
    model.eval()
    total = test_images.shape[0]
    out = torch.argmax(model(test_images), dim=1, keepdim=False)
    correct = torch.eq(out, test_labels).sum()
    acc = correct / total
    model.train()
    return acc


def train():
    n = 28
    num_classes = 10

    images, labels = read_train(n, num_classes)
    test_images, test_labels = read_test(n)

    model = MNIST(
        input_size=n * n,
        hidden_size=128,
        num_classes=num_classes,
    ).to(device)
    if os.environ["LOAD_MODEL"] == "1":
        model = torch.load("model.pickle")

    optim = torch.optim.SGD(model.parameters(), lr=5e-4)

    epochs = 100000
    pbar = tqdm(range(epochs))
    losses = []
    for step in pbar:
        ix = torch.randint(0, len(images), (512,))
        inp, tar = images[ix], labels[ix]
        out = model(inp)

        optim.zero_grad()
        loss = F.cross_entropy(out, tar)
        pbar.set_description(f"loss = {loss.item():.2f}")
        losses.append(loss.item())
        loss.backward()
        optim.step()

        if step % 1000 == 0:
            acc = evaluate(model, test_images, test_labels)
            print("Accuracy: {}%".format(100 * acc))

    torch.save(model, "model.pickle")

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    train()
