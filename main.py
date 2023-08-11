import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cnn
from reader import read_test, read_train

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

N = 28
NUM_CLASSES = 10


@torch.no_grad()
def evaluate(model, test_images, test_labels):
    model.eval()
    total = test_images.shape[0]
    inp_splits = torch.split(test_images.view(total, 1, N, N), 100)
    tar_splits = torch.split(test_labels, 100)
    correct = 0
    for inp, tar in zip(inp_splits, tar_splits):
        out = torch.argmax(model(inp), dim=1, keepdim=False)
        correct += torch.eq(out, tar).sum()
    acc = correct / total
    model.train()
    return acc


def train():
    images, labels = read_train(N, NUM_CLASSES)
    test_images, test_labels = read_test(N)

    model = cnn.CNN(N, NUM_CLASSES).to(device)
    if os.environ.get("LOAD_MODEL", "0") == "1":
        model = torch.load("convnet.pickle")

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    epochs = 5000
    pbar = tqdm(range(epochs))
    losses = []
    for step in pbar:
        B = 256
        ix = torch.randint(0, len(images), (B,))
        inp, tar = images[ix], labels[ix]
        inp = inp.view(B, 1, N, N)
        out = model(inp)

        optim.zero_grad()
        loss = F.cross_entropy(out, tar)
        pbar.set_description(f"loss = {loss.item():.2f}")
        losses.append(loss.item())
        loss.backward()
        optim.step()

        if step % 500 == 0:
            acc = evaluate(model, test_images, test_labels)
            print("accuracy: {}%".format(100 * acc))

    torch.save(model, "convnet.pickle")

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    train()
