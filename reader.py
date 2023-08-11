import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_train(n, num_classes):
    images = []
    with open("./train-images-idx3-ubyte", "rb") as f:
        inp = f.read()
        inp = inp[16:]
        images = (
            torch.tensor(list(inp), device=device, dtype=torch.float).view(-1, n * n)
            / 255
        )

    with open("./train-labels-idx1-ubyte", "rb") as f:
        inp = f.read()
        inp = inp[8:]
        labels = F.one_hot(
            torch.tensor(list(inp), device=device), num_classes=num_classes
        ).type(torch.float)

    return images, labels


def read_test(n):
    test_images = []
    with open("./t10k-images-idx3-ubyte", "rb") as f:
        inp = f.read()
        inp = inp[16:]
        test_images = (
            torch.tensor(list(inp), device=device, dtype=torch.float).view(-1, n * n)
            / 255
        )

    with open("./t10k-labels-idx1-ubyte", "rb") as f:
        inp = f.read()
        inp = inp[8:]
        test_labels = torch.tensor(list(inp), device=device)

    return test_images, test_labels
