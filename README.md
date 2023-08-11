# MNIST

## Models
- 2 layer DNN
    - hyper params
        - epochs = 1000_000
        - hidden size = 128
        - batch size = 512
        - lr = 5e-4
    - evaluation
        - accuracy: ~91.5%
- CNN
    - evaluation
        - accuracy: ~99%

## Training
```
LOAD_MODEL=0|1 python3 main.py
```
Saves model to `model.pickle` file.
