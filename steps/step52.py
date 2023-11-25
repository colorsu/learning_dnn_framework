if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.dataloaders import DataLoader


max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

if dezero.cuda.gpu_enable:
    print("GPU Enabled")
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss = 0
    start = time.time()

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
    elapsed_time = time.time() - start
    print('epoch: {}'.format(epoch+1))
    print('train loss: {: .4f}, time: {:.4f}'.format(sum_loss / len(train_set), elapsed_time))
