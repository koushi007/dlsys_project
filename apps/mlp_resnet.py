import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    softmax_loss = nn.SoftmaxLoss()
    training_loss = 0.0
    error_rate = 0.0
    num_batches = 0
    num_samples = len(dataloader.dataset)

    if opt is not None:
        model.train()
    else:
        model.eval()

    for X, y in dataloader:
        if opt is not None:
            opt.reset_grad()

        logits = model(X.reshape((X.shape[0], -1)))
        loss = softmax_loss(logits, y)

        if opt is not None:
            loss.backward()
            opt.step()

        training_loss += loss.numpy()
        error_rate += np.sum(np.argmax(logits.numpy(), axis=1) != y.numpy())
        num_batches += 1

    return error_rate/num_samples, training_loss/num_batches
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        data_dir+"/train-images-idx3-ubyte.gz", data_dir+"/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = ndl.data.MNISTDataset(
        data_dir+"/t10k-images-idx3-ubyte.gz", data_dir+"/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    trajectories = []
    for e in range(epochs):
        # start_time = time.time()
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        # elapsed_time = time.time() - start_time
        # print(
        #     f"Epoch {e+1} | Train error: {train_error:.2f} | Test error: {test_error:.2f} | "
        #     f"Train loss: {train_loss:.2f} | Test loss: {test_loss:.2f} | "
        #     f"Time(s): {elapsed_time:.2f}"
        # )
        trajectories.append((train_error, train_loss, test_error, test_loss))

    return trajectories[-1]
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
