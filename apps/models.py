import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype

        def ConvBN(a, b, k, s):
            return nn.Sequential(
                nn.Conv(in_channels=a, out_channels=b, kernel_size=k, stride=s,
                        device=device, dtype=dtype),
                nn.BatchNorm2d(b, device=device, dtype=dtype),
                nn.ReLU()
            )
        
        self.block1 = nn.Sequential(
            ConvBN(3, 16, 7, 4),
            ConvBN(16, 32, 3, 2)
        )
        self.block2 = nn.Sequential(
            ConvBN(32, 32, 3, 1),
            ConvBN(32, 32, 3, 1)
        )
        self.block3 = nn.Sequential(
            ConvBN(32, 64, 3, 2),
            ConvBN(64, 128, 3, 2)
        )
        self.block4 = nn.Sequential(
            ConvBN(128, 128, 3, 1),
            ConvBN(128, 128, 3, 1)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2 + x1)
        x4 = self.block4(x3)
        x5 = self.linear(x4 + x3)
        return x5
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.seq_model = nn.Transformer(embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, sequence_len=seq_len, device=device, dtype=dtype)
            self.linear = nn.Linear(in_features=embedding_size, out_features=output_size, device=device, dtype=dtype)
        else:
            raise ValueError("seq_model must be 'rnn', 'lstm', or 'transformer'")
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        em = self.embedding(x)
        out, h = self.seq_model(em, h)
        if isinstance(self.seq_model, nn.RNN) or isinstance(self.seq_model, nn.LSTM):
            out = out.reshape((x.shape[0]*x.shape[1], self.hidden_size))
        elif isinstance(self.seq_model, nn.Transformer):
            out = out.reshape((x.shape[0]*x.shape[1], self.embedding_size))
        return self.linear(out), h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)