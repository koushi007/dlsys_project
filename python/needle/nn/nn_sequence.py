"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1)**-1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1/hidden_size
        bound = k**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        if bias:
            self.bias = True
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        else:
            self.bias = False
        
        if nonlinearity == 'tanh':
            self.nonlinearity = ops.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = ops.relu
        else:
            raise ValueError(f"Unknown nonlinearity {nonlinearity}")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=X.dtype)
        out = (X @ self.W_ih) + (h @ self.W_hh)
        if self.bias:
            out = out + (self.bias_ih + self.bias_hh).reshape((1, self.hidden_size))\
                                                     .broadcast_to(out.shape)

        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        self.rnn_cells.extend([RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers-1)])
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h = [None] * self.num_layers
        else:
            h = list(ops.split(h0, axis=0).tuple())
        
        seq_len = X.shape[0]
        outputs = [None] * seq_len
        sequence = list(ops.split(X, axis=0).tuple())

        for t in range(seq_len):
            h[0] = self.rnn_cells[0](sequence[t], h[0])
            for i in range(1, self.num_layers):
                h[i] = self.rnn_cells[i](h[i-1], h[i])
            outputs[t] = h[-1]

        return ops.stack(outputs, axis=0), ops.stack(h, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1/hidden_size
        bound = k**0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        if bias:
            self.bias = True
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, dtype=dtype, device=device))
        else:
            self.bias = False

        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=X.dtype)
            c = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=X.dtype)
            h = (h, c)

        out = (X @ self.W_ih) + (h[0] @ self.W_hh)
        if self.bias:
            out = out + (self.bias_ih + self.bias_hh).reshape((1, 4*self.hidden_size))\
                                                     .broadcast_to(out.shape)
        out = out.reshape((out.shape[0], 4, self.hidden_size))
        i,f,g,o = (w.reshape((out.shape[0], self.hidden_size)) for w in ops.split(out, axis=1).tuple())
        i,f,g,o = self.sigmoid(i), self.sigmoid(f), ops.tanh(g), self.sigmoid(o)
        
        c = i * g
        c = c + f * h[1]
        h = o * ops.tanh(c)

        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        self.lstm_cells.extend([LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers-1)])
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = [None] * self.num_layers
        else:
            c = list(ops.split(h[1], axis=0).tuple())
            h = list(ops.split(h[0], axis=0).tuple())
            h = [(h[i], c[i]) for i in range(self.num_layers)]
        
        seq_len = X.shape[0]
        outputs = [None] * seq_len
        sequence = list(ops.split(X, axis=0).tuple())

        for t in range(seq_len):
            h[0] = self.lstm_cells[0](sequence[t], h[0])
            for i in range(1, self.num_layers):
                h[i] = self.lstm_cells[i](h[i-1][0], h[i])
            outputs[t] = h[-1][0]

        c = ops.stack([h[i][1] for i in range(self.num_layers)], axis=0)
        h = ops.stack([h[i][0] for i in range(self.num_layers)], axis=0)
        return ops.stack(outputs, axis=0), (h, c)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.n = num_embeddings
        self.dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x = list(ops.split(x, axis=0).tuple())
        for i in range(len(x)):
            x[i] = init.one_hot(self.n, x[i], device=x[i].device, dtype=x[i].dtype)
            x[i] = x[i] @ self.weight
        return ops.stack(x, axis=0)
        ### END YOUR SOLUTION