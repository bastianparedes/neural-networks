import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output, lr):
        error = y - output
        d_output = error * sigmoid_derivative(output)

        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(d_output) * lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr

        self.W1 += X.T.dot(d_hidden) * lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

        return np.mean(np.abs(error))


def train():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    y = np.array([[0], [1], [1], [0]])

    nn = SimpleNN(input_size=2, hidden_size=4, output_size=1)

    losses = []
    epochs = 10000
    lr = 0.1

    for i in range(epochs):
        output = nn.forward(X)
        loss = nn.backward(X, y, output, lr)
        losses.append(loss)

        if i % 1000 == 0:
            print(f'Epoch {i}, Loss: {loss:.4f}')

    return nn, losses, X


def test(nn, X):
    print('\nPredicciones finales:')
    for x in X:
        pred = nn.forward(np.array([x]))
        print(f'{x} -> {pred[0][0]:.4f}')


if __name__ == '__main__':
    nn, losses, X = train()
    test(nn, X)

    plt.plot(losses)
    plt.title('Evolución del error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
