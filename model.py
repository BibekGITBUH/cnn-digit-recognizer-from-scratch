from layers import *


class CNN:

    def __init__(self):

        self.conv1 = Conv2D(
            num_filters=8,
            kernel_size=3
        )

        self.relu1 = ReLU()

        self.pool1 = MaxPool()

        self.flatten = Flatten()

        self.fc1 = Dense(
            input_size=13 * 13 * 8,
            output_size=64
        )

        self.relu2 = ReLU()

        self.fc2 = Dense(
            input_size=64,
            output_size=10
        )

        self.softmax = Softmax()

    def forward(self, x):

        x = self.conv1.forward(x)

        x = self.relu1.forward(x)

        x = self.pool1.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)

        x = self.relu2.forward(x)

        x = self.fc2.forward(x)

        x = self.softmax.forward(x)

        return x

    def backward(
        self,
        y_true,
        lr
    ):

        grad = self.softmax.backward(
            y_true
        )

        grad = self.fc2.backward(
            grad,
            lr
        )

        grad = self.relu2.backward(
            grad
        )

        grad = self.fc1.backward(
            grad,
            lr
        )

        grad = self.flatten.backward(
            grad
        )

        grad = self.pool1.backward(
            grad
        )

        grad = self.relu1.backward(
            grad
        )

        self.conv1.backward(
            grad,
            lr
        )