import numpy as np


class Conv2D:
    def __init__(self, num_filters, kernel_size):

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.filters = (
            np.random.randn(
                num_filters,
                kernel_size,
                kernel_size
            )
            / (kernel_size * kernel_size)
        )

    def forward(self, input):

        self.input = input

        h, w = input.shape

        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1

        output = np.zeros(
            (out_h, out_w, self.num_filters)
        )

        for f in range(self.num_filters):

            kernel = self.filters[f]

            for i in range(out_h):
                for j in range(out_w):

                    region = input[
                        i:i+self.kernel_size,
                        j:j+self.kernel_size
                    ]

                    output[i, j, f] = np.sum(
                        region * kernel
                    )

        return output

    def backward(self, d_out, lr):

        d_filters = np.zeros_like(self.filters)

        out_h, out_w, _ = d_out.shape

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):

                    region = self.input[
                        i:i+self.kernel_size,
                        j:j+self.kernel_size
                    ]

                    d_filters[f] += (
                        d_out[i, j, f] * region
                    )

        self.filters -= lr * d_filters


class ReLU:

    def forward(self, x):

        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out):

        d = d_out.copy()
        d[self.input <= 0] = 0
        return d


class MaxPool:

    def forward(self, input):

        self.input = input

        h, w, c = input.shape

        output = np.zeros(
            (h // 2, w // 2, c)
        )

        for k in range(c):
            for i in range(0, h, 2):
                for j in range(0, w, 2):

                    region = input[
                        i:i+2,
                        j:j+2,
                        k
                    ]

                    output[
                        i//2,
                        j//2,
                        k
                    ] = np.max(region)

        return output

    def backward(self, d_out):

        h, w, c = self.input.shape

        d_input = np.zeros_like(
            self.input
        )

        for k in range(c):
            for i in range(0, h, 2):
                for j in range(0, w, 2):

                    region = self.input[
                        i:i+2,
                        j:j+2,
                        k
                    ]

                    max_val = np.max(region)

                    for ii in range(2):
                        for jj in range(2):

                            if (
                                region[ii, jj]
                                == max_val
                            ):
                                d_input[
                                    i+ii,
                                    j+jj,
                                    k
                                ] = d_out[
                                    i//2,
                                    j//2,
                                    k
                                ]

        return d_input


class Flatten:

    def forward(self, x):

        self.input_shape = x.shape
        return x.flatten()

    def backward(self, d_out):

        return d_out.reshape(
            self.input_shape
        )


class Dense:

    def __init__(
        self,
        input_size,
        output_size
    ):

        self.weights = (
            np.random.randn(
                input_size,
                output_size
            )
            * 0.01
        )

        self.bias = np.zeros(
            output_size
        )

    def forward(self, x):

        self.input = x
        return (
            np.dot(x, self.weights)
            + self.bias
        )

    def backward(
        self,
        d_out,
        lr
    ):

        d_weights = np.outer(
            self.input,
            d_out
        )

        d_bias = d_out

        d_input = np.dot(
            self.weights,
            d_out
        )

        self.weights -= (
            lr * d_weights
        )

        self.bias -= (
            lr * d_bias
        )

        return d_input


class Softmax:

    def forward(self, x):

        exps = np.exp(
            x - np.max(x)
        )

        self.output = (
            exps / np.sum(exps)
        )

        return self.output

    def backward(
        self,
        y_true
    ):

        return (
            self.output - y_true
        )