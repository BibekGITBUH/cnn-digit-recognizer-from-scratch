import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from model import CNN
import os

def one_hot(y):

    result = np.zeros((len(y), 10))

    result[np.arange(len(y)), y] = 1

    return result


def cross_entropy(
    y_true,
    y_pred
):

    return -np.sum(
        y_true *
        np.log(y_pred + 1e-10)
    )


def accuracy(
    y_true,
    y_pred
):

    return (
        np.argmax(y_true)
        ==
        np.argmax(y_pred)
    )


def save_model(model):

    data = {
        "conv1_filters":
            model.conv1.filters,

        "fc1_weights":
            model.fc1.weights,

        "fc1_bias":
            model.fc1.bias,

        "fc2_weights":
            model.fc2.weights,

        "fc2_bias":
            model.fc2.bias
    }

    with open(
        "cnn_weights.pkl",
        "wb"
    ) as f:
        pickle.dump(data, f)


def train(use_user_data=True):

    print(
        "Loading MNIST..."
    )

    (
        X_train,
        y_train
    ), (
        X_test,
        y_test
    ) = mnist.load_data()

    X_train = (
        X_train.astype(
            np.float32
        )
        / 255.0
    )

    X_test = (
        X_test.astype(
            np.float32
        )
        / 255.0
    )

    

    if (
        use_user_data and
        os.path.exists("userSideNewData.npy")
    ):
        data = np.load(
            "userSideNewData.npy",
            allow_pickle=True
        ).item()

        user_X = np.array(
            data["images"],
            dtype=np.float32
        )

        user_y = np.array(
            data["labels"],
            dtype=np.int64
        )

        if len(user_X) > 0:

            print(
                f"Adding {len(user_X)} "
                f"user samples."
            )

            X_train = np.concatenate(
                [X_train, user_X],
                axis=0
            )
            y_train = np.concatenate(
                [y_train, user_y],
                axis=0
)

        
    y_train = one_hot(
        y_train
    )

    y_test = one_hot(
        y_test
    )

    model = CNN()

    epochs = 3
    lr = 0.005

    for epoch in range(
        epochs
    ):

        print(
            f"\nEpoch {epoch+1}"
        )

        loss_sum = 0
        correct = 0

        for i in range(
            len(X_train)
        ):

            x = X_train[i]
            y = y_train[i]

            pred = model.forward(
                x
            )

            loss = cross_entropy(
                y,
                pred
            )

            loss_sum += loss

            if accuracy(
                y,
                pred
            ):
                correct += 1

            model.backward(
                y,
                lr
            )


            print(
                f"sample={i}"
                f" "
                f"loss={loss:.4f}"
                f" "
                f"acc={correct/(i+1):.4f}"
            )

        print(
            "\nTrain Loss:",
            loss_sum
            /
            len(X_train)
        )

        print(
            "Train Accuracy:",
            correct
            /
            len(X_train)
        )

    print(
        "\nSaving model..."
    )

    save_model(model)

    print(
        "Done."
    )


if __name__ == "__main__":
    train()