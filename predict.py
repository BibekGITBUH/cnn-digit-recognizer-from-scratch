import numpy as np
import pickle
from model import CNN


def load_model():

    model = CNN()

    with open(
        "cnn_weights.pkl",
        "rb"
    ) as f:
        data = pickle.load(f)

    model.conv1.filters = data[
        "conv1_filters"
    ]

    model.fc1.weights = data[
        "fc1_weights"
    ]

    model.fc1.bias = data[
        "fc1_bias"
    ]

    model.fc2.weights = data[
        "fc2_weights"
    ]

    model.fc2.bias = data[
        "fc2_bias"
    ]

    return model


def predict_digit(img):

    model = load_model()

    pred = model.forward(img)

    digit = np.argmax(pred)

    confidence = pred[digit]

    return digit, confidence