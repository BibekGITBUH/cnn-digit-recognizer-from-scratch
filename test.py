# # # import numpy as np
# # # import pickle
# # # from tensorflow.keras.datasets import mnist
# # # from model import CNN


# # # def one_hot(y):
# # #     result = np.zeros((len(y), 10))
# # #     result[np.arange(len(y)), y] = 1
# # #     return result


# # # model = CNN()

# # # with open("cnn_weights.pkl", "rb") as f:
# # #     data = pickle.load(f)

# # # model.conv1.filters = data["conv1_filters"]
# # # model.fc1.weights = data["fc1_weights"]
# # # model.fc1.bias = data["fc1_bias"]
# # # model.fc2.weights = data["fc2_weights"]
# # # model.fc2.bias = data["fc2_bias"]


# # # (_, _), (X_test, y_test) = mnist.load_data()

# # # X_test = X_test.astype(np.float32) / 255.0
# # # y_test = one_hot(y_test)

# # # correct = 0

# # # for i in range(1000):
# # #     pred = model.forward(X_test[i])

# # #     if np.argmax(pred) == np.argmax(y_test[i]):
# # #         correct += 1

# # # print("Test Accuracy:", correct / 1000)






import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from model import CNN


def load_model():

    model = CNN()

    with open("cnn_weights.pkl", "rb") as f:
        data = pickle.load(f)

    model.conv1.filters = data["conv1_filters"]
    model.fc1.weights = data["fc1_weights"]
    model.fc1.bias = data["fc1_bias"]
    model.fc2.weights = data["fc2_weights"]
    model.fc2.bias = data["fc2_bias"]

    return model


# Load model
model = load_model()

# Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()

X_test = X_test.astype(np.float32) / 255.0
correct = [0] * 10
total = [0] * 10

for i in range(len(X_test)):
    label = y_test[i]

    pred = model.forward(X_test[i])
    pred = np.argmax(pred)

    total[label] += 1

    if pred == label:
        correct[label] += 1

for i in range(10):
    print(f"{i}: {correct[i]}/{total[i]} = {correct[i]/total[i]:.4f}")

# print("Predicted Digit :", digit)
# print("Actual Digit    :", y_test[0])
# print("Confidence      :", confidence)
# print("Probabilities   :")
# print(pred)



# from tensorflow.keras.datasets import mnist
# from PIL import Image
# import numpy as np

# (_, _), (X_test, y_test) = mnist.load_data()

# Image.fromarray(
#     (X_test[11]).astype(np.uint8)
# ).save("mnist_sample.png")

# print(y_test[11])


# from tensorflow.keras.datasets import mnist
# from PIL import Image
# import numpy as np

# (_, _), (X_test, y_test) = mnist.load_data()

# mnist = X_test[0].astype(np.float32) / 255.0

# img = Image.open("debug.png").convert("L")
# canvas = np.array(img).astype(np.float32) / 255.0

# print("MNIST")
# print("shape:", mnist.shape)
# print("min:", mnist.min())
# print("max:", mnist.max())
# print("sum:", mnist.sum())

# print()

# print("Canvas")
# print("shape:", canvas.shape)
# print("min:", canvas.min())
# print("max:", canvas.max())
# print("sum:", canvas.sum())














# sample=60037 loss=0.0085 acc=0.9787
# sample=60038 loss=0.0191 acc=0.9787
# sample=60039 loss=1.5370 acc=0.9787
# sample=60040 loss=0.0191 acc=0.9787

# Train Loss: 0.0731621919775857
# Train Accuracy: 0.9787145450608751
