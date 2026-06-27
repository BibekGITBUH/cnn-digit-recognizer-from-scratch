# import tkinter as tk
# import numpy as np
# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageOps

# from predict import predict_digit


# WIDTH = 280
# HEIGHT = 280

# root = tk.Tk()
# root.title(
#     "Digit Recognizer"
# )

# canvas = tk.Canvas(
#     root,
#     width=WIDTH,
#     height=HEIGHT,
#     bg="black"
# )

# canvas.pack()

# image = Image.new(
#     "L",
#     (WIDTH, HEIGHT),
#     0
# )

# draw = ImageDraw.Draw(
#     image
# )


# def paint(event):

#     x = event.x
#     y = event.y

#     r = 3

#     canvas.create_oval(
#         x-r,
#         y-r,
#         x+r,
#         y+r,
#         fill="white",
#         outline="white"
#     )

#     draw.ellipse(
#         (
#             x-r,
#             y-r,
#             x+r,
#             y+r
#         ),
#         fill=255
#     )


# canvas.bind(
#     "<B1-Motion>",
#     paint
# )


# def clear():

#     canvas.delete("all")

#     draw.rectangle(
#         (
#             0,
#             0,
#             WIDTH,
#             HEIGHT
#         ),
#         fill=0
#     )

#     label.config(
#         text=""
#     )


# def predict():

#     # # img = image.resize(
#     # #     (28, 28)
#     # # )
#     bbox = image.getbbox()

#     # if bbox is None:
#     #     return

#     # img = image.crop(bbox)
#     # img = img.resize((20, 20))
    
#     # new_img = Image.new("L", (28, 28), 0)
#     # new_img.paste(img, (4, 4))

#     # img = new_img
    
    
    
    
    
    

#     # img = ImageOps.invert(
#     #     img
#     # )
#     # img.save("debug.png")

#     # arr = np.array(
#     #     img
#     # ).astype(
#     #     np.float32
#     # )

#     # arr = (arr > 100).astype(np.float32)
    
    
    
    
    
    
#     # print(arr.min(), arr.max())
#     # print(arr.shape)
#     # print("sum =", arr.sum())
#     # print("unique =", np.unique(arr))
#     # digit, conf = (
#     #     predict_digit(arr)
#     # )
    
#     img = image.crop(bbox)
#     img = img.resize((20, 20), Image.Resampling.NEAREST)

#     new_img = Image.new("L", (28, 28), 0)
#     new_img.paste(img, (4, 4))

#     img = new_img

#     arr = np.array(img).astype(np.float32) / 255.0

#     print("sum =", arr.sum())

#     digit, conf = predict_digit(arr)
        

#     label.config(
#         text=
#         f"Prediction: {digit}\n"
#         f"Confidence: {conf:.3f}"
#     )


# btn_predict = tk.Button(
#     root,
#     text="Predict",
#     command=predict
# )

# btn_predict.pack()

# btn_clear = tk.Button(
#     root,
#     text="Clear",
#     command=clear
# )

# btn_clear.pack()

# label = tk.Label(
#     root,
#     text="",
#     font=(
#         "Arial",
#         16
#     )
# )

# label.pack()

# root.mainloop()





import tkinter as tk
import numpy as np
from tkinter import simpledialog
import os
from tkinter import messagebox
import threading
import train
from predict import predict_digit

GRID_SIZE = 28
PIXEL_SIZE = 10

WIDTH = GRID_SIZE * PIXEL_SIZE
HEIGHT = GRID_SIZE * PIXEL_SIZE

root = tk.Tk()
root.title("MNIST Digit Recognizer")

canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="black"
)
canvas.pack()

# This is exactly what the model receives
mnist_img = np.zeros(
    (28, 28),
    dtype=np.float32
)


def paint(event):
    global mnist_img

    x = event.x // PIXEL_SIZE
    y = event.y // PIXEL_SIZE

    # small brush (3x3)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < GRID_SIZE and
                0 <= ny < GRID_SIZE
            ):
                mnist_img[ny, nx] = 1.0

                canvas.create_rectangle(
                    nx * PIXEL_SIZE,
                    ny * PIXEL_SIZE,
                    (nx + 1) * PIXEL_SIZE,
                    (ny + 1) * PIXEL_SIZE,
                    fill="white",
                    outline="white"
                )


canvas.bind(
    "<B1-Motion>",
    paint
)


def clear():
    global mnist_img

    canvas.delete("all")

    mnist_img = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    label.config(text="")


def wrong():
    global mnist_img

    actual = simpledialog.askstring(
        "Wrong Prediction",
        "Enter the correct digit (0-9):"
    )

    if actual is None:
        return

    if not actual.isdigit():
        return

    actual = int(actual)

    if actual < 0 or actual > 9:
        return

    # Load old data if file exists
    if os.path.exists("userSideNewData.npy"):
        data = np.load(
            "userSideNewData.npy",
            allow_pickle=True
        ).item()

        images = data["images"]
        labels = data["labels"]
    else:
        images = []
        labels = []

    # Save current image and label
    images.append(mnist_img.copy())
    labels.append(actual)

    np.save(
        "userSideNewData.npy",
        {
            "images": images,
            "labels": labels
        }
    )

    label.config(
        text=(
            f"Saved digit {actual}\n"
            f"Total custom samples: {len(labels)}"
        )
    )
    update_train_button()


def train_new_data():

    ok = messagebox.askokcancel(
        "Retrain Model",
        "You have enough new handwriting samples.\n\n"
        "Training may take 1-2 hours.\n\n"
        "Do you want to start training?"
    )

    if not ok:
        return

    start_training()

def update_train_button():
    if not os.path.exists("userSideNewData.npy"):
        return

    data = np.load(
        "userSideNewData.npy",
        allow_pickle=True
    ).item()

    n = len(data["labels"])

    if n >= 15:
        if not btn_train.winfo_ismapped():
            btn_train.pack()
    else:
        if btn_train.winfo_ismapped():
            btn_train.pack_forget()

def start_training():
    threading.Thread(
        target=train.train,
        args=(True,),
        daemon=True
    ).start()
    
def predict():
    global mnist_img

    print("sum =", mnist_img.sum())

    digit, conf = predict_digit(
        mnist_img
    )

    label.config(
        text=
        f"Prediction: {digit}\n"
        f"Confidence: {conf:.3f}"
    )


btn_predict = tk.Button(
    root,
    text="Predict",
    command=predict
)
btn_predict.pack()

btn_clear = tk.Button(
    root,
    text="Clear",
    command=clear
)
btn_clear.pack()

btn_wrong = tk.Button(
    root,
    text="Wrong Prediction",
    command=wrong
)
btn_wrong.pack()

btn_train = tk.Button(
    root,
    text="Train New Data",
    bg="red",
    fg="white",
    command=lambda: train_new_data()
)
# hidden initially
btn_train.pack_forget()

label = tk.Label(
    root,
    text="",
    font=("Arial", 16)
)
label.pack()

root.mainloop()