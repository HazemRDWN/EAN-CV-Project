import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score

data = np.load("mahdbase.npz")
xTrain = data["xTrain"]
yTrain = data["yTrain"]

xTest = data["xTest"]
yTest = data["yTest"]

xTrain = 255 - xTrain
xTest = 255 - xTest

xTrain = (xTrain / 255).astype(np.float32)
xTest = (xTest / 255).astype(np.float32)

print("________Data Type Check________\n")

print(f"xTrain type: {xTrain.dtype}\n")
print(f"yTrain type: {yTrain.dtype}\n")
print(f"xTest type: {xTest.dtype}\n")
print(f"yTest type: {yTest.dtype}\n")


print("________NULL values Check________\n")

if (np.isnan(xTrain).any() | np.isnan(yTrain).any() | np.isnan(xTest).any() | np.isnan(yTest).any()):
    print("NULL values detected.\n")

else:
    print("Data is clean of NULL values.\n")

print("________Shape Alignment Check________\n")

print(f"xTrain shape: {xTrain.shape}\n")
print(f"yTrain shape: {yTrain.shape}\n")
print(f"xTest shape: {xTest.shape}\n")
print(f"yTest shape: {yTest.shape}\n")

inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]
)

model.fit(xTrain, yTrain, epochs=5, verbose=1)
yPred = np.argmax(model.predict(xTest), axis=-1)

print(f"Accuracy: {(accuracy_score(yTest, yPred)) * 100}%")
print(f"F1 score per class: {(f1_score(yTest, yPred, average=None))}")

model.save('mahdbaseModel.keras')

print ("Model saved.")