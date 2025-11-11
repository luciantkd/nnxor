# 👋 Tell the computer to use TensorFlow as Keras's backend (the brain engine)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# 🧮 Import the tools we need
import numpy as np              # For numbers and arrays
import tensorflow as tf         # For deep learning and computation
from keras import Input, Model  # To build the neural network
from keras.layers import Dense  # "Dense" means a fully connected layer
from keras.optimizers import Adam  # A smart learning helper (optimizer)
import matplotlib.pyplot as plt  # For plotting graphs

# 🎯 Make sure the results are the same every time we run (for reproducibility)
np.random.seed(444)
tf.random.set_seed(444)

# 📊 XOR truth table — our "questions" (X) and "answers" (y)
# X has 4 examples, each with two numbers (0 or 1)
# y has the correct XOR output for each example
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 🧠 Build the neural network (the robot’s brain)
# Step 1: Define what goes into the brain (2 numbers)
inp = Input(shape=(2,))

# Step 2: Hidden layer with 8 "neurons" (mini brains)
# It uses the 'tanh' activation function — helps it learn complex patterns
h   = Dense(4, activation="tanh", kernel_initializer="glorot_uniform")(inp)
h   = Dense(4, activation="tanh", kernel_initializer="glorot_uniform")(h)

# Step 3: Output layer with 1 neuron that guesses 0 or 1
# 'sigmoid' squashes numbers between 0 and 1 (good for binary answers)
out = Dense(1, activation="sigmoid")(h)

# Step 4: Combine input and output to make the full model
model = Model(inp, out)

# ⚙️ Tell the model how to learn
# - Adam is the teacher (optimizer)
# - binary_crossentropy measures how wrong the guesses are
# - accuracy tells us how often it’s right
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 🏋️ Train the brain
# - epochs = 3000 means show it the data 3000 times
# - batch_size = 4 means all 4 examples at once
# - verbose=0 keeps output quiet
# - shuffle=True mixes the data each time to help learning
history = model.fit(X, y, epochs=5000, batch_size=4, verbose=0, shuffle=True)

#🔮 Test what the model learned
# 'predict' makes guesses for all XOR inputs
pred = model.predict(X, verbose=0).ravel()  # flatten results to 1D

# 📢 Print what it predicted
# Raw = the actual numbers (like 0.02 or 0.97)
# Rounded = 0 if less than 0.5, 1 if greater than 0.5
print("Raw:", pred)
print("Rounded:", (pred > 0.5).astype(int))

# 🎨 Visualise how the loss changed while training
plt.plot(history.history['loss'])
plt.title("Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

