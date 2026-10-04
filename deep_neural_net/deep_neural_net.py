import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Bot: Constructing Deep Learning Neural Network Architecture...")

# 1. Stacking the Layers: Building a neural network brain layer-by-layer
model = Sequential([
    # Input layer + Hidden layer: 8 neurons listening to 4 input data features
    Dense(8, input_dim=4, activation='relu'),
    # Hidden layer 2: 4 neurons passing processed signals forward
    Dense(4, activation='relu'),
    # Output layer: 1 neuron outputting a final probability score (0 to 1)
    Dense(1, activation='sigmoid')
])

print("Bot: Stacking complete. Compiling network tensor matrices...")
# 2. Compile: Equip the brain with an optimizer (Adam) and a loss math calculator
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Dummy Data: 10 rows of data with 4 features each (e.g., system signals)
X_data = np.random.rand(10, 4)
Y_labels = np.random.randint(2, size=(10, 1))

print("\nBot: Initializing forward-propagation training cycles (Epochs)...")
# 4. Train: Force the neurons to adjust their internal weights over 5 cycles
model.fit(X_data, Y_labels, epochs=5, verbose=1)

print("\nBot: Success! Deep Learning Neural Network trained and operational.")