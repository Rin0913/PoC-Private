import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, TimeDistributed, Input, LSTM
from tensorflow.keras.models import Sequential
import numpy as np

R = 100
epochs = 50

def generate_horizontal_vertical_lines(num_samples=1000, N = 10):
    X = []
    Y = []

    for _ in range(num_samples):
        rectangles = [(0, 0, R, R)]
        lines = []

        while len(rectangles) < N:
            idx = random.randint(0, len(rectangles) - 1)
            x1, y1, x2, y2 = rectangles.pop(idx)

            if random.choice(["vertical", "horizontal"]) == "vertical" and x2 - x1 > 1:
                split = random.randint(x1 + 1, x2 - 1)
                rectangles.append((x1, y1, split, y2))
                rectangles.append((split, y1, x2, y2))
                lines.append((split, y1, split, y2))
            elif y2 - y1 > 1:
                split = random.randint(y1 + 1, y2 - 1)
                rectangles.append((x1, y1, x2, split))
                rectangles.append((x1, split, x2, y2))
                lines.append((x1, split, x2, split))
            else:
                rectangles.append((x1, y1, x2, y2))

        lines.append((0, 0, R, 0))
        lines.append((0, 0, 0, R))
        lines.append((R, 0, R, R))
        lines.append((0, R, R, R))

        original_sample = lines
        noisy_sample = []

        for line in original_sample:
            x1, y1, x2, y2 = line

            x1_noisy = x1 + random.choice([-1, 0, 1]) if x1 != x2 else x1
            y1_noisy = y1 + random.choice([-1, 0, 1]) if y1 != y2 else y1
            x2_noisy = x2 + random.choice([-1, 0, 1]) if x1 != x2 else x2
            y2_noisy = y2 + random.choice([-1, 0, 1]) if y1 != y2 else y2

            noisy_sample.append((x1_noisy, y1_noisy, x2_noisy, y2_noisy))

        X.append(noisy_sample)
        Y.append(original_sample)

    return np.array(X), np.array(Y)

def my_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    x1, y1, x2, y2 = tf.split(y_pred, 4, axis=-1)

    is_horizontal = tf.abs(y1 - y2) < 1 / R / 2
    is_vertical = tf.abs(x1 - x2) < 1 / R / 2
    is_valid_line = tf.logical_or(is_horizontal, is_vertical)

    penalty = tf.where(is_valid_line, tf.zeros_like(mse_loss), tf.ones_like(mse_loss) * 100.0)
    penalty_loss = tf.reduce_mean(penalty)

    return mse_loss + penalty_loss

def draw_ascii(lines):
    grid_size = 2 * R
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid[y + R // 2][x1 + R // 2] = "|"
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid[y1 + R // 2][x + R // 2] = "-"

    for row in reversed(grid):
        print("".join(row))

model = Sequential([
	LSTM(128, return_sequences=True, input_shape=(None, 4)),
	LSTM(128, return_sequences=True),
	Dense(4)
])

model.compile(optimizer='adam', loss=my_loss)

weights_file = 'm.weights.h5'
if os.path.exists(weights_file):
    print(f"Loading weights from {weights_file}")
    model.load_weights(weights_file)

if __name__ == '__main__':
    with tf.device('/GPU:0'):
        while True:
            X_train, y_train = generate_horizontal_vertical_lines(num_samples=50000, N=random.randint(5, 20))
            X_test, y_test = generate_horizontal_vertical_lines(num_samples=2000, N=random.randint(5, 20))

            X_train = X_train / R
            y_train = y_train / R 
            X_test = X_test / R
            y_test = y_test / R

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=5000, validation_data=(X_test, y_test))

            print("Model training complete.")

            for i in range(10):
                print('ANS:')
                draw_ascii(np.round(y_train[i] * R))
                print('RAW:')
                draw_ascii(np.round(X_train[i] * R))
                print('OUTPUT:')
                prediction = model.predict(X_train[i][np.newaxis, ...])[0]
                draw_ascii(np.round(prediction * R))
                print("-" * 10)


            print(f"Saving weights to {weights_file}")
            model.save_weights(weights_file)

