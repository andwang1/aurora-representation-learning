import time
import matplotlib.pyplot as plt
import numpy as np


# Scatter plot or lines
SCATTER = True

# Pause after trajectory fully plotted
PAUSE = 1

predictions = []
labels = []

BATCH_SIZE = 64
ROOM_W = 5
ROOM_H = 5

# Read data
with open("../../ae/pred.txt", "r") as f_predictions:
    # If the file is from python AE, need to strip the brackets
    for line in f_predictions.readlines():
        line = line.strip("[").replace("]", "")
        predictions.append([float(i.strip()) for i in line.split(",")])

with open("../../ae/labels.txt", "r") as f_labels:
    # If the file is from python AE, need to strip the brackets
    for line in f_labels.readlines():
        line = line.strip("[").replace("]", "")
        labels.append([float(i.strip()) for i in line.split(",")])

len_trajectory = int(len(predictions[0]) / BATCH_SIZE)
max_error = 1

for prediction, label in zip(predictions, labels):
    for split in range(BATCH_SIZE):
        f = plt.figure(figsize=(10, 5))
        spec = f.add_gridspec(3, 2)
        # both kwargs together make the box squared
        ax1 = f.add_subplot(spec[:2, :], aspect='equal', adjustable='box')
        ax2 = f.add_subplot(spec[2, :])

        ax1.set_title("Trajectories")
        ax2.set_title("Error", loc="left")

        # grab only one trajectory at a time
        start = split * len_trajectory
        end = (split + 1) * len_trajectory

        ax1.set_ylim([ROOM_H, 0])
        ax1.set_xlim([0, ROOM_W])
        x_pred = np.array([i for i in prediction[start:end:2]])
        y_pred = np.array([i for i in prediction[start + 1:end:2]])
        x_label = np.array([i for i in label[start:end:2]])
        y_label = np.array([i for i in label[start + 1:end:2]])

        error = np.sqrt((x_pred - x_label) ** 2 + (y_pred - y_label) ** 2)

        ax2.set_ylim([0, max_error])
        ax2.set_xlim([0, len_trajectory / 2])
        ax2.set_xlabel("Trajectory Step")
        ax2.yaxis.grid(True)

        if SCATTER:
            for i, j in zip(x_label, y_label):
                ax1.scatter(i, j, c="black")
                plt.pause(0.001)

            for index, (i, j, e) in enumerate(zip(x_pred, y_pred, error)):
                ax1.scatter(i, j, c="red")
                ax2.scatter(index, e, s=4, c="black")
                plt.pause(0.001)

        # this needs impact points to look good
        else:
            plt.plot(x_pred, y_pred, c="red")
            plt.pause(1)
            plt.plot(x_label, y_label, c="black")
            plt.pause(1)

        time.sleep(PAUSE)
        plt.close()