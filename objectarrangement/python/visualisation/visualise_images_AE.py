import matplotlib.pyplot as plt
import numpy as np
import time
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from exp_config import *

GEN_NUMBER = 6000

FULL_PATH = "/home/andwang1/airl/imagesd/test_results/vistest/results_imagesd_vae/gen6001_random0.2_fulllosstrue_beta1_extension0_lossfunc1_sigmoidfalse/2020-06-25_11_51_42_14672"
os.chdir(FULL_PATH)
FILE_NAME = f'images_{GEN_NUMBER}.dat'

FILE = FULL_PATH + "/" + FILE_NAME

with open(FILE, 'r') as f:
    lines = f.readlines()[1:]

# list of lists of recon, trajectories, losses
plotting_data = []

num_individuals = int(lines[-1].strip().split(",")[0])

indiv_counter = 0
line_number = 0
file_length = len(lines)
for i in range(num_individuals + 1):
    indiv_data = []
    while line_number != file_length:
        data = lines[line_number].strip().split(",")
        if int(data[0]) != i:
            data.append(indiv_data)
            break
        indiv_data.append([float(i) for i in data[2:]])
        line_number += 1
    plotting_data.append(indiv_data)

for indiv in plotting_data[100:]:
    f = plt.figure(figsize=(10, 15))
    spec = f.add_gridspec(2, 4)
    # both kwargs together make the box squared
    ax1 = f.add_subplot(spec[0, :2], aspect='equal', adjustable='box')
    prediction = indiv[0]

    x = []
    y = []
    counter_x = 0
    counter_y = 0
    for entry in prediction:
        if counter_x >= DISCRETISATION:
            counter_x = 0
            counter_y += 1
            if counter_y >= DISCRETISATION:
                counter_y = 0

        if entry >= 0.5:
            x.append(counter_x * (ROOM_W / DISCRETISATION))
            y.append(counter_y * (ROOM_H / DISCRETISATION))
        counter_x += 1

    ax1.set_ylim([0, ROOM_H])
    ax1.set_xlim([0, ROOM_W])
    ax1.scatter(x, y)

    actual = indiv[1]
    ax2 = f.add_subplot(spec[0, 2:], aspect='equal', adjustable='box')

    x = []
    y = []
    x_random = []
    y_random = []
    counter_x = 0
    counter_y = 0
    for entry in actual:
        if counter_x >= DISCRETISATION:
            counter_x = 0
            counter_y += 1
            if counter_y >= DISCRETISATION:
                counter_y = 0

        if entry == 1:
            x.append(counter_x * (ROOM_W / DISCRETISATION))
            y.append(counter_y * (ROOM_H / DISCRETISATION))
        elif entry == -1:
            x_random.append(counter_x * (ROOM_W / DISCRETISATION))
            y_random.append(counter_y * (ROOM_H / DISCRETISATION))
        counter_x += 1

    ax2.set_ylim([0, ROOM_H])
    ax2.set_xlim([0, ROOM_W])
    ax2.scatter(x, y, c="green", label="Actual")
    ax2.scatter(x_random, y_random, c="red", label="Random")
    ax2.legend(loc="best")
    ax1.set_title("Recon")
    ax2.set_title("Observations")

    rows = []
    column = []
    counter_x = 0
    for i in indiv[2]:
        column.append(float(i))
        counter_x += 1
        if counter_x >= DISCRETISATION:
            counter_x = 0
            rows.append(column)
            column = []

    L2 = np.array(indiv[2]).reshape(DISCRETISATION, DISCRETISATION)
    ax3 = f.add_subplot(spec[1, 1:3], aspect='equal', adjustable='box')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sns.heatmap(L2, ax=ax3, vmin=0, vmax=1, cbar_ax=cax,
                xticklabels=np.arange(0, ROOM_W, ROOM_W / DISCRETISATION), yticklabels=np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
    ax3.invert_yaxis()
    ax3.set_title("L2")

    plt.subplots_adjust(hspace=0.6)
    plt.show()