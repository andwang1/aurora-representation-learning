import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from exp_config import *

GEN_NUMBER = 6000

FULL_PATH = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_imagesd_exp1/l1nosampletrain/results_imagesd_vae/gen6001_random0_fulllossfalse_beta1_extension0_lossfunc1_sigmoidfalse_samplefalse_tsne0"
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

for indiv in plotting_data[6:]:
    f = plt.figure(figsize=(10, 10))
    spec = f.add_gridspec(1, 1)
    # both kwargs together make the box squared
    ax2 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')

    actual = indiv[1]

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
    ax2.set_title("Observation")
    plt.savefig("synth_img.pdf")
    plt.show()