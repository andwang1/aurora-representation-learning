import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from exp_config import *

GEN_NUMBER = 6000

FULL_PATH = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_imagesd_exp1/l1nosampletrain/results_imagesd_vae/gen6001_random0_fulllossfalse_beta1_extension0_lossfunc1_sigmoidfalse_samplefalse_tsne0"
os.chdir(FULL_PATH)
FILE_NAME = f'images_{GEN_NUMBER}.dat'

FILE = FULL_PATH + "/" + FILE_NAME

with open(FILE, 'r') as f:
    lines = f.readlines()[1:]

constructed_images = []
for image in lines[::6]:
    data = image.split(",")[2:]
    constructed_images.append([float(i) for i in data])

# constructed_images = [image.split(",")[2:] for image in lines[::6]]

constructed_images_array = np.array(constructed_images)


average_image_archive = np.mean(constructed_images, axis=0)

thresholded_image = average_image_archive > 0.5

sns.heatmap(average_image_archive.reshape(DISCRETISATION, DISCRETISATION))
plt.show()

# plt.imshow(thresholded_image.reshape(DISCRETISATION, DISCRETISATION))
# plt.show()
#
# f = plt.figure(figsize=(10, 10))
# spec = f.add_gridspec(1, 1)
# # both kwargs together make the box squared
# ax2 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')
#
# x = []
# y = []
# x_random = []
# y_random = []
# counter_x = 0
# counter_y = 0
# for entry in thresholded_image:
#     if counter_x >= DISCRETISATION:
#         counter_x = 0
#         counter_y += 1
#         if counter_y >= DISCRETISATION:
#             counter_y = 0
#
#     if entry == 1:
#         x.append(counter_x * (ROOM_W / DISCRETISATION))
#         y.append(counter_y * (ROOM_H / DISCRETISATION))
#     elif entry == -1:
#         x_random.append(counter_x * (ROOM_W / DISCRETISATION))
#         y_random.append(counter_y * (ROOM_H / DISCRETISATION))
#     counter_x += 1
#
# ax2.set_ylim([0, ROOM_H])
# ax2.set_xlim([0, ROOM_W])
# ax2.scatter(x, y, c="green")
# ax2.set_xlabel("")
# ax2.set_title("Average Construction in Archive")
# plt.savefig("average_construction.pdf")
# # plt.show()