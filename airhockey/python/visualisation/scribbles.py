import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd
import torch

sns.set_style("dark")
#
# sigmoid_outputs = torch.from_numpy(np.random.uniform(0, 1, 1000))
# target = 1
# print(max(sigmoid_outputs))
#
# sigmoid_outputs = sigmoid_outputs.sigmoid()
#
# sigmoid_deriv = sigmoid_outputs * (1 - sigmoid_outputs)
# bce_deriv = - 1 / sigmoid_outputs
#
# # print(sigmoid_deriv)
# # print(bce_deriv)
#
# full_grad = sigmoid_deriv * bce_deriv
#
# print(max(sigmoid_outputs))
#
# plt.scatter(sigmoid_outputs, full_grad)

plt.plot((0, 0.5, 1), (-0.5, -0.5, 0), label="Huber", color="blue")
plt.plot((0, 1), (-2, 0), label="L2", color="red")

plt.ylabel("Gradient")
plt.xlabel("Output Layer Activation")
plt.title("Gradient Value - Label = 1")
plt.legend()
os.chdir("/home/andwang1/Pictures/final_report/exp3")
plt.savefig("gradient_magnitude_l2.pdf")
# plt.show()