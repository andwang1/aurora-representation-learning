import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
import seaborn as sns

GEN_NUMBER = 4000

# make font bigger
plt.rc('font', size=26)
def plot_pos_var_in_dir(path1, path2):
    os.chdir(path1)
    files = os.listdir()

    # Find generation numbers
    generations = []
    for fname in files:
        if fname.startswith("end_positions_") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("end_positions_"):])

    generations = sorted(int(gen) for gen in generations)

    # for var plot across generations
    lower_var = []
    upper_var = []
    both_var = []
    undist_lower_var = []
    undist_upper_var = []
    undist_both_var = []

    FILE_NAME = f'end_positions_{GEN_NUMBER}.dat'

    with open(FILE_NAME, "r") as f:
        lines = f.readlines()
    lower_ball_x = np.array([float(i) for i in lines[1].strip().split(",")[:-1]])
    lower_ball_y = np.array([float(i) for i in lines[2].strip().split(",")[:-1]])
    upper_ball_x = np.array([float(i) for i in lines[3].strip().split(",")[:-1]])
    upper_ball_y = np.array([float(i) for i in lines[4].strip().split(",")[:-1]])
    undist_lower_ball_x = np.array([float(i) for i in lines[5].strip().split(",")[:-1]])
    undist_lower_ball_y = np.array([float(i) for i in lines[6].strip().split(",")[:-1]])
    undist_upper_ball_x = np.array([float(i) for i in lines[7].strip().split(",")[:-1]])
    undist_upper_ball_y = np.array([float(i) for i in lines[8].strip().split(",")[:-1]])
    ball_idx_affected_by_noise = [int(i) for i in lines[-1].strip().split(",")[:-1]]
    lower_ball_affected_by_noise = np.array(ball_idx_affected_by_noise) == 1
    upper_ball_affected_by_noise = np.array(ball_idx_affected_by_noise) == 0
    any_ball_affected_by_noise = np.logical_or(lower_ball_affected_by_noise, upper_ball_affected_by_noise)

    lower_var.append((np.var(lower_ball_x) + np.var(lower_ball_y)) / 2)
    upper_var.append((np.var(upper_ball_x) + np.var(upper_ball_y)) / 2)
    both_var.append((lower_var[-1] + upper_var[-1]) / 2)
    undist_lower_var.append((np.var(undist_lower_ball_x) + np.var(undist_lower_ball_y)) / 2)
    undist_upper_var.append((np.var(undist_upper_ball_x) + np.var(undist_upper_ball_y)) / 2)
    undist_both_var.append((undist_lower_var[-1] + undist_upper_var[-1]) / 2)

    fig = plt.figure(figsize=(20, 10))
    fig.text(0.5, 0.03, 'Table X', ha='center')
    spec = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax1.scatter(undist_lower_ball_x, undist_lower_ball_y, c="orange")
    ax1.scatter(undist_upper_ball_x, undist_upper_ball_y, c="purple")
    ax1.set_xlim([0, 5])
    ax1.set_ylim([0, 5])
    ax1.set_ylabel("Table Y")
    ax1.set_title("AE")


    os.chdir(path2)
    files = os.listdir()

    # Find generation numbers
    generations = []
    for fname in files:
        if fname.startswith("end_positions_") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("end_positions_"):])

    generations = sorted(int(gen) for gen in generations)

    # for var plot across generations
    lower_var = []
    upper_var = []
    both_var = []
    undist_lower_var = []
    undist_upper_var = []
    undist_both_var = []

    FILE_NAME = f'end_positions_{GEN_NUMBER}.dat'

    with open(FILE_NAME, "r") as f:
        lines = f.readlines()
    lower_ball_x = np.array([float(i) for i in lines[1].strip().split(",")[:-1]])
    lower_ball_y = np.array([float(i) for i in lines[2].strip().split(",")[:-1]])
    upper_ball_x = np.array([float(i) for i in lines[3].strip().split(",")[:-1]])
    upper_ball_y = np.array([float(i) for i in lines[4].strip().split(",")[:-1]])
    undist_lower_ball_x = np.array([float(i) for i in lines[5].strip().split(",")[:-1]])
    undist_lower_ball_y = np.array([float(i) for i in lines[6].strip().split(",")[:-1]])
    undist_upper_ball_x = np.array([float(i) for i in lines[7].strip().split(",")[:-1]])
    undist_upper_ball_y = np.array([float(i) for i in lines[8].strip().split(",")[:-1]])
    ball_idx_affected_by_noise = [int(i) for i in lines[-1].strip().split(",")[:-1]]
    lower_ball_affected_by_noise = np.array(ball_idx_affected_by_noise) == 1
    upper_ball_affected_by_noise = np.array(ball_idx_affected_by_noise) == 0
    any_ball_affected_by_noise = np.logical_or(lower_ball_affected_by_noise, upper_ball_affected_by_noise)

    lower_var.append((np.var(lower_ball_x) + np.var(lower_ball_y)) / 2)
    upper_var.append((np.var(upper_ball_x) + np.var(upper_ball_y)) / 2)
    both_var.append((lower_var[-1] + upper_var[-1]) / 2)
    undist_lower_var.append((np.var(undist_lower_ball_x) + np.var(undist_lower_ball_y)) / 2)
    undist_upper_var.append((np.var(undist_upper_ball_x) + np.var(undist_upper_ball_y)) / 2)
    undist_both_var.append((undist_lower_var[-1] + undist_upper_var[-1]) / 2)

    ax2.scatter(undist_lower_ball_x, undist_lower_ball_y, c="orange", label="Object 1")
    ax2.scatter(undist_upper_ball_x, undist_upper_ball_y, c="purple", label="Object 2")
    ax2.set_xlim([0, 5])
    ax2.set_ylim([0, 5])
    ax2.axes.yaxis.set_visible(False)
    lgd = ax2.legend()
    lgd.legendHandles[0]._sizes = [50]
    lgd.legendHandles[1]._sizes = [40]
    ax2.set_title("RAED")
    plt.subplots_adjust(wspace=0.05)
    plt.suptitle("Position of Objects at End of Simulation")
    # plt.show()
    plt.savefig(f"/home/andwang1/Pictures/ICLR/undisturbed_position_{GEN_NUMBER}.pdf")
    plt.close()




if __name__ == "__main__":
    plot_pos_var_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2/AURORA/results_arrangementsd_aurora/gen4001_random1_fulllossfalse_beta1_extension0_lossfunc2_sigmoidfalse_sampletraintrue_sample_0_tsnefalse/2020-09-19_06_42_50_32683",
    "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2/best/results_arrangementsd_vae/gen4001_random1_fulllossfalse_beta1_extension0_lossfunc2_sigmoidfalse_sampletrainfalse_sample_0_tsnefalse/2020-09-19_07_10_13_733")
