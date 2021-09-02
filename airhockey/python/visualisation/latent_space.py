import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os

# make font bigger
plt.rc('font', size=20)
def plot_latent_space_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers from distances as we need the moved data
    generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("distances"):])
    generations = sorted(int(gen) for gen in generations)

    # for plots across generations
    latent_variance_excl_nomove = []

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'archive_{GEN_NUMBER}.dat'
        with open(FILE_NAME, "r") as f:
            lines = f.readlines()

        x = []
        y = []
        for line in lines:
            data = line.strip().split()
            x.append(float(data[1]))
            y.append(float(data[2]))

        MOVED_INDICES_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(MOVED_INDICES_FILE_NAME, "r") as f:
            lines = f.readlines()
        moved_indices = [int(i) for i in lines[5].strip().split()]

        is_moved = np.array([False] * len(x))
        is_moved[moved_indices] = True

        fig = plt.figure(figsize=(15, 15))
        max_value = np.max(np.abs(np.array([x, y])))
        # print(max_value)
        # max_value = 3.7584500312805176
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])

        x = np.array(x)
        y = np.array(y)

        ax1 = fig.add_subplot()
        ax1.scatter(x[is_moved], y[is_moved], c="green", label="Puck Moved")
        ax1.scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Puck Did Not Move")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        plt.title(f"BD Space in Final Generation - % Solutions Moving the Puck: {round(100 * (len(x[is_moved]) / len(x)), 1)}")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")
        plt.legend()

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_space_{GEN_NUMBER}.png")
        # plt.savefig(f"latent_space_{GEN_NUMBER}.pdf")
        plt.close()

        latent_variance_excl_nomove.append((np.var(x[np.invert(is_moved)]) + np.var(y[np.invert(is_moved)])) / 2)

    if generate_images:
        f = plt.figure(figsize=(5, 5))
        spec = f.add_gridspec(1, 1)
        ax1 = f.add_subplot(spec[0, 0])
        ln1 = ax1.plot(generations, latent_variance_excl_nomove, label="Variance", color="red")
        ax1.set_ylabel("Mean Variance")
        ax1.set_title("Variance of Latent Descriptors Excl. No-Move")

        lns = ln1
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        plt.savefig("latent_var.png")
        plt.close()

    data_dict = {"gen": generations, "LV": latent_variance_excl_nomove}
    return data_dict


if __name__ == "__main__":
    plot_latent_space_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain/results_balltrajectorysd_vae/gen6001_random1_fulllossfalse_beta1_extension0_lossfunc2_samplefalse/2020-07-25_22_03_22_3937758")
        # "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain_extend50/results_balltrajectorysd_vae/gen6001_random0.8_fulllosstrue_beta1_extension0.5_lossfunc2_samplefalse_tsne0/2020-08-21_00_26_22_15459")
