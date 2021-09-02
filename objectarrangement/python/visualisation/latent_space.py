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
        lower_ball_moved = np.array([float(i) for i in lines[3].strip().split(",")[:-1]]) > 1e-6
        higher_ball_moved = np.array([float(i) for i in lines[4].strip().split(",")[:-1]]) > 1e-6
        is_moved = np.logical_or(lower_ball_moved, higher_ball_moved)

        fig = plt.figure(figsize=(15, 15))
        max_value = np.max(np.abs(np.array([x, y])))
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])

        x = np.array(x)
        y = np.array(y)

        ax1 = fig.add_subplot()
        ax1.scatter(x[is_moved], y[is_moved], c="green", label="Moved")
        ax1.scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Not Moved")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        plt.title(f"BD Space in Final Generation - % Solutions Moving an Object: {round(100 * (len(x[is_moved]) / len(x)), 1)}")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")
        plt.legend()

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_space_{GEN_NUMBER}.png")
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
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta1_extension0_l2true/2020-06-23_08_42_10_25136")
