import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os

# make font bigger
plt.rc('font', size=20)
def plot_latent_dist_space_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers from distances as we need the moved data
    generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("distances"):])
    generations = sorted(int(gen) for gen in generations)

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

        DIST_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(DIST_FILE_NAME, "r") as f:
            lines = f.readlines()
        distances = [float(i) for i in lines[-1][:-2].strip().split(",")]
        moved_indices = [int(i) for i in lines[5].strip().split()]

        fig = plt.figure(figsize=(15, 15))
        spec = fig.add_gridspec(1,1)
        ax1 = fig.add_subplot(spec[:, :], aspect='equal', adjustable='box')

        max_value = np.max(np.abs(np.array([x, y])))
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])
        x = np.array(x)
        y = np.array(y)

        # ax1 = fig.add_subplot()
        scatterplot = ax1.scatter(x, y, c=distances)
        cbar = plt.colorbar(scatterplot)
        cbar.set_label('Trajectory Length', rotation=270, labelpad=30)

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        plt.title(f"BD Space")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_dist_{GEN_NUMBER}.png")
        plt.close()


if __name__ == "__main__":
    plot_latent_dist_space_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_imagesd_exp1/tsnebeta0_nosampletrain/results_imagesd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_sigmoidfalse_samplefalse_tsne2/2020-08-19_16_26_36_3595927")
# "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2beta0nosample/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_samplefalse/2020-07-21_01_26_48_639513")
