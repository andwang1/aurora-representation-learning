import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from exp_config import *
import os


def plot_dist_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers
    dist_generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            dist_generations.append(fname.rstrip(r".dat")[len("distances"):])

    dist_generations = sorted(int(gen) for gen in dist_generations)

    # for distance plot across generations
    lower_ball_dist = []
    upper_ball_dist = []
    undist_lower_ball_dist = []
    undist_upper_ball_dist = []
    gens = []

    for GEN_NUMBER in dist_generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'distances{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            lower_dist = [float(i) for i in lines[1].strip().split(",")[:-1]]
            upper_dist = [float(i) for i in lines[2].strip().split(",")[:-1]]
            undist_lower_dist = [float(i) for i in lines[3].strip().split(",")[:-1]]
            undist_upper_dist = [float(i) for i in lines[4].strip().split(",")[:-1]]

        gens.append([int(GEN_NUMBER)] * len(lower_dist))
        lower_ball_dist.append(lower_dist)
        upper_ball_dist.append(upper_dist)
        undist_lower_ball_dist.append(undist_lower_dist)
        undist_upper_ball_dist.append(undist_upper_dist)

    data_dict = {"gen": gens, "LBD": lower_ball_dist, "UBD": upper_ball_dist, "ULBD": undist_lower_ball_dist, "UUBD": undist_upper_ball_dist}

    if generate_images:
        data = {"gen": np.array(gens).flatten(),
                "LBD": np.array(lower_ball_dist).flatten(), "UBD": np.array(upper_ball_dist).flatten(),
                "ULBD": np.array(undist_lower_ball_dist).flatten(), "UUBD": np.array(undist_upper_ball_dist).flatten()}
        data_stats = pd.DataFrame(data)

        f = plt.figure(figsize=(6, 10))
        spec = f.add_gridspec(2, 1)
        ax1 = f.add_subplot(spec[0, 0])

        sns.lineplot(data_dict["gen"], data_dict["LBD"], estimator=np.median, ci=None,
                     ax=ax1, color="blue", label="Object 1")
        line_stats = data_stats[["gen", "LBD"]].groupby("gen").describe()
        quart25 = line_stats[("LBD", '25%')]
        quart75 = line_stats[("LBD", '75%')]
        ax1.fill_between(dist_generations, quart25, quart75, alpha=0.3, color="blue")

        sns.lineplot(data_dict["gen"], data_dict["UBD"], estimator=np.median, ci=None,
                     ax=ax1, color="red", label="Object 2")
        line_stats = data_stats[["gen", "UBD"]].groupby("gen").describe()
        quart25 = line_stats[("UBD", '25%')]
        quart75 = line_stats[("UBD", '75%')]
        ax1.fill_between(dist_generations, quart25, quart75, alpha=0.3, color="red")
        ax1.set_title("With Noise")

        ax2 = f.add_subplot(spec[1, 0])

        sns.lineplot(data_dict["gen"], data_dict["ULBD"], estimator=np.median, ci=None,
                     ax=ax2, color="blue", label="Object 1")
        line_stats = data_stats[["gen", "ULBD"]].groupby("gen").describe()
        quart25 = line_stats[("ULBD", '25%')]
        quart75 = line_stats[("ULBD", '75%')]
        ax2.fill_between(dist_generations, quart25, quart75, alpha=0.3, color="blue")

        sns.lineplot(data_dict["gen"], data_dict["UUBD"], estimator=np.median, ci=None,
                     ax=ax2, color="red", label="Object 2")
        line_stats = data_stats[["gen", "UUBD"]].groupby("gen").describe()
        quart25 = line_stats[("UUBD", '25%')]
        quart75 = line_stats[("UUBD", '75%')]
        ax2.fill_between(dist_generations, quart25, quart75, alpha=0.3, color="red")
        ax2.set_title("Without Noise")
        plt.suptitle("Distance Travelled by Objects")
        ax1.set_ylabel("Distance")
        ax2.set_ylabel("Distance")
        ax2.set_xlabel("Generation")
        plt.savefig("distance.png")
        plt.close()
    return data_dict


if __name__ == "__main__":
    plot_dist_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2/results_arrangementsd_aurora/gen4001_random0.8_fulllossfalse_beta1_extension0_lossfunc2_sigmoidfalse_sampletraintrue_sample_0_tsnefalse/2020-09-18_19_08_14_5639")
