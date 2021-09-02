import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os

# make font bigger
plt.rc('font', size=14)


def plot_pct_move_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers from distances as we need the moved data
    generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("distances"):])
    generations = sorted(int(gen) for gen in generations)

    # for plots across generations
    lower_pct = []
    higher_pct = []
    either_pct = []
    both_pct = []

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        MOVED_INDICES_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(MOVED_INDICES_FILE_NAME, "r") as f:
            lines = f.readlines()
        lower_ball_moved = np.array([float(i) for i in lines[3].strip().split(",")[:-1]]) > 1e-6
        higher_ball_moved = np.array([float(i) for i in lines[4].strip().split(",")[:-1]]) > 1e-6

        is_moved = np.logical_or(lower_ball_moved, higher_ball_moved)
        both_moved = np.logical_and(lower_ball_moved, higher_ball_moved)
        lower_pct.append((1 - lower_ball_moved.sum() / len(lower_ball_moved)) * 100)
        higher_pct.append((1 - higher_ball_moved.sum() / len(higher_ball_moved)) * 100)
        either_pct.append((1 - is_moved.sum() / len(is_moved)) * 100)
        both_pct.append((1 - both_moved.sum() / len(both_moved)) * 100)

    if generate_images:
        f = plt.figure(figsize=(5, 5))
        spec = f.add_gridspec(1, 1)
        ax1 = f.add_subplot(spec[0, 0])
        ln1 = ax1.plot(generations, lower_pct, label="% Moving Lower Object", color="red")
        ln2 = ax1.plot(generations, higher_pct, label="% Moving Upper Object", color="blue")
        ln3 = ax1.plot(generations, either_pct, label="% Moving Either Object", color="green")
        ln4 = ax1.plot(generations, both_pct, label="% Moving Both Objects", color="orange")
        ax1.set_ylabel("%")
        ax1.set_xlabel("Generation")
        ax1.set_title("% Solutions Moving Objects")

        ax1.legend(loc='best')

        plt.savefig("pct_move.png")
        plt.close()

    data_dict = {"gen": generations, "PLOW": lower_pct, "PUPP": higher_pct, "PEIT": either_pct, "PBOT": both_pct}
    return data_dict


if __name__ == "__main__":
    plot_pct_move_in_dir(
        "/home/andwang1/temp_vae/results_arrangementsd_aurora/gen6_random0.2_fulllossfalse_beta1_extension0_lossfunc2_sigmoidfalse_sampletraintrue_sample_0_tsnefalse/2020-09-16_22_52_47_3719")
