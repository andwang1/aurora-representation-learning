import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
import seaborn as sns

# make font bigger
plt.rc('font', size=16)
def plot_pos_var_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
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

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

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

        if generate_images:
            # with noise
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(lower_ball_x, lower_ball_y, c="red", label="Object 1")
            ax1.scatter(upper_ball_x, upper_ball_y, c="blue", label="Object 2")
            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions with Noise")
            plt.savefig(f"position_{GEN_NUMBER}.png")
            plt.close()

            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(undist_lower_ball_x, undist_lower_ball_y, c="red", label="Object 1")
            ax1.scatter(undist_upper_ball_x, undist_upper_ball_y, c="blue", label="Object 2")
            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions")
            plt.savefig(f"undisturbed_position_{GEN_NUMBER}.png")
            plt.close()

            # mark positions that have been affected by noise
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(lower_ball_x[lower_ball_affected_by_noise],
                        lower_ball_y[lower_ball_affected_by_noise], c="red")
            ax1.scatter(lower_ball_x[np.invert(lower_ball_affected_by_noise)],
                        lower_ball_y[np.invert(lower_ball_affected_by_noise)], c="green")

            ax1.scatter(upper_ball_x[upper_ball_affected_by_noise],
                        upper_ball_y[upper_ball_affected_by_noise], c="red", label="Moved by Noise")
            ax1.scatter(upper_ball_x[np.invert(upper_ball_affected_by_noise)],
                        upper_ball_y[np.invert(upper_ball_affected_by_noise)], c="green", label="Not Moved by Noise")

            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions with Noise")
            plt.savefig(f"position_movedby_noisemarkings{GEN_NUMBER}.png")
            plt.close()

            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(undist_lower_ball_x[lower_ball_affected_by_noise],
                        undist_lower_ball_y[lower_ball_affected_by_noise], c="red")
            ax1.scatter(undist_lower_ball_x[np.invert(lower_ball_affected_by_noise)],
                        undist_lower_ball_y[np.invert(lower_ball_affected_by_noise)], c="green")

            ax1.scatter(undist_upper_ball_x[upper_ball_affected_by_noise],
                        undist_upper_ball_y[upper_ball_affected_by_noise], c="red", label="Moved by Noise")
            ax1.scatter(undist_upper_ball_x[np.invert(upper_ball_affected_by_noise)],
                        undist_upper_ball_y[np.invert(upper_ball_affected_by_noise)], c="green", label="Not Moved by Noise")

            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions")
            plt.savefig(f"undisturbed_position_movedby_noisemarkings{GEN_NUMBER}.png")
            plt.close()

            # mark solutions affected by noise
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(undist_lower_ball_x[any_ball_affected_by_noise],
                        undist_lower_ball_y[any_ball_affected_by_noise], c="red")
            ax1.scatter(undist_upper_ball_x[any_ball_affected_by_noise],
                        undist_upper_ball_y[any_ball_affected_by_noise], c="red", label="Affected by Noise")

            ax1.scatter(undist_lower_ball_x[np.invert(any_ball_affected_by_noise)],
                        undist_lower_ball_y[np.invert(any_ball_affected_by_noise)], c="green")
            ax1.scatter(undist_upper_ball_x[np.invert(any_ball_affected_by_noise)],
                        undist_upper_ball_y[np.invert(any_ball_affected_by_noise)], c="green",
                        label="Not Affected by Noise")

            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions")
            plt.savefig(f"undisturbed_position_affectedby_noisemarkings{GEN_NUMBER}.png")
            plt.close()

            # mark solutions affected by noise
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot()
            ax1.scatter(lower_ball_x[any_ball_affected_by_noise],
                        lower_ball_y[any_ball_affected_by_noise], c="red")
            ax1.scatter(upper_ball_x[any_ball_affected_by_noise],
                        upper_ball_y[any_ball_affected_by_noise], c="red", label="Affected by Noise")

            ax1.scatter(lower_ball_x[np.invert(any_ball_affected_by_noise)],
                        lower_ball_y[np.invert(any_ball_affected_by_noise)], c="green")
            ax1.scatter(upper_ball_x[np.invert(any_ball_affected_by_noise)],
                        upper_ball_y[np.invert(any_ball_affected_by_noise)], c="green",
                        label="Not Affected by Noise")

            ax1.legend()
            ax1.set_xlim([0, 5])
            ax1.set_ylim([0, 5])
            ax1.set_xlabel("Room X")
            ax1.set_ylabel("Room Y")
            ax1.set_title("Object End Positions with Noise")
            plt.savefig(f"position_affectedby_noisemarkings{GEN_NUMBER}.png")
            plt.close()

    if generate_images:
        f = plt.figure(figsize=(10, 10))
        spec = f.add_gridspec(2, 1)
        ax1 = f.add_subplot(spec[0, 0])
        ln1 = ax1.plot(generations, lower_var, label="Object 1", color="red")
        ln2 = ax1.plot(generations, upper_var, label="Object 2", color="blue")
        ln3 = ax1.plot(generations, both_var, label="Both Objects", color="green")
        ax1.set_ylabel("Variance")
        ax1.set_title("With Noise")
        ax1.legend(loc="best")

        ax2 = f.add_subplot(spec[1, 0])
        ln4 = ax2.plot(generations, undist_lower_var, label="Object 1", color="red")
        ln5 = ax2.plot(generations, undist_upper_var, label="Object 2", color="blue")
        ln6 = ax2.plot(generations, undist_both_var, label="Both Objects", color="green")
        ax2.set_title("Without Noise")
        ax2.set_ylabel("Variance")
        ax2.legend(loc="best")
        plt.suptitle("Variance in Object Positions")
        plt.xlabel("Generation")
        # make space between subplots
        plt.subplots_adjust(hspace=0.6)

        plt.savefig("pos_var.png")
        plt.close()

    data_dict = {"gen": generations, "LOWVAR": lower_var, "UPPVAR": upper_var, "BOTVAR": both_var,
                 "UNLOWVAR": undist_lower_var, "UNUPPVAR": undist_upper_var, "UNBOTVAR": undist_both_var}
    return data_dict


if __name__ == "__main__":
    plot_pos_var_in_dir(
        "/home/andwang1/temp_vae/results_arrangementsd_aurora/gen6_random0.2_fulllossfalse_beta1_extension0_lossfunc2_sigmoidfalse_sampletraintrue_sample_0_tsnefalse/2020-09-16_22_52_47_3719")
