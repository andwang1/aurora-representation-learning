import matplotlib.pyplot as plt
import os


def plot_loss_in_dir_AE(path, generate_images=True, show_train_lines=False, save_path=None):
    os.chdir(path)
    FILE = f'ae_loss.dat'

    total_recon = []
    total_L2 = []
    undisturbed_actual_trajectories_L2 = []
    train_epochs = []

    data_dict = {}

    with open(FILE, "r") as f:
        for line in f.readlines():
            data = line.strip().split(",")
            total_recon.append(float(data[1]))
            total_L2.append(float(data[2]))
            undisturbed_actual_trajectories_L2.append(float(data[3]))
            if "IS_TRAIN" in data[-1]:
                # gen number, epochstrained / total
                train_epochs.append((int(data[0]), data[-2].strip()))

    if generate_images:
        f = plt.figure(figsize=(10, 5))

        spec = f.add_gridspec(1, 1)
        # both kwargs together make the box squared
        ax1 = f.add_subplot(spec[0, 0])

        ax1.set_ylabel("L2")
        ax1.set_ylim([0, max(total_L2)])
        ln1 = ax1.plot(range(len(total_L2)), total_L2, c="red", label="L2 - Overall")
        ax1.annotate(f"{round(total_L2[-1], 2)}", (len(total_L2) - 1, total_L2[-1]), xytext=(len(total_L2) - 1, total_L2[-1] * 1.5))

        ln2 = ax1.plot(range(len(undisturbed_actual_trajectories_L2)), undisturbed_actual_trajectories_L2, c="blue", label="L2 - Undist. Image")
        ax1.annotate(f"{round(undisturbed_actual_trajectories_L2[-1], 2)}", (len(undisturbed_actual_trajectories_L2) - 1, undisturbed_actual_trajectories_L2[-1]), xytext=(len(undisturbed_actual_trajectories_L2) - 1, undisturbed_actual_trajectories_L2[-1] * 0.5))

        # train marker
        if show_train_lines:
            for (train_gen, train_ep) in train_epochs:
                ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")

        # add in legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax1.set_title(f"AE Loss")
        ax1.set_xlabel("Generation")
        plt.savefig(f"ae_loss.png")
        plt.close()

    data_dict["L2"] = total_L2
    data_dict["TR_EPOCHS"] = train_epochs
    data_dict["UL"] = undisturbed_actual_trajectories_L2
    return data_dict


if __name__ == "__main__":
    plot_loss_in_dir_AE(
        "/home/andwang1/airl/imagesd/test_results/new_traj_stat/results_imagesd_aurora/gen6001_random0.2_fulllossfalse_beta1_extension0/2020-06-23_14_21_26_26214")
