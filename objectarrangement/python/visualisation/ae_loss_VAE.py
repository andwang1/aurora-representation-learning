import matplotlib.pyplot as plt
import os



def plot_loss_in_dir_VAE(path, full_loss=True, generate_images=True, show_train_lines=False, save_path=None):
    os.chdir(path)
    FILE = f'ae_loss.dat'

    total_recon = []
    L2 = []
    undisturbed_actual_trajectories_L2 = []
    KL = []
    encoder_var = []
    decoder_var = []
    train_epochs = []
    tsne = []

    exp_spec = path.split("/")[-2]
    tsne_option = exp_spec.split("_")[-1][-1]


    data_dict = {}

    with open(FILE, "r") as f:
        for line in f.readlines():
            data = line.strip().split(",")
            total_recon.append(float(data[1]))
            KL.append(float(data[2]))
            encoder_var.append(float(data[3]))
            decoder_var.append(float(data[4]))
            L2.append(float(data[5]))
            undisturbed_actual_trajectories_L2.append(float(data[6]))
            if "IS_TRAIN" in data[-1]:
                if tsne_option and "tsne" in exp_spec:
                    tsne.append(float(data[7]))
                # gen number, epochstrained / total
                train_epochs.append((int(data[0]), data[-2].strip()))

    if generate_images:
        f = plt.figure(figsize=(15, 10))

        spec = f.add_gridspec(3, 2)
        ax1 = f.add_subplot(spec[0, :])
        ax2 = f.add_subplot(spec[1, :])
        ax3 = f.add_subplot(spec[2, :])

        # L2 and variance on one plot
        ax1.set_ylabel("L2")
        ax1.set_ylim([0, max(L2)])
        ln1 = ax1.plot(range(len(L2)), L2, c="red", label="L2 - Overall")
        ax1.annotate(f"{round(L2[-1], 2)}",
                     (len(L2) - 1, L2[-1]),
                     xytext=(len(L2) - 1, L2[-1] * 1.5))

        ln2 = ax1.plot(range(len(undisturbed_actual_trajectories_L2)), undisturbed_actual_trajectories_L2, c="brown",
                       label="L2 - Undist. Actual Image")
        ax1.annotate(f"{round(undisturbed_actual_trajectories_L2[-1], 2)}",
                     (len(undisturbed_actual_trajectories_L2) - 1, undisturbed_actual_trajectories_L2[-1]),
                     xytext=(len(undisturbed_actual_trajectories_L2) - 1, undisturbed_actual_trajectories_L2[-1] * 0.5))

        if full_loss:
            var_ax = ax1.twinx()
            var_ax.set_ylabel("Variance")
            var_ax.set_ylim([0, max(decoder_var)])
            ln3 = var_ax.plot(range(len(L2)), decoder_var, c="green", label="Variance")
            var_ax.annotate(f"{round(decoder_var[-1], 2)}", (len(decoder_var) - 1, decoder_var[-1]))


        # train marker
        if show_train_lines:
            for (train_gen, train_ep) in train_epochs:
                ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")
                ax2.axvline(train_gen, ls="--", lw=0.1, c="grey")

        # add in legends
        lns = ln1+ln2+ln3 if full_loss else ln1+ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        # aggregate loss and KL on one plot
        KL_ax = ax2.twinx()
        KL_ax.set_ylim([0, max(KL)])
        ax2.set_ylabel("Total Loss")
        KL_ax.set_ylabel("KL")
        ax2.set_ylim([min(total_recon), max(total_recon)])
        ln4 = ax2.plot(range(len(total_recon)), total_recon, c="red", label="Total Recon Loss")
        ln5 = KL_ax.plot(range(len(total_recon)), KL, c="blue", label="KL")

        # add in legends
        lns = ln4+ln5
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='best')

        ax3.plot(range(len(total_recon)), encoder_var, c="red", label="Encoder Variance")
        ax3.legend(loc='best')
        ax3.annotate(f"{round(encoder_var[-1], 2)}", (len(encoder_var) - 1, encoder_var[-1]))
        ax3.set_ylabel("Variance")


        ax1.set_title(f"VAE Loss")
        ax3.set_xlabel("Generation")
        plt.savefig(f"vae_loss.png")
        plt.close()
        # plt.show()

    data_dict["TL"] = total_recon
    data_dict["L2"] = L2
    data_dict["UL"] = undisturbed_actual_trajectories_L2
    data_dict["KL"] = KL
    data_dict["VAR"] = decoder_var
    data_dict["ENVAR"] = encoder_var
    data_dict["TR_EPOCHS"] = train_epochs
    data_dict["TSNE"] = tsne
    return data_dict



if __name__ == "__main__":
    plot_loss_in_dir_VAE(
        "/home/andwang1/airl/balltrajectorysd/results_box2d_exp1/box2dtest/bigger_network/results_balltrajectorysd_vae/--number-gen=6001_--pct-random=0.2_--full-loss=true_--beta=1_--pct-extension=0/2020-06-14_17_51_34_5577")
