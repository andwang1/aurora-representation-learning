import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pk
import pandas as pd
import seaborn as sns
from collections import defaultdict
from visualisation.diversity import plot_diversity_in_dir
from visualisation.dist_grid import plot_dist_grid_in_dir
from visualisation.pos_var_grid import plot_pos_var_grid_in_dir
from visualisation.entropy_grid import plot_entropy_grid_in_dir
from visualisation.ae_loss_AE import plot_loss_in_dir_AE
from visualisation.ae_loss_VAE import plot_loss_in_dir_VAE
from visualisation.latent_space import plot_latent_space_in_dir
from visualisation.recon_notmoved_var import plot_recon_not_moved_var_in_dir

GENERATE_PID_IMAGES = False
GENERATE_EXP_IMAGES = False
START_GEN_LOSS_PLOT = 500

results_dir = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/15k/isd"
groups = {group_name for group_name in os.listdir(results_dir) if
            os.path.isdir(os.path.join(results_dir, group_name)) and group_name != "plots"}

# exclude_dirs = {"huber"}
# groups -= exclude_dirs

only_dirs = {
"beta0",
}
groups &= only_dirs

# make legend bigger
plt.rc('legend', fontsize=14)
# make lines thicker
plt.rc('lines', linewidth=2, linestyle='-.')
# make font bigger
plt.rc('font', size=12)
sns.set_style("dark")

print(groups)

for group in groups:
    EXP_FOLDER = f"{results_dir}/{group}"
    BASE_NAME = "results_imagesd_"
    variants = [exp_name.split("_")[-1] for exp_name in os.listdir(EXP_FOLDER) if
                os.path.isdir(os.path.join(EXP_FOLDER, exp_name))]

    # store all data
    diversity_stoch_dict = {}
    loss_stoch_dict = {}
    distance_stoch_dict = {}
    entropy_stoch_dict = {}
    pos_var_stoch_dict = {}
    recon_var_stoch_dict = {}
    pct_stoch_dict = {}
    latent_var_stoch_dict = {}

    for variant in variants:
        os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
        exp_names = [exp_name for exp_name in os.listdir() if
                     os.path.isdir(os.path.join(f"{EXP_FOLDER}/{BASE_NAME}{variant}", exp_name))]

        is_full_loss = [False] * len(exp_names)

        if variant == "vae":
            for i, name in enumerate(exp_names):
                if "true" in name:
                    is_full_loss[i] = True

        variant_diversity_dict = defaultdict(list)
        variant_loss_dict = defaultdict(list)
        variant_dist_dict = defaultdict(list)
        variant_pos_var_dict = defaultdict(list)
        variant_entropy_dict = defaultdict(list)
        variant_recon_var_dict = defaultdict(list)
        variant_latent_var_dict = defaultdict(list)

        for i, exp in enumerate(exp_names):
            exp_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}"
            pids = [pid for pid in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, pid))]
            for pid in pids:
                full_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}/{pid}"
                print(f"PROCESSING - {full_path}")
                div_dict, max_diversity = plot_diversity_in_dir(full_path, GENERATE_PID_IMAGES)
                variant_diversity_dict[exp].append(div_dict)
                variant_latent_var_dict[exp].append(plot_latent_space_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_dist_dict[exp].append(plot_dist_grid_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_pos_var_dict[exp].append(plot_pos_var_grid_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_entropy_dict[exp].append(plot_entropy_grid_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_recon_var_dict[exp].append(plot_recon_not_moved_var_in_dir(full_path, GENERATE_PID_IMAGES))
                # PID level plotting
                if variant == "vae":
                    variant_loss_dict[exp].append(plot_loss_in_dir_VAE(full_path, is_full_loss[i], GENERATE_PID_IMAGES))
                else:
                    variant_loss_dict[exp].append(plot_loss_in_dir_AE(full_path, GENERATE_PID_IMAGES))

            if not GENERATE_EXP_IMAGES:
                continue

            # experiment level plotting
            os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}")

            # at experiment level, plot mean and stddev curves for diversity over generations
            generations = list(variant_diversity_dict[exp][0].keys()) * len(variant_diversity_dict[exp])
            y = np.array([list(repetition.values()) for repetition in variant_diversity_dict[exp]])
            y = y.flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(1, 1)
            ax1 = f.add_subplot(spec[0, 0])
            sns.lineplot(generations, y, estimator=np.median, ci=None, label="Diversity", ax=ax1)
            data_stats = pd.DataFrame({"x": generations, "y": y}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3)
            plt.title("Diversity")
            plt.xlabel("Generation")
            plt.ylabel("Diversity")
            plt.hlines(max_diversity, 0, generations[-1], linestyles="--", label="Max Diversity")
            plt.legend()
            plt.savefig("diversity.pdf")
            plt.close()
            # plot recon var at experiment level
            NMV_values = np.array([repetition["NMV"] for repetition in variant_recon_var_dict[exp]])
            generations = variant_recon_var_dict[exp][0]["gen"] * len(NMV_values)

            f = plt.figure(figsize=(5, 5))
            spec = f.add_gridspec(1, 1)
            ax1 = f.add_subplot(spec[0, 0])
            ln1 = sns.lineplot(generations, NMV_values.flatten(), estimator=np.median, ci=None, ax=ax1,
                               color="red")
            data_stats = pd.DataFrame({"x": generations, "y": NMV_values.flatten()}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3)
            ax1.set_ylabel("Variance")
            ax1.set_xlabel("Generation")
            plt.title("Var. of Constructions of No-Move solutions")
            plt.savefig("recon_not_moved_var.pdf")
            plt.close()

            # at experiment level plot distance metrices
            MD_values = np.array([repetition["MD"] for repetition in variant_dist_dict[exp]])
            MDE_values = np.array([repetition["MDE"] for repetition in variant_dist_dict[exp]]).flatten()
            VD_values = np.array([repetition["VD"] for repetition in variant_dist_dict[exp]]).flatten()
            VDE_values = np.array([repetition["VDE"] for repetition in variant_dist_dict[exp]]).flatten()
            PCT_values = np.array([repetition["PCT"] for repetition in variant_dist_dict[exp]]).flatten()
            generations = variant_dist_dict[exp][0]["gen"] * len(MD_values)

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(3, 1)
            ax1 = f.add_subplot(spec[:2, 0])
            ln1 = sns.lineplot(generations, MD_values.flatten(), estimator=np.median, ci=None, label="Mean Distance", ax=ax1,
                               color="red", linestyle="--")
            ln2 = sns.lineplot(generations, MDE_values, estimator=np.median, ci=None, label="Mean Distance excl. 0s", ax=ax1,
                               color="red")
            ax1.set_ylabel("Mean Distance")

            ax2 = ax1.twinx()
            ln3 = sns.lineplot(generations, VD_values, estimator=np.median, ci=None, label="Variance", ax=ax2, color="blue",
                               linestyle="--")
            ln4 = sns.lineplot(generations, VDE_values, estimator=np.median, ci=None, label="Variance excl. 0s", ax=ax2,
                               color="blue")
            ax2.set_ylabel("Variance")
            ax2.set_title("Distance Stats over Generations")

            # first remove default legends automatically added then add combined set
            ax1.get_legend().remove()
            ax2.get_legend().remove()
            lns = ln2.get_lines() + ln4.get_lines()
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='best')

            ax3 = f.add_subplot(spec[2, 0])
            ax3.set_title("% Solutions Moving The Ball")
            ax3.set_ylim([25, 100])
            ax3.set_yticks([25, 50, 75, 100])
            ax3.yaxis.grid(True)
            sns.lineplot(generations, PCT_values, estimator=np.median, ci=None, ax=ax3, color="red")
            ax3.set_ylabel("%")
            ax3.set_xlabel("Generation")

            # make space between subplots
            plt.subplots_adjust(hspace=0.6)

            plt.savefig("distance.pdf")
            plt.close()

            # plot pos_var at experiment level
            PV_values = np.array([repetition["PV"] for repetition in variant_pos_var_dict[exp]]).flatten()
            PVE_values = np.array([repetition["PVE"] for repetition in variant_pos_var_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(3, 1)
            ax1 = f.add_subplot(spec[:2, 0])
            # ln1 = sns.lineplot(generations, PV_values, estimator=np.median, ci=None, label="Mean Variance", ax=ax1,
            #                    color="red")

            ln2 = sns.lineplot(generations, PV_values, estimator=np.median, ci=None, ax=ax1,
                               color="blue")
            data_stats = pd.DataFrame({"x": generations, "y": PV_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3, color="blue")
            ax1.set_ylabel("Variance")
            ax1.set_title("Variance in Trajectories")

            # ax1.get_legend().remove()
            # lns = ln2.get_lines()
            # labs = [l.get_label() for l in lns]
            # ax1.legend(lns, labs, loc='best')

            ax3 = f.add_subplot(spec[2, 0])
            ax3.set_title("% Solutions Moving The Ball")
            ax3.set_ylim([25, 100])
            ax3.set_yticks([25, 50, 75, 100])
            ax3.yaxis.grid(True)
            sns.lineplot(generations, PCT_values, estimator=np.median, ci=None, ax=ax3, color="red")
            data_stats = pd.DataFrame({"x": generations, "y": PCT_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax3.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3, color="red")
            ax3.set_ylabel("%")
            ax3.set_xlabel("Generation")

            # make space between subplots
            plt.subplots_adjust(hspace=1.2)

            plt.savefig("pos_var.pdf")
            plt.close()

            # plot entropy at experiment level
            EV_values = np.array([repetition["EV"] for repetition in variant_entropy_dict[exp]]).flatten()
            EVE_values = np.array([repetition["EVE"] for repetition in variant_entropy_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(3, 1)
            ax1 = f.add_subplot(spec[:2, 0])
            # ln1 = sns.lineplot(generations, EV_values, estimator=np.median, ci=None, label="Mean Entropy", ax=ax1,
            #                    color="red", linestyle="--")
            ln2 = sns.lineplot(generations, EVE_values, estimator=np.median, ci=None, ax=ax1,
                               color="blue")
            data_stats = pd.DataFrame({"x": generations, "y": EVE_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3, color="blue")
            ax1.set_ylabel("Entropy")
            ax1.set_title("Entropy of Trajectory Positions Excl. No-Move")

            # ax1.get_legend().remove()
            # lns = ln2.get_lines()
            # labs = [l.get_label() for l in lns]
            # ax1.legend(lns, labs, loc='best')

            ax3 = f.add_subplot(spec[2, 0])
            ax3.set_title("% Solutions Moving The Ball")
            ax3.set_ylim([25, 100])
            ax3.set_yticks([25, 50, 75, 100])
            ax3.yaxis.grid(True)
            sns.lineplot(generations, PCT_values, estimator=np.median, ci=None, ax=ax3, color="red")
            data_stats = pd.DataFrame({"x": generations, "y": PCT_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax3.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3, color="red")
            ax3.set_ylabel("%")
            ax3.set_xlabel("Generation")

            # make space between subplots
            plt.subplots_adjust(hspace=1.2)

            plt.savefig("entropy.pdf")
            plt.close()

            # plot latent var at experiment level
            LV_values = np.array([repetition["LV"] for repetition in variant_latent_var_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(1, 1)
            ax1 = f.add_subplot(spec[0, 0])
            ln1 = sns.lineplot(generations, LV_values, estimator=np.median, ci=None, ax=ax1,
                               color="red", linestyle="--")
            data_stats = pd.DataFrame({"x": generations, "y": LV_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between(list(range(0, 15001, 500)),
                             quart25, quart75, alpha=0.3, color="red")
            ax1.set_ylabel("Variance")
            ax1.set_xlabel("Generation")
            ax1.set_title("Variance of Latent Descriptors of No-Move Solutions")

            plt.savefig("latent_var.pdf")
            plt.close()

            # at experiment level, plot losses
            L2_values = np.array([repetition["L2"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]])
            UL_values = np.array([repetition["UL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
            if variant == "vae":
                ENVAR_values = np.array([repetition["ENVAR"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
            x = list(range(START_GEN_LOSS_PLOT, len(L2_values[0]) + START_GEN_LOSS_PLOT)) * len(L2_values)

            if "fulllosstrue" in exp:
                f = plt.figure(figsize=(15, 10))
                spec = f.add_gridspec(3, 2)
                ax2 = f.add_subplot(spec[1, :])
                ax3 = f.add_subplot(spec[2, :])
                VAR_values = np.array([repetition["VAR"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
                TL_values = np.array([repetition["TL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()

                var_ax = ax1.twinx()
                ln3 = sns.lineplot(x, VAR_values, estimator="mean", ci="sd", label="Decoder Var.", ax=var_ax, color="green")
                var_ax.set_ylabel("Variance")
                var_ax.get_legend().remove()

                ln4 = sns.lineplot(x, TL_values, estimator="mean", ci="sd", label="Total Loss", ax=ax2, color="red")
                ax2.set_ylabel("Total Loss")
            elif variant == "vae":
                f = plt.figure(figsize=(10, 5))
                spec = f.add_gridspec(2, 2)
                ax3 = f.add_subplot(spec[1, :])
                KL_values = np.array(
                    [repetition["KL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
            else:
                f = plt.figure(figsize=(5, 5))
                spec = f.add_gridspec(1, 2)

            ax1 = f.add_subplot(spec[0, :])

            # plot overall L2 and actual L2
            # ln1 = sns.lineplot(x, L2_values.flatten(), estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
            ln2 = sns.lineplot(x, UL_values, estimator="mean", ci="sd", label="Undisturbed L2", ax=ax1, color="brown")
            ax1.set_ylabel("L2")

            # add in legends, one return value of lineplot will have all lines on the axis
            # ax1.get_legend().remove()
            # lns = ln2.get_lines() + ln3.get_lines() if "fulllosstrue" in exp else ln2.get_lines()
            # labs = [l.get_label() for l in lns]
            # ax1.legend(lns, labs, loc='best')

            if variant == "vae":
                KL_ax = ax3.twinx()
                ln5 = sns.lineplot(x, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
                KL_ax.set_ylabel("KL")
                #
                # ln7 = sns.lineplot(x, ENVAR_values, estimator="mean", ci="sd", label="Encoder Var.", ax=ax3, color="red")
                # labs = [l.get_label() for l in ln7.get_lines()]
                # ax3.legend(ln7.get_lines(), labs, loc='best')
                #
                # ax3.get_legend().remove()
                # KL_ax.get_legend().remove()
                # lns = ln7.get_lines() + ln5.get_lines()
                # labs = [l.get_label() for l in lns]
                # ax3.legend(lns, labs, loc='best')

            ax1.set_title(f"Losses")
            # if variant == "vae":
            #     ax3.set_xlabel("Generation")
            # else:
            ax1.set_xlabel("Generation")

            plt.savefig("losses.pdf")
            plt.close()

        # variant plotting
        # retrieve stochasticity levels from file names
        os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
        generations = list(variant_diversity_dict[exp][0].keys())
        all_experiments = variant_diversity_dict.keys()
        stochasticities = []

        for name in all_experiments:
            components = name.split("_")
            # "random" part of experiment name
            stochasticities.append((components[1][len("random"):]))

        # remove duplicates from fulllosstrue and fulllossfalse and sort
        stochasticities = sorted(list(set(stochasticities)))

        # plot diversity across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for generation in generations:
                diversity_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_diversity_dict["_".join(components)]:
                        diversity_values.append(repetition[generation])
                        stochasticity_values.append(stochasticity)

                sns.lineplot(stochasticity_values, diversity_values, estimator="mean", ci="sd", label="Diversity")
                plt.title(f"Diversity Mean and Std.Dev. - Gen {generation}")
                plt.xlabel("Stochasticity")
                plt.ylabel("Diversity")
                plt.hlines(max_diversity, 0, 1, linestyles="--", label="Max Diversity")
                plt.legend()
                if loss_type == "fulllosstrue":
                    plt.savefig(f"diversity_gen{generation}_fullloss.png")
                else:
                    plt.savefig(f"diversity_gen{generation}_notfullloss.png")
                plt.close()
            # record last generation
            diversity_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "DIV": diversity_values}

        # plot entropy across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                NMV_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_recon_var_dict["_".join(components)]:
                        NMV_values.append(repetition["NMV"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(5, 5))
                spec = f.add_gridspec(1, 1)
                ax1 = f.add_subplot(spec[0, 0])
                ln1 = sns.lineplot(stochasticity_values, NMV_values, estimator="mean", ci="sd", label="Mean Variance",
                                   ax=ax1,
                                   color="red", linestyle="--")
                ax1.set_ylabel("Mean Reconstruction Variance")
                ax1.set_xlabel("Stochasticity")
                ax1.legend()
                plt.title(f"Reconstruction Var. of No-Move solutions - Gen {generation}")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"recon_not_moved_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"recon_not_moved_var{generation}_notfullloss.png")
                plt.close()

            # record last generation
            recon_var_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "NMV": NMV_values}

        # plot distances across stochasticity for each generation
        generations = list(variant_dist_dict[exp][0]["gen"])
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                MD_values = []
                MDE_values = []
                VD_values = []
                VDE_values = []
                PCT_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_dist_dict["_".join(components)]:
                        MD_values.append(repetition["MD"][i])
                        MDE_values.append(repetition["MDE"][i])
                        VD_values.append(repetition["VD"][i])
                        VDE_values.append(repetition["VDE"][i])
                        PCT_values.append(repetition["PCT"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(15, 5))
                spec = f.add_gridspec(5, 1)
                ax1 = f.add_subplot(spec[:2, 0])
                ln1 = sns.lineplot(stochasticity_values, MD_values, estimator="mean", ci="sd", label="Mean Distance",
                                   ax=ax1, color="red")
                ln2 = sns.lineplot(stochasticity_values, MDE_values, estimator="mean", ci="sd",
                                   label="Mean Distance excl. 0s",
                                   ax=ax1, color="blue")
                ax1.set_title(f"Distance Stats over Stochasticity - Gen {generation}")
                ax1.set_ylabel("Mean Distance")
                # first remove default legends automatically added then add combined set
                if ax1.get_legend():
                    ax1.get_legend().remove()
                lns = ln2.get_lines()
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='best')

                ax2 = f.add_subplot(spec[2:4, 0])
                ln3 = sns.lineplot(stochasticity_values, VD_values, estimator="mean", ci="sd", label="Variance", ax=ax2,
                                   color="red")
                ln4 = sns.lineplot(stochasticity_values, VDE_values, estimator="mean", ci="sd",
                                   label="Variance excl. 0s", ax=ax2,
                                   color="blue")
                ax2.set_ylabel("Variance")
                if ax2.get_legend():
                    ax2.get_legend().remove()
                lns = ln4.get_lines()
                labs = [l.get_label() for l in lns]
                ax2.legend(lns, labs, loc='best')

                ax3 = f.add_subplot(spec[4, 0])
                ax3.set_title("% Solutions Moving The Ball")
                ax3.set_ylim([0, 100])
                ax3.set_yticks([0, 25, 50, 75, 100])
                ax3.yaxis.grid(True)
                sns.lineplot(stochasticity_values, PCT_values, estimator="mean", ci="sd", ax=ax3)
                ax3.set_ylabel("%")
                ax3.set_xlabel("Stochasticity")

                # make space between subplots
                plt.subplots_adjust(hspace=1)

                if loss_type == "fulllosstrue":
                    plt.savefig(f"distance_gen{generation}_fullloss.png")
                else:
                    plt.savefig(f"distance_gen{generation}_notfullloss.png")
                plt.close()

            # record last generation
            distance_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "MD": MD_values, "MDE": MDE_values,
                                                            "VD": VD_values, "VDE": VDE_values}
            pct_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "PCT": PCT_values}

        # plot pos var across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                PV_values = []
                PVE_values = []
                PCT_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_pos_var_dict["_".join(components)]:
                        PV_values.append(repetition["PV"][i])
                        PVE_values.append(repetition["PVE"][i])
                        stochasticity_values.append(stochasticity)
                    for repetition in variant_dist_dict["_".join(components)]:
                        PCT_values.append(repetition["PCT"][i])

                f = plt.figure(figsize=(10, 5))
                spec = f.add_gridspec(3, 1)
                ax1 = f.add_subplot(spec[:2, 0])
                ln1 = sns.lineplot(stochasticity_values, PV_values, estimator="mean", ci="sd", label="Mean Variance",
                                   ax=ax1,
                                   color="red")
                ln2 = sns.lineplot(stochasticity_values, PVE_values, estimator="mean", ci="sd",
                                   label="Mean Variance excl. 0s",
                                   ax=ax1,
                                   color="blue")
                ax1.set_ylabel("Mean Variance")
                ax1.set_title(f"Variance of Trajectory Positions - Gen {generation}")

                if ax1.get_legend():
                    ax1.get_legend().remove()

                lns = ln2.get_lines()
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='best')

                ax3 = f.add_subplot(spec[2, 0])
                ax3.set_title("% Solutions Moving The Ball")
                ax3.set_ylim([0, 100])
                ax3.set_yticks([0, 25, 50, 75, 100])
                ax3.yaxis.grid(True)
                sns.lineplot(stochasticity_values, PCT_values, estimator="mean", ci="sd", ax=ax3)
                ax3.set_ylabel("%")
                ax3.set_xlabel("Stochasticity")

                # make space between subplots
                plt.subplots_adjust(hspace=0.6)

                if loss_type == "fulllosstrue":
                    plt.savefig(f"pos_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"pos_var{generation}_notfullloss.png")
                plt.close()

            pos_var_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "PV": PV_values,
                                                            "PVE": PVE_values}

        # plot entropy across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                EV_values = []
                EVE_values = []
                PCT_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_entropy_dict["_".join(components)]:
                        EV_values.append(repetition["EV"][i])
                        EVE_values.append(repetition["EVE"][i])
                        stochasticity_values.append(stochasticity)
                    for repetition in variant_dist_dict["_".join(components)]:
                        PCT_values.append(repetition["PCT"][i])

                f = plt.figure(figsize=(10, 5))
                spec = f.add_gridspec(3, 1)
                ax1 = f.add_subplot(spec[:2, 0])
                ln1 = sns.lineplot(stochasticity_values, EV_values, estimator="mean", ci="sd", label="Mean Entropy",
                                   ax=ax1,
                                   color="red")
                ln2 = sns.lineplot(stochasticity_values, EVE_values, estimator="mean", ci="sd",
                                   label="Mean Entropy excl. 0s",
                                   ax=ax1,
                                   color="blue")
                ax1.set_ylabel("Mean Entropy")
                ax1.set_title(f"Entropy - Gen {generation}")
                if ax1.get_legend():
                    ax1.get_legend().remove()

                lns = ln2.get_lines()
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='best')

                ax3 = f.add_subplot(spec[2, 0])
                ax3.set_title("% Solutions Moving The Ball")
                ax3.set_ylim([0, 100])
                ax3.set_yticks([0, 25, 50, 75, 100])
                ax3.yaxis.grid(True)
                sns.lineplot(stochasticity_values, PCT_values, estimator="mean", ci="sd", ax=ax3)
                ax3.set_ylabel("%")
                ax3.set_xlabel("Stochasticity")

                # make space between subplots
                plt.subplots_adjust(hspace=0.6)

                if loss_type == "fulllosstrue":
                    plt.savefig(f"entropy{generation}_fullloss.png")
                else:
                    plt.savefig(f"entropy{generation}_notfullloss.png")
                plt.close()

            entropy_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "EV": EV_values,
                                                           "EVE": EVE_values}


        # plot latent var across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                LV_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_latent_var_dict["_".join(components)]:
                        LV_values.append(repetition["LV"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(5, 5))
                spec = f.add_gridspec(1, 1)
                ax1 = f.add_subplot(spec[0, 0])
                ln1 = sns.lineplot(stochasticity_values, LV_values, estimator="mean", ci="sd", label="Mean Variance", ax=ax1,
                                   color="red", linestyle="--")
                ax1.set_ylabel("Mean Variance")
                ax1.set_xlabel("Stochasticity")
                ax1.set_title(f"Variance of Latent Descriptors of No-Move Solutions - Gen {generation}")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"latent_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"latent_var{generation}_notfullloss.png")
                plt.close()

            latent_var_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "LV": LV_values}

        # # plot losses across stochasticity for each generation
        # for loss_type in ["fulllosstrue", "fulllossfalse"]:
        #     if variant != "vae" and loss_type == "fulllosstrue":
        #         continue
        #     components[2] = loss_type
        #     L2_values = []
        #     UL_values = []
        #     ENVAR_values = []
        #     VAR_values = []
        #     TL_values = []
        #     KL_values = []
        #     TSNE_values = []
        #
        #     stochasticity_values = []
        #     TSNE_stochasticity_values = []
        #
        #     is_data_recorded = True
        #     for stochasticity in stochasticities:
        #         # take correct dictionary according to stochasticity
        #         components[1] = f"random{stochasticity}"
        #
        #         # if we do not have the data at the moment, skip plotting
        #         if not "_".join(components) in variant_loss_dict:
        #             print(f"Loss data not recorded - Skipping. - Missing {'_'.join(components)}")
        #             is_data_recorded = False
        #             continue
        #         for repetition in variant_loss_dict["_".join(components)]:
        #             L2_values.append(repetition["L2"][START_GEN_LOSS_PLOT:])
        #             UL_values.append(repetition["UL"][START_GEN_LOSS_PLOT:])
        #             stochasticity_values.extend([stochasticity] * (len(repetition["L2"]) - START_GEN_LOSS_PLOT))
        #             # stochasticity_values.append(stochasticity)
        #
        #             if variant == "vae":
        #                 ENVAR_values.append(repetition["ENVAR"][START_GEN_LOSS_PLOT:])
        #                 KL_values.append(repetition["KL"][START_GEN_LOSS_PLOT:])
        #
        #             if loss_type == "fulllosstrue":
        #                 VAR_values.append(repetition["VAR"][START_GEN_LOSS_PLOT:])
        #                 TL_values.append(repetition["TL"][START_GEN_LOSS_PLOT:])
        #             if "TSNE" in repetition and repetition["TSNE"]:
        #                 TSNE_values.append(repetition["TSNE"][int(START_GEN_LOSS_PLOT/10 - 1):])
        #                 TSNE_stochasticity_values.append([stochasticity] * (len(repetition["TSNE"]) - int(START_GEN_LOSS_PLOT/10 - 1)))
        #
        #     if not is_data_recorded:
        #         continue
        #
        #     # TR_EPOCHS = repetition["TR_EPOCHS"]
        #
        #     stochasticity_values = np.array(stochasticity_values).flatten()
        #     L2_values = np.array(L2_values).flatten()
        #     UL_values = np.array(UL_values).flatten()
        #     if variant == "vae":
        #         ENVAR_values = np.array(ENVAR_values).flatten()
        #         KL_values = np.array(KL_values).flatten()
        #     if TSNE_values:
        #         TSNE_values = np.array(TSNE_values).flatten()
        #
        #     if loss_type == "fulllosstrue":
        #         f = plt.figure(figsize=(15, 10))
        #         spec = f.add_gridspec(3, 2)
        #     elif variant == "vae":
        #         f = plt.figure(figsize=(10, 5))
        #         spec = f.add_gridspec(2, 2)
        #         ax3 = f.add_subplot(spec[1, :])
        #     else:
        #         f = plt.figure(figsize=(5, 5))
        #         spec = f.add_gridspec(1, 2)
        #
        #     ax1 = f.add_subplot(spec[0, :])
        #
        #     if loss_type == "fulllosstrue":
        #         VAR_values = np.array(VAR_values).flatten()
        #         TL_values = np.array(TL_values).flatten()
        #
        #         ax2 = f.add_subplot(spec[1, :])
        #         ax3 = f.add_subplot(spec[2, :])
        #
        #         var_ax = ax1.twinx()
        #         ln3 = sns.lineplot(stochasticity_values, VAR_values, estimator="mean", ci="sd", label="Decoder Var.", ax=var_ax,
        #                            color="green")
        #         var_ax.set_ylabel("Variance")
        #         var_ax.get_legend().remove()
        #
        #         ln4 = sns.lineplot(stochasticity_values, TL_values, estimator="mean", ci="sd", label="Total Loss", ax=ax2, color="red")
        #         ax2.set_ylabel("Total Loss")
        #         KL_ax = ax2.twinx()
        #         ln5 = sns.lineplot(stochasticity_values, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
        #         KL_ax.set_ylabel("KL")
        #
        #         # first remove default legends automatically added then add combined set
        #         ax2.get_legend().remove()
        #         KL_ax.get_legend().remove()
        #         lns = ln4.get_lines() + ln5.get_lines()
        #         labs = [l.get_label() for l in lns]
        #         ax2.legend(lns, labs, loc='best')
        #
        #
        #     # plot overall L2 and actual L2
        #     ln1 = sns.lineplot(stochasticity_values, L2_values, estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
        #     ln2 = sns.lineplot(stochasticity_values, UL_values, estimator="mean", ci="sd", label="Undist. L2", ax=ax1, color="brown")
        #     ax1.set_ylabel("L2")
        #
        #     # add in legends, one return value of lineplot will have all lines on the axis
        #     if ax1.get_legend():
        #         ax1.get_legend().remove()
        #
        #     lns = ln2.get_lines() + ln3.get_lines() if "fulllosstrue" in exp else ln2.get_lines()
        #     labs = [l.get_label() for l in lns]
        #     ax1.legend(lns, labs, loc='best')
        #
        #     if variant == "vae":
        #         ln7 = sns.lineplot(stochasticity_values, ENVAR_values, estimator="mean", ci="sd", label="Encoder Var.", ax=ax3, color="red")
        #         labs = [l.get_label() for l in ln7.get_lines()]
        #         ax3.legend(ln7.get_lines(), labs, loc='best')
        #
        #     ax1.set_title(f"Losses")
        #     if variant == "vae":
        #         ax3.set_xlabel("Stochasticity")
        #     else:
        #         ax1.set_xlabel("Stochasticity")
        #
        #     if loss_type == "fulllosstrue":
        #         plt.savefig(f"losses_fullloss.png")
        #     else:
        #         plt.savefig(f"losses_notfullloss.png")
        #
        #     plt.close()
        #     loss_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "L2": L2_values, "ENVAR": ENVAR_values,
        #                                                    "UL": UL_values, "TSNE": TSNE_values, "KL": KL_values,
        #                                                 "VAR": VAR_values, "TSNEstoch": np.array(TSNE_stochasticity_values).flatten()}

    os.chdir(f"{EXP_FOLDER}")

    with open("diversity_data.pk", "wb") as f:
        pk.dump(diversity_stoch_dict, f)
    with open("loss_data.pk", "wb") as f:
        pk.dump(loss_stoch_dict, f)
    with open("dist_data.pk", "wb") as f:
        pk.dump(distance_stoch_dict, f)
    with open("entropy_data.pk", "wb") as f:
        pk.dump(entropy_stoch_dict, f)
    with open("posvar_data.pk", "wb") as f:
        pk.dump(pos_var_stoch_dict, f)
    with open("pct_moved_data.pk", "wb") as f:
        pk.dump(pct_stoch_dict, f)
    with open("recon_var_data.pk", "wb") as f:
        pk.dump(recon_var_stoch_dict, f)
    with open("latent_var_data.pk", "wb") as f:
        pk.dump(latent_var_stoch_dict, f)