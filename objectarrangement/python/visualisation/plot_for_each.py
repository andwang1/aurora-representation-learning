import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pk
import pandas as pd
import seaborn as sns
from collections import defaultdict
from visualisation.distances import plot_dist_in_dir
from visualisation.pos_var import plot_pos_var_in_dir
from visualisation.pct_move import plot_pct_move_in_dir
from visualisation.ae_loss_AE import plot_loss_in_dir_AE
from visualisation.ae_loss_VAE import plot_loss_in_dir_VAE
from visualisation.latent_space import plot_latent_space_in_dir

GENERATE_PID_IMAGES = False
GENERATE_EXP_IMAGES = False
START_GEN_LOSS_PLOT = 500

results_dir = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2"
groups = {group_name for group_name in os.listdir(results_dir) if
            os.path.isdir(os.path.join(results_dir, group_name)) and group_name != "plots"}

# exclude_dirs = {"huber"}
# groups -= exclude_dirs

only_dirs = {
"AURORA_smaller",
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
    BASE_NAME = "results_arrangementsd_"
    variants = [exp_name.split("_")[-1] for exp_name in os.listdir(EXP_FOLDER) if
                os.path.isdir(os.path.join(EXP_FOLDER, exp_name))]

    # store all data
    loss_stoch_dict = {}
    distance_stoch_dict = {}
    pos_var_stoch_dict = {}
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

        variant_loss_dict = defaultdict(list)
        variant_dist_dict = defaultdict(list)
        variant_pos_var_dict = defaultdict(list)
        variant_latent_dict = defaultdict(list)
        variant_pct_dict = defaultdict(list)

        for i, exp in enumerate(exp_names):
            exp_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}"
            pids = [pid for pid in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, pid))]
            for pid in pids:
                full_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}/{pid}"
                print(f"PROCESSING - {full_path}")
                variant_latent_dict[exp].append(plot_latent_space_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_dist_dict[exp].append(plot_dist_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_pos_var_dict[exp].append(plot_pos_var_in_dir(full_path, GENERATE_PID_IMAGES))
                variant_pct_dict[exp].append(plot_pct_move_in_dir(full_path, GENERATE_PID_IMAGES))
                # PID level plotting
                if "manual" not in exp_path:
                    if variant == "vae":
                        variant_loss_dict[exp].append(plot_loss_in_dir_VAE(full_path, is_full_loss[i], GENERATE_PID_IMAGES))
                    else:
                        variant_loss_dict[exp].append(plot_loss_in_dir_AE(full_path, GENERATE_PID_IMAGES))

            if not GENERATE_EXP_IMAGES:
                continue

            # experiment level plotting
            os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}")

            # at experiment level plot distance metrices
            LBD_values = []
            UBD_values = []
            ULBD_values = []
            UUBD_values = []
            generations = []
            for repetition in variant_dist_dict[exp]:
                for i in repetition["LBD"]:
                    LBD_values.extend(i)
                for i in repetition["UBD"]:
                    UBD_values.extend(i)
                for i in repetition["ULBD"]:
                    ULBD_values.extend(i)
                for i in repetition["UUBD"]:
                    UUBD_values.extend(i)
                for i in repetition["gen"]:
                    generations.extend(i)

            f = plt.figure(figsize=(6, 10))
            spec = f.add_gridspec(2, 1)
            ax1 = f.add_subplot(spec[0, 0])
            ln1 = sns.lineplot(generations, LBD_values, estimator=np.median, ci=None, label="Object 1", ax=ax1,
                               color="red")
            ln2 = sns.lineplot(generations, UBD_values, estimator=np.median, ci=None, label="Object 2", ax=ax1,
                               color="blue")
            ax1.set_title("With Noise")

            ax2 = f.add_subplot(spec[1, 0])
            ln3 = sns.lineplot(generations, ULBD_values, estimator=np.median, ci=None, label="Object 1", ax=ax2, color="red")
            ln4 = sns.lineplot(generations, UUBD_values, estimator=np.median, ci=None, label="Object 2", ax=ax2,
                               color="blue")
            ax2.set_title("Without Noise")
            plt.suptitle("Distance Travelled by Objects")
            ax1.set_ylabel("Distance")
            ax2.set_ylabel("Distance")
            ax2.set_xlabel("Generation")

            plt.savefig("distance.png")
            plt.close()

            # plot pos_var at experiment level
            LOWVAR_values = np.array([repetition["LOWVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            UPPVAR_values = np.array([repetition["UPPVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            BOTVAR_values = np.array([repetition["BOTVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            UNLOWVAR_values = np.array([repetition["UNLOWVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            UNUPPVAR_values = np.array([repetition["UNUPPVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            UNBOTVAR_values = np.array([repetition["UNBOTVAR"] for repetition in variant_pos_var_dict[exp]]).flatten()
            generations = np.array([repetition["gen"] for repetition in variant_pos_var_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 10))
            spec = f.add_gridspec(2, 1)
            ax1 = f.add_subplot(spec[0, 0])

            ln1 = sns.lineplot(generations, LOWVAR_values, estimator=np.median, ci=None, label="Object 1", ax=ax1,
                               color="red")
            ln2 = sns.lineplot(generations, UPPVAR_values, estimator=np.median, ci=None, label="Object 2", ax=ax1,
                               color="blue")
            ln3 = sns.lineplot(generations, BOTVAR_values, estimator=np.median, ci=None, label="Both Objects", ax=ax1,
                               color="green")

            ax1.set_ylabel("Variance")
            ax1.set_title("With Noise")

            ax2 = f.add_subplot(spec[1, 0])
            ln1 = sns.lineplot(generations, UNLOWVAR_values, estimator=np.median, ci=None, label="Object 1", ax=ax2,
                               color="red")
            ln2 = sns.lineplot(generations, UNUPPVAR_values, estimator=np.median, ci=None, label="Object 2", ax=ax2,
                               color="blue")
            ln3 = sns.lineplot(generations, UNBOTVAR_values, estimator=np.median, ci=None, label="Both Objects", ax=ax2,
                               color="green")
            ax2.set_title("Without Noise")
            ax2.set_ylabel("Variance")
            plt.suptitle("Variance in Object Positions")
            plt.xlabel("Generation")

            plt.savefig("pos_var.png")
            plt.close()

            # plot latent var at experiment level
            LV_values = np.array([repetition["LV"] for repetition in variant_latent_dict[exp]]).flatten()
            generations = np.array([repetition["gen"] for repetition in variant_latent_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(1, 1)
            ax1 = f.add_subplot(spec[0, 0])
            ln1 = sns.lineplot(generations, LV_values, estimator=np.median, ci=None, ax=ax1,
                               color="red", linestyle="--")
            data_stats = pd.DataFrame({"x": generations, "y": LV_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                             quart25, quart75, alpha=0.3, color="red")
            ax1.set_ylabel("Variance")
            ax1.set_xlabel("Generation")
            ax1.set_title("Variance of Latent Descriptors of No-Move Solutions")

            plt.savefig("latent_var.png")
            plt.close()

            # plot pct move at experiment level
            PLOW_values = np.array([repetition["PLOW"] for repetition in variant_pct_dict[exp]]).flatten()
            PUPP_values = np.array([repetition["PUPP"] for repetition in variant_pct_dict[exp]]).flatten()
            PEIT_values = np.array([repetition["PEIT"] for repetition in variant_pct_dict[exp]]).flatten()
            PBOT_values = np.array([repetition["PBOT"] for repetition in variant_pct_dict[exp]]).flatten()
            generations = np.array([repetition["gen"] for repetition in variant_pct_dict[exp]]).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(1, 1)
            ax1 = f.add_subplot(spec[0, 0])

            ln1 = sns.lineplot(generations, PLOW_values, estimator=np.median, ci=None, ax=ax1,
                               color="red", label="% Moving Lower Object")
            data_stats = pd.DataFrame({"x": generations, "y": PLOW_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                             quart25, quart75, alpha=0.3, color="red")
            ln2 = sns.lineplot(generations, PUPP_values, estimator=np.median, ci=None, ax=ax1,
                               color="blue", label="% Moving Upper Object")
            data_stats = pd.DataFrame({"x": generations, "y": PUPP_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                             quart25, quart75, alpha=0.3, color="blue")
            ln3 = sns.lineplot(generations, PEIT_values, estimator=np.median, ci=None, ax=ax1,
                               color="green", label="% Moving Either Object")
            data_stats = pd.DataFrame({"x": generations, "y": PEIT_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                             quart25, quart75, alpha=0.3, color="green")
            ln4 = sns.lineplot(generations, PBOT_values, estimator=np.median, ci=None, ax=ax1,
                               color="orange", label="% Moving Both Objects")
            data_stats = pd.DataFrame({"x": generations, "y": PBOT_values}).groupby("x").describe()
            quart25 = data_stats[("y", '25%')]
            quart75 = data_stats[("y", '75%')]
            ax1.fill_between([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                             quart25, quart75, alpha=0.3, color="orange")

            ax1.set_ylabel("%")
            ax1.set_xlabel("Generation")
            ax1.set_title("% Solutions Moving Objects")

            plt.savefig("pct_move.png")
            plt.close()

            # at experiment level, plot losses
            if "manual" in exp_path:
                continue
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
            ln1 = sns.lineplot(x, L2_values.flatten(), estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
            ln2 = sns.lineplot(x, UL_values, estimator="mean", ci="sd", label="Undist. L2", ax=ax1, color="brown")
            ax1.set_ylabel("L2")

            # add in legends, one return value of lineplot will have all lines on the axis
            ax1.get_legend().remove()

            lns = ln2.get_lines() + ln3.get_lines() if "fulllosstrue" in exp else ln2.get_lines()
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='best')

            if variant == "vae":
                KL_ax = ax3.twinx()
                ln5 = sns.lineplot(x, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
                KL_ax.set_ylabel("KL")

                ln7 = sns.lineplot(x, ENVAR_values, estimator="mean", ci="sd", label="Encoder Var.", ax=ax3, color="red")
                labs = [l.get_label() for l in ln7.get_lines()]
                ax3.legend(ln7.get_lines(), labs, loc='best')

                ax3.get_legend().remove()
                KL_ax.get_legend().remove()
                lns = ln7.get_lines() + ln5.get_lines()
                labs = [l.get_label() for l in lns]
                ax3.legend(lns, labs, loc='best')

            ax1.set_title(f"Losses")
            if variant == "vae":
                ax3.set_xlabel("Generation")
            else:
                ax1.set_xlabel("Generation")

            plt.savefig("losses.png")
            plt.close()

        # variant plotting
        # retrieve stochasticity levels from file names
        os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
        generations = list(variant_latent_dict[exp][0]["gen"])
        all_experiments = variant_latent_dict.keys()
        stochasticities = []

        for name in all_experiments:
            components = name.split("_")
            # "random" part of experiment name
            stochasticities.append((components[1][len("random"):]))

        # remove duplicates from fulllosstrue and fulllossfalse and sort
        stochasticities = sorted(list(set(stochasticities)))

        # plot distances across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                LBD_values = []
                UBD_values = []
                ULBD_values = []
                UUBD_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_dist_dict["_".join(components)]:
                        LBD_values.extend(repetition["LBD"][i])
                        UBD_values.extend(repetition["UBD"][i])
                        ULBD_values.extend(repetition["ULBD"][i])
                        UUBD_values.extend(repetition["UUBD"][i])
                        stochasticity_values.extend([stochasticity] * len(repetition["LBD"][i]))

                f = plt.figure(figsize=(6, 10))
                spec = f.add_gridspec(2, 1)
                ax1 = f.add_subplot(spec[0, 0])

                ln1 = sns.lineplot(stochasticity_values, LBD_values, estimator=np.median, ci=None, label="Object 1", ax=ax1,
                                   color="red")
                ln2 = sns.lineplot(stochasticity_values, UBD_values, estimator=np.median, ci=None, label="Object 2", ax=ax1,
                                   color="blue")
                ax1.set_title("With Noise")

                ax2 = f.add_subplot(spec[1, 0])
                ln3 = sns.lineplot(stochasticity_values, ULBD_values, estimator=np.median, ci=None, label="Object 1", ax=ax2,
                                   color="red")
                ln4 = sns.lineplot(stochasticity_values, UUBD_values, estimator=np.median, ci=None, label="Object 2", ax=ax2,
                                   color="blue")
                ax2.set_title("Without Noise")
                plt.suptitle(f"Distance Travelled by Objects - Gen {generation}")
                ax1.set_ylabel("Distance")
                ax2.set_ylabel("Distance")
                ax2.set_xlabel("Stochasticity")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"distance_gen{generation}_fullloss.png")
                else:
                    plt.savefig(f"distance_gen{generation}_notfullloss.png")
                plt.close()
                plt.close()

            # record last generation
            distance_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values,
            "LBD": LBD_values, "UBD": UBD_values, "ULBD": ULBD_values, "UUBD": UUBD_values}

        # plot pos var across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                LOWVAR_values = []
                UPPVAR_values = []
                BOTVAR_values = []
                UNLOWVAR_values = []
                UNUPPVAR_values = []
                UNBOTVAR_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_pos_var_dict["_".join(components)]:
                        LOWVAR_values.append(repetition["LOWVAR"][i])
                        UPPVAR_values.append(repetition["UPPVAR"][i])
                        BOTVAR_values.append(repetition["BOTVAR"][i])
                        UNLOWVAR_values.append(repetition["UNLOWVAR"][i])
                        UNUPPVAR_values.append(repetition["UNUPPVAR"][i])
                        UNBOTVAR_values.append(repetition["UNBOTVAR"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(10, 10))
                spec = f.add_gridspec(2, 1)
                ax1 = f.add_subplot(spec[0, 0])

                ln1 = sns.lineplot(stochasticity_values, LOWVAR_values, estimator=np.median, ci=None, label="Object 1", ax=ax1,
                                   color="red")
                ln2 = sns.lineplot(stochasticity_values, UPPVAR_values, estimator=np.median, ci=None, label="Object 2", ax=ax1,
                                   color="blue")
                ln3 = sns.lineplot(stochasticity_values, BOTVAR_values, estimator=np.median, ci=None, label="Both Objects",
                                   ax=ax1,
                                   color="green")

                ax1.set_ylabel("Variance")
                ax1.set_title("With Noise")

                ax2 = f.add_subplot(spec[1, 0])
                ln1 = sns.lineplot(stochasticity_values, UNLOWVAR_values, estimator=np.median, ci=None, label="Object 1", ax=ax2,
                                   color="red")
                ln2 = sns.lineplot(stochasticity_values, UNUPPVAR_values, estimator=np.median, ci=None, label="Object 2", ax=ax2,
                                   color="blue")
                ln3 = sns.lineplot(stochasticity_values, UNBOTVAR_values, estimator=np.median, ci=None, label="Both Objects",
                                   ax=ax2,
                                   color="green")
                ax2.set_title("Without Noise")
                ax2.set_ylabel("Variance")
                plt.suptitle("Variance in Object Positions")
                plt.xlabel("Stochasticity")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"pos_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"pos_var{generation}_notfullloss.png")
                plt.close()

            pos_var_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "LOWVAR": LOWVAR_values, "UPPVAR": UPPVAR_values, "BOTVAR": BOTVAR_values,
                 "UNLOWVAR": UNLOWVAR_values, "UNUPPVAR": UNUPPVAR_values, "UNBOTVAR": UNBOTVAR_values}

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
                    for repetition in variant_latent_dict["_".join(components)]:
                        LV_values.append(repetition["LV"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(5, 5))
                spec = f.add_gridspec(1, 1)
                ax1 = f.add_subplot(spec[0, 0])
                ln1 = sns.lineplot(stochasticity_values, LV_values, estimator=np.median, ci=None, ax=ax1,
                                   color="red", linestyle="--")
                ax1.set_ylabel("Variance")
                ax1.set_xlabel("Stochasticity")
                ax1.set_title(f"Variance of Latent Descriptors of No-Move Solutions - Gen {generation}")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"latent_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"latent_var{generation}_notfullloss.png")
                plt.close()

            latent_var_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "LV": LV_values}

        # plot pct move across stochasticity for each generation
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            for i, generation in enumerate(generations):
                PLOW_values = []
                PUPP_values = []
                PEIT_values = []
                PBOT_values = []
                stochasticity_values = []

                for stochasticity in stochasticities:
                    # take correct dictionary according to stochasticity
                    components[1] = f"random{stochasticity}"
                    components[2] = loss_type
                    for repetition in variant_pct_dict["_".join(components)]:
                        PLOW_values.append(repetition["PLOW"][i])
                        PUPP_values.append(repetition["PUPP"][i])
                        PEIT_values.append(repetition["PEIT"][i])
                        PBOT_values.append(repetition["PBOT"][i])
                        stochasticity_values.append(stochasticity)

                f = plt.figure(figsize=(10, 5))
                spec = f.add_gridspec(1, 1)
                ax1 = f.add_subplot(spec[0, 0])

                ln1 = sns.lineplot(stochasticity_values, PLOW_values, estimator=np.median, ci=None,
                                   ax=ax1,
                                   color="red", label="% Moving Lower Object")
                ln2 = sns.lineplot(stochasticity_values, PUPP_values, estimator=np.median, ci=None,
                                   ax=ax1,
                                   color="blue", label="% Moving Upper Object")
                ln3 = sns.lineplot(stochasticity_values, PEIT_values, estimator=np.median, ci=None,
                                   ax=ax1,
                                   color="green", label="% Moving Either Object")

                ln4 = sns.lineplot(stochasticity_values, PBOT_values, estimator=np.median, ci=None,
                                   ax=ax1,
                                   color="orange", label="% Moving Both Objects")
                ax1.set_ylabel("%")
                ax1.set_title("% Solutions Moving Objects")
                ax1.set_xlabel("Stochasticity")

                if loss_type == "fulllosstrue":
                    plt.savefig(f"pos_var{generation}_fullloss.png")
                else:
                    plt.savefig(f"pos_var{generation}_notfullloss.png")
                plt.close()

            pct_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "PLOW": PLOW_values,
                                                           "PUPP": PUPP_values, "PEIT": PEIT_values,
                                                           "PBOT": PBOT_values}

        # plot losses across stochasticity for each generation
        if "manual" in variant:
            continue
        for loss_type in ["fulllosstrue", "fulllossfalse"]:
            if variant != "vae" and loss_type == "fulllosstrue":
                continue
            components[2] = loss_type
            L2_values = []
            UL_values = []
            ENVAR_values = []
            VAR_values = []
            TL_values = []
            KL_values = []
            TSNE_values = []

            stochasticity_values = []
            TSNE_stochasticity_values = []

            is_data_recorded = True
            for stochasticity in stochasticities:
                # take correct dictionary according to stochasticity
                components[1] = f"random{stochasticity}"

                # if we do not have the data at the moment, skip plotting
                if not "_".join(components) in variant_loss_dict:
                    print(f"Loss data not recorded - Skipping. - Missing {'_'.join(components)}")
                    is_data_recorded = False
                    continue
                for repetition in variant_loss_dict["_".join(components)]:
                    L2_values.append(repetition["L2"][START_GEN_LOSS_PLOT:])
                    UL_values.append(repetition["UL"][START_GEN_LOSS_PLOT:])
                    stochasticity_values.extend([stochasticity] * (len(repetition["L2"]) - START_GEN_LOSS_PLOT))
                    # stochasticity_values.append(stochasticity)

                    if variant == "vae":
                        ENVAR_values.append(repetition["ENVAR"][START_GEN_LOSS_PLOT:])
                        KL_values.append(repetition["KL"][START_GEN_LOSS_PLOT:])

                    if loss_type == "fulllosstrue":
                        VAR_values.append(repetition["VAR"][START_GEN_LOSS_PLOT:])
                        TL_values.append(repetition["TL"][START_GEN_LOSS_PLOT:])
                    if "TSNE" in repetition and repetition["TSNE"]:
                        TSNE_values.append(repetition["TSNE"][int(START_GEN_LOSS_PLOT/10 - 1):])
                        TSNE_stochasticity_values.append([stochasticity] * (len(repetition["TSNE"]) - int(START_GEN_LOSS_PLOT/10 - 1)))

            if not is_data_recorded:
                continue

            # TR_EPOCHS = repetition["TR_EPOCHS"]

            stochasticity_values = np.array(stochasticity_values).flatten()
            L2_values = np.array(L2_values).flatten()
            UL_values = np.array(UL_values).flatten()
            if variant == "vae":
                ENVAR_values = np.array(ENVAR_values).flatten()
                KL_values = np.array(KL_values).flatten()
            if TSNE_values:
                TSNE_values = np.array(TSNE_values).flatten()

            if loss_type == "fulllosstrue":
                f = plt.figure(figsize=(15, 10))
                spec = f.add_gridspec(3, 2)
            elif variant == "vae":
                f = plt.figure(figsize=(10, 5))
                spec = f.add_gridspec(2, 2)
                ax3 = f.add_subplot(spec[1, :])
            else:
                f = plt.figure(figsize=(5, 5))
                spec = f.add_gridspec(1, 2)

            ax1 = f.add_subplot(spec[0, :])

            if loss_type == "fulllosstrue":
                VAR_values = np.array(VAR_values).flatten()
                TL_values = np.array(TL_values).flatten()

                ax2 = f.add_subplot(spec[1, :])
                ax3 = f.add_subplot(spec[2, :])

                var_ax = ax1.twinx()
                ln3 = sns.lineplot(stochasticity_values, VAR_values, estimator="mean", ci="sd", label="Decoder Var.", ax=var_ax,
                                   color="green")
                var_ax.set_ylabel("Variance")
                var_ax.get_legend().remove()

                ln4 = sns.lineplot(stochasticity_values, TL_values, estimator="mean", ci="sd", label="Total Loss", ax=ax2, color="red")
                ax2.set_ylabel("Total Loss")
                KL_ax = ax2.twinx()
                ln5 = sns.lineplot(stochasticity_values, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
                KL_ax.set_ylabel("KL")

                # first remove default legends automatically added then add combined set
                ax2.get_legend().remove()
                KL_ax.get_legend().remove()
                lns = ln4.get_lines() + ln5.get_lines()
                labs = [l.get_label() for l in lns]
                ax2.legend(lns, labs, loc='best')


            # plot overall L2 and actual L2
            ln1 = sns.lineplot(stochasticity_values, L2_values, estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
            ln2 = sns.lineplot(stochasticity_values, UL_values, estimator="mean", ci="sd", label="Undist. L2", ax=ax1, color="brown")
            ax1.set_ylabel("L2")

            # add in legends, one return value of lineplot will have all lines on the axis
            if ax1.get_legend():
                ax1.get_legend().remove()

            lns = ln2.get_lines() + ln3.get_lines() if "fulllosstrue" in exp else ln2.get_lines()
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='best')

            if variant == "vae":
                ln7 = sns.lineplot(stochasticity_values, ENVAR_values, estimator="mean", ci="sd", label="Encoder Var.", ax=ax3, color="red")
                labs = [l.get_label() for l in ln7.get_lines()]
                ax3.legend(ln7.get_lines(), labs, loc='best')

            ax1.set_title(f"Losses")
            if variant == "vae":
                ax3.set_xlabel("Stochasticity")
            else:
                ax1.set_xlabel("Stochasticity")

            if loss_type == "fulllosstrue":
                plt.savefig(f"losses_fullloss.png")
            else:
                plt.savefig(f"losses_notfullloss.png")

            plt.close()
            loss_stoch_dict[f"{variant}{loss_type}"] = {"stoch": stochasticity_values, "L2": L2_values, "ENVAR": ENVAR_values,
                                                           "UL": UL_values, "TSNE": TSNE_values, "KL": KL_values,
                                                        "VAR": VAR_values, "TSNEstoch": np.array(TSNE_stochasticity_values).flatten()}

    os.chdir(f"{EXP_FOLDER}")

    with open("loss_data.pk", "wb") as f:
        pk.dump(loss_stoch_dict, f)
    with open("dist_data.pk", "wb") as f:
        pk.dump(distance_stoch_dict, f)
    with open("posvar_data.pk", "wb") as f:
        pk.dump(pos_var_stoch_dict, f)
    with open("pct_moved_data.pk", "wb") as f:
        pk.dump(pct_stoch_dict, f)
    with open("latent_var_data.pk", "wb") as f:
        pk.dump(latent_var_stoch_dict, f)