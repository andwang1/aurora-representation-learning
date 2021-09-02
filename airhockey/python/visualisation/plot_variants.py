import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd
from visualisation.produce_name import produce_name

path = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/15k/isd"
os.chdir(path)

plotting_groups = [
    ["AURORA", "best"],
    # ["best", "sample"],
    ["AURORA", "beta0"],
    ["best", "beta0"],
]
colours = ["brown", "green"]
# colours = ["blue", "brown", "grey", "green", "purple", "red", "pink", "orange"]


skip_loss_type = {
    # "true"
}


# make legend bigger
plt.rc('legend', fontsize=35)
# make lines thicker
plt.rc('lines', linewidth=4, linestyle='-.')
# make font bigger
plt.rc('font', size=30)
# sns.set_style("dark")

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)
    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant:
                continue
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            if any({loss in variant for loss in skip_loss_type}):
                continue
            print(variant, member)
            print(data)
            variant_name = variant if not variant.startswith("ae") else "ae"
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            sns.lineplot(data["stoch"], data["UL"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            data_stats = pd.DataFrame(data)[["stoch", "UL"]].groupby("stoch").describe()
            quart25 = data_stats[("UL", '25%')]
            quart75 = data_stats[("UL", '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Losses - Undisturbed L2")
    ax1.set_ylabel("L2")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/losses_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/losses_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant:
                continue
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["L2"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            data_stats = pd.DataFrame(data)[["stoch", "L2"]].groupby("stoch").describe()
            quart25 = data_stats[("L2", '25%')]
            quart75 = data_stats[("L2", '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Losses - Total L2")
    ax1.set_ylabel("L2")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/total_losses_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/total_losses_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/diversity_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            print(data)
            sns.lineplot(data["stoch"], data["DIV"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1, color=colours[colour_count])
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            data_stats = pd.DataFrame(data)[["stoch", "DIV"]].groupby("stoch").describe()
            quart25 = data_stats[('DIV', '25%')]
            quart75 = data_stats[('DIV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Diversity Score")
    ax1.set_ylabel("Diversity")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/diversity_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/diversity_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PCT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PCT"]].groupby("stoch").describe()
            quart25 = data_stats[('PCT', '25%')]
            quart75 = data_stats[('PCT', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylim([0, 100])
    ax1.set_title("Proportion of Solutions Not Moving the Non-Random Puck")
    ax1.set_ylabel("%")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/pct_moved_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/pct_moved_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["MDE"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "MDE"]].groupby("stoch").describe()
            quart25 = data_stats[('MDE', '25%')]
            quart75 = data_stats[('MDE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Trajectory Distances Excl. No-Move Solutions")
    ax1.set_ylabel("Distance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/dist_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/dist_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["VDE"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "VDE"]].groupby("stoch").describe()
            quart25 = data_stats[('VDE', '25%')]
            quart75 = data_stats[('VDE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Variance in Trajectory Distances Excl. No-Move Solutions")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/dist_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/dist_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/entropy_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l1nosampletrain" in member and "fulllosstrue" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["EVE"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "EVE"]].groupby("stoch").describe()
            quart25 = data_stats[('EVE', '25%')]
            quart75 = data_stats[('EVE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Entropy of Trajectory Positions Excl. No-Move")
    ax1.set_ylabel("Entropy")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/entropy_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/entropy_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PV"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PV"]].groupby("stoch").describe()
            quart25 = data_stats[('PV', '25%')]
            quart75 = data_stats[('PV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.legend(loc='lower center')
    ax1.set_title("Variance in Trajectories")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/posvar_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/posvar_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PVE"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PVE"]].groupby("stoch").describe()
            quart25 = data_stats[('PVE', '25%')]
            quart75 = data_stats[('PVE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.legend(loc='lower center')
    ax1.set_title("Variance in Trajectories excl. No-Move")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/posvar_exclnomove_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/posvar_exclnomove_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/recon_var_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["NMV"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "NMV"]].groupby("stoch").describe()
            quart25 = data_stats[('NMV', '25%')]
            quart75 = data_stats[('NMV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Variance in Constructions of No-Move solutions")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/recon_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/recon_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/latent_var_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["LV"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "LV"]].groupby("stoch").describe()
            quart25 = data_stats[('LV', '25%')]
            quart75 = data_stats[('LV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Variance in Latent Descriptors of No-Move Solutions")
    plt.savefig(f"{save_dir}/pdf/latent_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/latent_var_{'_'.join(group)}.png")
    plt.close()



    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant or variant.startswith("ae") or "beta0" in member:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            sns.lineplot(data["stoch"], data["KL"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            data = pd.DataFrame(data)[["stoch", "KL"]]
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('KL', '25%')]
            quart75 = data_stats[('KL', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.set_ylabel("KL")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title("KL Loss")
    plt.savefig(f"{save_dir}/pdf/kl_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/kl_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant or variant.startswith("ae") or "nosampletrain" in member or "ENVAR" not in data:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            sns.lineplot(data["stoch"], data["ENVAR"] / 2, estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            data = pd.DataFrame(data)[["stoch", "ENVAR"]]
            data["ENVAR"] /= 2
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('ENVAR', '25%')]
            quart75 = data_stats[('ENVAR', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Encoder Variance")
    plt.savefig(f"{save_dir}/pdf/encoder_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/encoder_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant or variant.startswith("ae") or "VAR" not in data:
                continue
            if len(data["VAR"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            data.pop("TSNE", None)
            data.pop("TSNEstoch", None)
            sns.lineplot(data["stoch"], data["VAR"] / 400, estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            data = pd.DataFrame(data)[["stoch", "VAR"]]
            data["VAR"] /= 400
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('VAR', '25%')]
            quart75 = data_stats[('VAR', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Decoder Variance")
    plt.savefig(f"{save_dir}/pdf/decoder_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/decoder_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant or variant.startswith("ae") or "TSNE" not in data or len(data["TSNE"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            sns.lineplot(data["TSNEstoch"], data["TSNE"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            data = {"TSNEstoch": data["TSNEstoch"], "TSNE": data["TSNE"]}
            data = pd.DataFrame(data)[["TSNEstoch", "TSNE"]]
            data_stats = data.groupby("TSNEstoch").describe()
            quart25 = data_stats[('TSNE', '25%')]
            quart75 = data_stats[('TSNE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("SNE" if "tsne" not in member else "T-SNE")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title("SNE" if "tsne" not in member else "T-SNE")
    plt.savefig(f"{save_dir}/pdf/tsne_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/tsne_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 10))
    f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, 0])
    ax2 = f.add_subplot(spec[0, 1])

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PCT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PCT"]].groupby("stoch").describe()
            quart25 = data_stats[('PCT', '25%')]
            quart75 = data_stats[('PCT', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylim([0, 100])
    ax1.set_title("Proportion of No-Move Solutions", y = 1.02)
    ax1.set_ylabel("%")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            if "l2nosampletrain" in member and "fulllosstrue" in variant or "snebeta0_nosampletrain" in member and "fulllossfalse" in variant:
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PV"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax2,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PV"]].groupby("stoch").describe()
            quart25 = data_stats[('PV', '25%')]
            quart75 = data_stats[('PV', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax2.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.get_legend().remove()
    ax2.legend(loc='best')
    ax2.set_title("Variance in Puck Trajectories", y=1.02)
    ax2.set_ylabel("Variance")
    plt.subplots_adjust(wspace=0.25)
    # ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/pctmoveposvar_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/pctmoveposvar_{'_'.join(group)}.png")