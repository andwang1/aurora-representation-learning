import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd
from visualisation.produce_name import produce_name

path = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2"
os.chdir(path)

plotting_groups = [
    # ["AURORA", "best"],
# ["AURORA", "beta0"],
#     ["best", "beta0"],
    ["AURORA", "AURORA_smaller", "beta0", "largerbeta0",  "sample", "best",  "largerbeta1", ]
# ["best", "sample"],
    # ["AURORA", "manualBD"],
    # ["best", "manualBD"]
]
# colours = ["brown", "green"]
colours = [ "brown" ,"green", "grey",  "blue", "purple", "red", "pink", "orange"]


skip_loss_type = {
    # "false"
}


# make legend bigger
plt.rc('legend', fontsize=28)
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

    f = plt.figure(figsize=(40, 10))
    f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
    spec = f.add_gridspec(1, 4)
    ax1 = f.add_subplot(spec[0, 0])
    ax2 = f.add_subplot(spec[0, 1])
    ax3 = f.add_subplot(spec[0, 2])
    ax4 = f.add_subplot(spec[0, 3])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PLOW"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PLOW"]].groupby("stoch").describe()
            quart25 = data_stats[('PLOW', '25%')]
            quart75 = data_stats[('PLOW', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Object 1")
    ax1.set_ylim([-5, 100])
    ax1.set_ylabel("%")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PUPP"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax2,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PUPP"]].groupby("stoch").describe()
            quart25 = data_stats[('PUPP', '25%')]
            quart75 = data_stats[('PUPP', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax2.lines[-1].set_linestyle("--")
            colour_count += 1
    ax2.set_title("Object 2")
    ax2.set_ylim([-5, 100])

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PEIT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax3,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PEIT"]].groupby("stoch").describe()
            quart25 = data_stats[('PEIT', '25%')]
            quart75 = data_stats[('PEIT', '75%')]
            ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax3.lines[-1].set_linestyle("--")
            colour_count += 1
    ax3.set_title("Any Object")
    ax3.set_ylim([-5, 100])
    # ax3.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PBOT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax4,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PBOT"]].groupby("stoch").describe()
            quart25 = data_stats[('PBOT', '25%')]
            quart75 = data_stats[('PBOT', '75%')]
            ax4.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax4.lines[-1].set_linestyle("--")
            colour_count += 1
    ax4.set_title("Object 1 and Object 2")
    ax4.set_ylim([-5, 100])
    # ax4.set_xlabel("Stochasticity")

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    # ax3.get_legend().remove()
    ax4.get_legend().remove()
    plt.suptitle("Proportion of Controllers Not Moving...")

    plt.savefig(f"{save_dir}/pdf/pct_moved_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/pct_moved_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(2, 2)
    ax1 = f.add_subplot(spec[0, 0])
    ax2 = f.add_subplot(spec[0, 1])
    ax3 = f.add_subplot(spec[1, 0])
    ax4 = f.add_subplot(spec[1, 1])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["LBD"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "LBD"]].groupby("stoch").describe()
            quart25 = data_stats[('LBD', '25%')]
            quart75 = data_stats[('LBD', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Object 1 with Noise")
    ax1.set_ylabel("Distance")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)
        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UBD"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax2,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UBD"]].groupby("stoch").describe()
            quart25 = data_stats[('UBD', '25%')]
            quart75 = data_stats[('UBD', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax2.lines[-1].set_linestyle("--")
            colour_count += 1
    ax2.set_title("Object 2 with Noise")


    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)
        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["ULBD"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax3,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "ULBD"]].groupby("stoch").describe()
            quart25 = data_stats[('ULBD', '25%')]
            quart75 = data_stats[('ULBD', '75%')]
            ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax3.lines[-1].set_linestyle("--")
            colour_count += 1
    ax3.set_title("Object 1")
    ax3.set_ylabel("Distance")
    ax3.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)
        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UUBD"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax4,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UUBD"]].groupby("stoch").describe()
            quart25 = data_stats[('UUBD', '25%')]
            quart75 = data_stats[('UUBD', '75%')]
            ax4.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax4.lines[-1].set_linestyle("--")
            colour_count += 1
    ax4.set_title("Object 2")
    ax4.set_xlabel("Stochasticity")
    plt.suptitle("Distance Moved")
    plt.savefig(f"{save_dir}/pdf/dist{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/dist{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(30, 20))
    spec = f.add_gridspec(2, 3)
    ax1 = f.add_subplot(spec[0, 0])
    ax2 = f.add_subplot(spec[0, 1])
    ax3 = f.add_subplot(spec[0, 2])
    ax4 = f.add_subplot(spec[1, 0])
    ax5 = f.add_subplot(spec[1, 1])
    ax6 = f.add_subplot(spec[1, 2])

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["LOWVAR"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "LOWVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('LOWVAR', '25%')]
            quart75 = data_stats[('LOWVAR', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Object 1 with Noise")
    ax1.set_ylabel("Variance")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UPPVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax2,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UPPVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UPPVAR', '25%')]
            quart75 = data_stats[('UPPVAR', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax2.lines[-1].set_linestyle("--")
            colour_count += 1
    ax2.set_title("Object 2 with Noise")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["BOTVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax3,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "BOTVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('BOTVAR', '25%')]
            quart75 = data_stats[('BOTVAR', '75%')]
            ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax3.lines[-1].set_linestyle("--")
            colour_count += 1
    ax3.set_title("Both Objects with Noise")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNLOWVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax4,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNLOWVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNLOWVAR', '25%')]
            quart75 = data_stats[('UNLOWVAR', '75%')]
            ax4.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax4.lines[-1].set_linestyle("--")
            colour_count += 1
    ax4.set_title("Object 1")
    ax4.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNUPPVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax5,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNUPPVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNUPPVAR', '25%')]
            quart75 = data_stats[('UNUPPVAR', '75%')]
            ax5.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax5.lines[-1].set_linestyle("--")
            colour_count += 1
    ax5.set_title("Object 2")
    ax5.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNBOTVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax6,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNBOTVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNBOTVAR', '25%')]
            quart75 = data_stats[('UNBOTVAR', '75%')]
            ax6.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax6.lines[-1].set_linestyle("--")
            colour_count += 1
    ax6.set_title("Both Objects")
    ax6.set_xlabel("Stochasticity")
    plt.suptitle("Variance in Object Positions")

    plt.savefig(f"{save_dir}/pdf/posvar_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/posvar_{'_'.join(group)}.png")
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
    # ax1.yaxis.get_major_formatter().set_useOffset(False)
    # ax1.yaxis.get_major_formatter().set_scientific(False)
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

    f = plt.figure(figsize=(30, 10))
    # f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
    spec = f.add_gridspec(1, 3)
    ax4 = f.add_subplot(spec[0, 0])
    ax5 = f.add_subplot(spec[0, 1])
    ax6 = f.add_subplot(spec[0, 2])

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNLOWVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax4,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNLOWVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNLOWVAR', '25%')]
            quart75 = data_stats[('UNLOWVAR', '75%')]
            ax4.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax4.lines[-1].set_linestyle("--")
            colour_count += 1
    ax4.set_title("Object 1")
    ax4.set_ylabel("Variance")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNUPPVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax5,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNUPPVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNUPPVAR', '25%')]
            quart75 = data_stats[('UNUPPVAR', '75%')]
            ax5.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax5.lines[-1].set_linestyle("--")
            colour_count += 1
    ax5.set_title("Object 2")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNBOTVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax6,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNBOTVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNBOTVAR', '25%')]
            quart75 = data_stats[('UNBOTVAR', '75%')]
            ax6.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax6.lines[-1].set_linestyle("--")
            colour_count += 1
    ax6.set_title("Both Objects")
    plt.suptitle("Variance in Object Positions")

    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()

    ax5.set_xlabel("Environment Stochasticity")

    # shrink boxes to make space for legend
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    box = ax5.get_position()
    ax5.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    box = ax6.get_position()
    ax6.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below the plot
    leg = ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), fancybox=True,
               ncol=len(group))
    leg.get_frame().set_edgecolor('black')

    plt.savefig(f"{save_dir}/pdf/posvar_onlynonoise_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/posvar_onlynonoise_{'_'.join(group)}.png")
    plt.close()


    f = plt.figure(figsize=(30, 10))
    f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
    spec = f.add_gridspec(1, 3)
    ax1 = f.add_subplot(spec[0, 0])
    ax2 = f.add_subplot(spec[0, 1])
    ax3 = f.add_subplot(spec[0, 2])

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PEIT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PEIT"]].groupby("stoch").describe()
            quart25 = data_stats[('PEIT', '25%')]
            quart75 = data_stats[('PEIT', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Any Object")
    ax1.set_ylim([-5, 100])
    # ax3.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PBOT"], estimator=np.median, ci=None, label=produce_name(member, variant),
                         ax=ax2,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PBOT"]].groupby("stoch").describe()
            quart25 = data_stats[('PBOT', '25%')]
            quart75 = data_stats[('PBOT', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax2.lines[-1].set_linestyle("--")
            colour_count += 1
    ax2.set_title("Object 1 and Object 2")
    ax2.set_ylim([-5, 100])
    # ax4.set_xlabel("Stochasticity")

    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if any({loss in variant for loss in skip_loss_type}):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["UNBOTVAR"], estimator=np.median, ci=None,
                         label=produce_name(member, variant),
                         ax=ax3,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNBOTVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNBOTVAR', '25%')]
            quart75 = data_stats[('UNBOTVAR', '75%')]
            ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax3.lines[-1].set_linestyle("--")
            colour_count += 1
    ax3.set_title("Variance in Both Objects' Positions", y=1.065)

    # ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    plt.suptitle("Proportion of Controllers Not Moving...", x=0.40)

    plt.savefig(f"{save_dir}/pdf/pct_movedandposvar_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/pct_movedandposvar_{'_'.join(group)}.png")
    plt.close()
