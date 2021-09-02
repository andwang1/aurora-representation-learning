import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd
from visualisation.produce_name import produce_name








skip_loss_type = {
    # "false"
}


# make legend bigger
plt.rc('legend', fontsize=20)
# make lines thicker
plt.rc('lines', linewidth=4, linestyle='-.')
# make font bigger
plt.rc('font', size=28)
# sns.set_style("dark")

path1 = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/asd/BD2"
path2 = "/media/andwang1/SAMSUNG/MSC_INDIV/ICLR/15k/isd"


# posvar
f = plt.figure(figsize=(20, 10))
f.text(0.5, 0.11, 'Environment Stochasticity', ha='center')
spec = f.add_gridspec(1, 2)
ax1 = f.add_subplot(spec[0, 0])
ax2 = f.add_subplot(spec[0, 1])

os.chdir(path1)
plotting_groups = [
    ["AURORA", "AURORA_smaller", "beta0", "largerbeta0",  "sample", "best",  "largerbeta1"]
]
colours = sns.color_palette()

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)

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
                         ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "UNBOTVAR"]].groupby("stoch").describe()
            quart25 = data_stats[('UNBOTVAR', '25%')]
            quart75 = data_stats[('UNBOTVAR', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Object Arrangement Task")
    ax1.set_ylabel("Variance")


os.chdir(path2)
plotting_groups = [
    ["AURORA", "beta0",  "sample", "best"]
]
c = sns.color_palette()
colours = [c[0], c[2], c[4], c[5], c[6], c[7]]
# colours = ["brown","grey", "green", "blue", "purple", "red", "pink", "orange"]

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)

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
    ax2.set_title("Air-Hockey Task")
plt.suptitle("Variance in Positions of Objects of Interest (Larger is Better)")
ax1.get_legend().remove()
ax2.get_legend().remove()

# ax5.set_xlabel("Environment Stochasticity")

# shrink boxes to make space for legend
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])

# Put a legend below the plot
leg = ax1.legend(loc='upper center', bbox_to_anchor=(1.05, -0.11), fancybox=True,
           ncol=4)
leg.get_frame().set_edgecolor('black')

plt.savefig(f"/home/andwang1/Pictures/ICLR/posvar.pdf")
plt.savefig(f"/home/andwang1/Pictures/ICLR/posvar.png")
plt.close()



# pctmove

# make legend bigger
plt.rc('legend', fontsize=28)

f = plt.figure(figsize=(30, 10))
f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
spec = f.add_gridspec(1, 3)
ax1 = f.add_subplot(spec[0, 0])
ax2 = f.add_subplot(spec[0, 1])
ax3 = f.add_subplot(spec[0, 2])

os.chdir(path1)
plotting_groups = [
    ["AURORA", "AURORA_smaller", "beta0", "largerbeta0",  "sample", "best",  "largerbeta1"]
]
colours = sns.color_palette()

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)

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
            print("ASD", data_stats)
            quart25 = data_stats[('PEIT', '25%')]
            quart75 = data_stats[('PEIT', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Any Object (Arrangement Task)")
    ax1.set_ylim([-5, 100])
    ax1.set_ylabel("%")
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
    ax2.set_title("Both Object 1 and Object 2")
    ax2.set_ylim([-5, 100])
    # ax4.set_xlabel("Stochasticity")
os.chdir(path2)
plotting_groups = [
    ["AURORA", "beta0",  "sample", "best"]
]
c = sns.color_palette()
colours = [c[0], c[2], c[4], c[5], c[6], c[7]]

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)

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
                         ax=ax3,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PCT"]].groupby("stoch").describe()
            print(data_stats)
            quart25 = data_stats[('PCT', '25%')]
            quart75 = data_stats[('PCT', '75%')]
            ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax3.lines[-1].set_linestyle("--")
            colour_count += 1
    ax3.set_ylim([-5, 100])
    ax3.set_title("Non-Random Air-Hockey Puck")
plt.suptitle("Proportion of Controllers Failing to Move (Lower is Better)")

ax2.get_legend().remove()
ax3.get_legend().remove()

ax2.axes.yaxis.set_visible(False)
ax3.yaxis.set_ticks_position("right")

plt.subplots_adjust(wspace=0.02)
plt.savefig(f"/home/andwang1/Pictures/ICLR/pctmove.pdf")
plt.savefig(f"/home/andwang1/Pictures/ICLR/pctmove.png")

plt.close()


    #
    # f = plt.figure(figsize=(30, 10))
    # f.text(0.5, 0.02, 'Environment Stochasticity', ha='center')
    # spec = f.add_gridspec(1, 3)
    # ax1 = f.add_subplot(spec[0, 0])
    # ax2 = f.add_subplot(spec[0, 1])
    # ax3 = f.add_subplot(spec[0, 2])
    #
    # colour_count = 0
    # for i, member in enumerate(group):
    #     with open(f"{member}/pct_moved_data.pk", "rb") as f:
    #         log_data = pk.load(f)
    #
    #     for variant, data in log_data.items():
    #         if len(data["stoch"]) == 0:
    #             continue
    #         if any({loss in variant for loss in skip_loss_type}):
    #             continue
    #         variant_name = variant if not variant.startswith("ae") else "ae"
    #         sns.lineplot(data["stoch"], data["PEIT"], estimator=np.median, ci=None, label=produce_name(member, variant),
    #                      ax=ax1,
    #                      color=colours[colour_count])
    #         data_stats = pd.DataFrame(data)[["stoch", "PEIT"]].groupby("stoch").describe()
    #         quart25 = data_stats[('PEIT', '25%')]
    #         quart75 = data_stats[('PEIT', '75%')]
    #         ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
    #         if i == 0 and len(group) > 1:
    #             ax1.lines[-1].set_linestyle("--")
    #         colour_count += 1
    # ax1.set_title("Any Object")
    # ax1.set_ylim([-5, 100])
    # # ax3.set_xlabel("Stochasticity")
    #
    # colour_count = 0
    # for i, member in enumerate(group):
    #     with open(f"{member}/pct_moved_data.pk", "rb") as f:
    #         log_data = pk.load(f)
    #
    #     for variant, data in log_data.items():
    #         if len(data["stoch"]) == 0:
    #             continue
    #         if any({loss in variant for loss in skip_loss_type}):
    #             continue
    #         variant_name = variant if not variant.startswith("ae") else "ae"
    #         sns.lineplot(data["stoch"], data["PBOT"], estimator=np.median, ci=None, label=produce_name(member, variant),
    #                      ax=ax2,
    #                      color=colours[colour_count])
    #         data_stats = pd.DataFrame(data)[["stoch", "PBOT"]].groupby("stoch").describe()
    #         quart25 = data_stats[('PBOT', '25%')]
    #         quart75 = data_stats[('PBOT', '75%')]
    #         ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
    #         if i == 0 and len(group) > 1:
    #             ax2.lines[-1].set_linestyle("--")
    #         colour_count += 1
    # ax2.set_title("Object 1 and Object 2")
    # ax2.set_ylim([-5, 100])
    # # ax4.set_xlabel("Stochasticity")
    #
    # colour_count = 0
    # for i, member in enumerate(group):
    #     with open(f"{member}/posvar_data.pk", "rb") as f:
    #         log_data = pk.load(f)
    #
    #     for variant, data in log_data.items():
    #         if len(data["stoch"]) == 0:
    #             continue
    #         if any({loss in variant for loss in skip_loss_type}):
    #             continue
    #         variant_name = variant if not variant.startswith("ae") else "ae"
    #         sns.lineplot(data["stoch"], data["UNBOTVAR"], estimator=np.median, ci=None,
    #                      label=produce_name(member, variant),
    #                      ax=ax3,
    #                      color=colours[colour_count])
    #         data_stats = pd.DataFrame(data)[["stoch", "UNBOTVAR"]].groupby("stoch").describe()
    #         quart25 = data_stats[('UNBOTVAR', '25%')]
    #         quart75 = data_stats[('UNBOTVAR', '75%')]
    #         ax3.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
    #         if i == 0 and len(group) > 1:
    #             ax3.lines[-1].set_linestyle("--")
    #         colour_count += 1
    # ax3.set_title("Variance in Both Objects' Positions", y=1.065)
    #
    # # ax1.get_legend().remove()
    # ax2.get_legend().remove()
    # ax3.get_legend().remove()
    # plt.suptitle("Proportion of Controllers Not Moving...", x=0.40)
    #
    # plt.savefig(f"{save_dir}/pdf/pct_movedandposvar_{'_'.join(group)}.pdf")
    # plt.savefig(f"{save_dir}/pct_movedandposvar_{'_'.join(group)}.png")
    # plt.close()
