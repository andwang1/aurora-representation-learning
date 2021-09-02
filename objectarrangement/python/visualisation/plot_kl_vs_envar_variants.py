import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_imagesd_exp1"
os.chdir(path)

plotting_groups = [
    # ["l2withsampling", "l2withsampling_beta0"],
    # ["l2", "l2withsampling"],
    # ["l2", "l2beta0"],
    # ["l2_nosampletrain", "l2"],
    # ["l2_nosampletrain", "l2beta0_nosampletrain"],
    # ["l2beta0", "l2"],
    # [""]
    # ["l2", "l2nosample"],
# ["l2nosampletrain"],
    ["l2", "l2beta0"]
]

colours = ["blue", "brown", "grey", "green", "purple", "red", "pink", "orange"]

# make legend bigger
plt.rc('legend', fontsize=35)
# make lines thicker
plt.rc('lines', linewidth=4, linestyle='-.')
# make font bigger
plt.rc('font', size=30)
sns.set_style("dark")

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    ax2 = ax1.twinx()
    lns = []
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if len(data["stoch"]) == 0:
                continue
            if "aurora" in variant or variant.startswith("ae") or "beta0" in member:
                continue
            ln1 =sns.lineplot(data["stoch"], data["KL"], estimator=np.median, ci=None, label=f"{member}-{variant}-KL",
                         ax=ax1,
                         color=colours[colour_count])
            # lns.extend(ln.get_lines())
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            df = pd.DataFrame(data)[["stoch", "KL"]]
            data_stats = df.groupby("stoch").describe()
            quart25 = data_stats[('KL', '25%')]
            quart75 = data_stats[('KL', '75%')]
            ax1.lines[-1].set_linestyle("--")
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            colour_count += 1

            ln2 = sns.lineplot(data["stoch"], data["ENVAR"] / 2, estimator=np.median, ci=None, label=f"{member}-{variant}-Variance",
                         ax=ax2,
                         color=colours[colour_count])
            # lns.extend(ln.get_lines())
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            df = pd.DataFrame(data)[["stoch", "ENVAR"]]
            df["ENVAR"] /= 2
            data_stats = df.groupby("stoch").describe()
            quart25 = data_stats[('ENVAR', '25%')]
            quart75 = data_stats[('ENVAR', '75%')]
            ax2.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            colour_count += 1

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    lns = ln1.get_lines() + ln2.get_lines()
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ax1.set_ylabel("KL")
    ax2.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title("KL Loss and Encoder Variance")
    plt.savefig(f"{save_dir}/pdf/envar_kl_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/envar_kl_{'_'.join(group)}.png")
    plt.close()
