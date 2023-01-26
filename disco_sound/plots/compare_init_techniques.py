import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

import disco_sound.plots.figure_utils as fig_util

plt.style.use("ggplot")


if __name__ == "__main__":

    ensemble_name_map = {
        "ensemble_10_random_init": "random init.",
        "ensemble_10_bootstrap": "bootstrap",
    }

    sorted_keys = [
        "ensemble_2_random_init",
        "ensemble_2_bootstrap",
        "ensemble_1",
        "ensemble_2",
        "ensemble_3",
        "ensemble_10_random_init",
        "ensemble_10_bootstrap",
        "ensemble_30_random_init",
        "ensemble_30_bootstrap",
    ]

    ensemble_name_map = {
        "ensemble_2_random_init": "2 member, random init.",
        "ensemble_2_bootstrap": "2 member, bootstrap",
        "ensemble_10_random_init": "10 member, random init.",
        "ensemble_10_bootstrap": "10 member, bootstrap",
        "ensemble_30_random_init": "30 member, random init.",
        "ensemble_30_bootstrap": "30 member, bootstrap",
    }

    fig, ax = plt.subplots(ncols=2, figsize=(12.2, 5), sharex=True, sharey=True)
    ensemble_directories = glob(fig_util.root + "/audio_2020_ensembles/*ensemble*")
    # add more:
    more_random_ensembles = glob(
        f"{fig_util.root}/compare_init_techniques/more_ensembles/random_init/UNet1D/*ensemble*"
    )
    more_bootstrap_ensembles = glob(
        f"{fig_util.root}/compare_init_techniques/more_ensembles/bootstrap/UNet1D/*ensemble*"
    )

    ensemble_directories = [
        d for d in ensemble_directories if "10" in os.path.basename(d)
    ]

    ensemble_directories.extend(more_bootstrap_ensembles)
    ensemble_directories.extend(more_random_ensembles)

    recalls = defaultdict(lambda: defaultdict(list))
    precisions = defaultdict(lambda: defaultdict(list))

    for ensemble_directory in ensemble_directories:
        print(ensemble_directory)

        # if "ensemble_2" not in ensemble_directory:
        #     continue

        for iqr_threshold in fig_util.iqr_thresholds[::-1]:
            trues = []
            preds = []
            hits_and_misses = []

            for test_directory, label_file in zip(
                fig_util.test_directories, fig_util.label_files
            ):
                # load the data
                (
                    label_to_true,
                    label_to_predicted,
                    label_to_hit_and_miss,
                ) = fig_util.load_and_compute(
                    ensemble_directory=ensemble_directory,
                    test_directory=test_directory,
                    label_file=label_file,
                    iqr_threshold=iqr_threshold,
                    root=fig_util.root,
                )
                trues.append(label_to_true)
                preds.append(label_to_predicted)
                hits_and_misses.append(label_to_hit_and_miss)

            # label to predicted contains a bunch of small hits.
            # aha! that's why we are getting weird metrics.
            # because label_to_predicted is overestimating
            # at really low IQR thresholds we're getting
            # underestimates of precision: the denominator
            # is TOO large. this means that the counts of predictions is a huge overestimate
            # why?
            # now, we can get precision and recall
            # recall is easy: use the hit and miss dictionary

            a_recall = fig_util.compute_recall(hits_and_misses, class_name="A")
            b_recall = fig_util.compute_recall(hits_and_misses, class_name="B")
            a_precision = fig_util.compute_precision(
                hits_and_misses, preds, class_name="A"
            )
            b_precision = fig_util.compute_precision(
                hits_and_misses, preds, class_name="B"
            )

            if ensemble_directory[-1] in ("1", "2", "3"):
                if "bootstrap" in ensemble_directory:
                    ensemble_name = f"bootstrap_{ensemble_directory[-1]}"
                else:
                    ensemble_name = f"random_{ensemble_directory[-1]}"
            else:
                ensemble_name = os.path.basename(ensemble_directory)

            recalls[ensemble_name]["A"].append(a_recall)
            recalls[ensemble_name]["B"].append(b_recall)

            precisions[ensemble_name]["A"].append(a_precision)
            precisions[ensemble_name]["B"].append(b_precision)

    random_a = []
    random_b = []
    bootstrap_a = []
    bootstrap_b = []

    for ensemble_type in precisions.keys():

        r = recalls[ensemble_type]
        p = precisions[ensemble_type]

        if "random" in ensemble_type:
            random_a.append((p["A"], r["A"]))
            random_b.append((p["B"], r["B"]))
        else:
            bootstrap_a.append((p["A"], r["A"]))
            bootstrap_b.append((p["B"], r["B"]))

        # (line,) = ax[0].plot(r["A"], p["A"], "-", label=ensemble_name_map[ensemble_type])
        if "bootstrap" in ensemble_type:
            color = "r"
        else:
            color = "b"

        ax[0].plot(r["A"], p["A"], "-", alpha=0.2, color=color)
        ax[1].plot(r["B"], p["B"], "-", color=color, alpha=0.2)

    random_a = np.median(np.asarray(random_a), axis=0)
    random_b = np.median(np.asarray(random_b), axis=0)

    bootstrap_a = np.median(np.asarray(bootstrap_a), axis=0)
    bootstrap_b = np.median(np.asarray(bootstrap_b), axis=0)

    ax[0].plot(random_a[1], random_a[0], label="random median", color="b")
    ax[0].plot(bootstrap_a[1], bootstrap_a[0], label="bootstrap median", color="r")

    ax[1].plot(random_b[1], random_b[0], label="random median", color="b")
    ax[1].plot(bootstrap_b[1], bootstrap_b[0], label="bootstrap median", color="r")

    ax[0].set_xlim(0, 1.01)
    ax[1].set_xlim(0, 1.01)

    ax[0].legend(
        fancybox=True, framealpha=0.5, frameon=False, facecolor="white", fontsize=17
    )

    ax[0].set_title("A chirp", color="black", fontsize=14)
    ax[1].set_title("B chirp", color="black", fontsize=14)

    # fig.text(y=0.047, x=0.51, s="recall", ha="center", fontsize=16, color="black")

    ax[0].set_ylabel("precision", color="black", fontsize=16)
    # from https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots

    subplot = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("recall", color="black", fontsize=16)
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.spines["bottom"].set_visible(False)
    subplot.spines["left"].set_visible(False)

    subplot.set_facecolor("none")
    subplot.grid(alpha=0.0, color="#808080")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_color("#808080")
        a.spines["left"].set_color("#808080")

        a.set_facecolor("none")
        a.grid(alpha=0.1, color="#808080")

    plt.savefig(
        f"{os.path.join(os.environ['HOME'], 'compare_init_techniques.pdf')}",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )

    plt.show()
