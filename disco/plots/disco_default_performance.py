import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

import disco.figure_utils as fig_util

if __name__ == "__main__":

    fig, ax = plt.subplots()
    ensemble_directories = glob(fig_util.root + "*ensemble_10_random*")
    ensemble_directory = ensemble_directories[0]
    iqr_thresholds = np.logspace(-2, 0, num=50)[::-1]

    recalls = defaultdict(lambda: defaultdict(list))
    precisions = defaultdict(lambda: defaultdict(list))

    for iqr_threshold in fig_util.iqr_thresholds:
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

        # now, we can get precision and recall
        # recall is easy: use the hit and miss dictionary
        a_recall = fig_util.compute_recall(hits_and_misses, class_name="A")
        b_recall = fig_util.compute_recall(hits_and_misses, class_name="B")

        a_precision = fig_util.compute_precision(hits_and_misses, preds, class_name="A")

        b_precision = fig_util.compute_precision(hits_and_misses, preds, class_name="B")

        recalls[os.path.basename(ensemble_directory)]["A"].append(a_recall)
        recalls[os.path.basename(ensemble_directory)]["B"].append(b_recall)

        precisions[os.path.basename(ensemble_directory)]["A"].append(a_precision)
        precisions[os.path.basename(ensemble_directory)]["B"].append(b_precision)

    for ensemble_type in recalls:
        r = recalls[ensemble_type]
        p = precisions[ensemble_type]

        (line,) = ax.plot(r["A"], p["A"], "-", label="A chirp")
        ax.plot(r["B"], p["B"], "-", label="B chirp")

    ax.set_title("precision recall curve, default DISCO parameters")

    ax.set_xlim(0, 1.01)
    leg = ax.legend(
        loc="best",
        fancybox=True,
        framealpha=0.5,
        frameon=False,
        facecolor="white",
        fontsize=16,
    )

    ax.set_ylim(0.7, 1.01)

    ax.set_xlabel("recall", fontsize=16, color="black")
    ax.set_ylabel("precision", fontsize=16, color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    ax.spines["left"].set_color("#808080")

    ax.set_facecolor("none")
    ax.grid(alpha=0.1, color="#808080")

    plt.savefig(
        f"{os.path.join(os.environ['HOME'], 'disco_default_parameters.pdf')}",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )
    plt.show()
