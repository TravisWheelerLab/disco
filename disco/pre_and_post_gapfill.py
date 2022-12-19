import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import disco.cfg as cfg
from disco.eventwise_roc_plot import *

if __name__ == "__main__":

    root = "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/audio_2020/"
    # compare the different ensembles.
    test1 = "180101_0133S12-viz"
    test2 = "180101_0183S34D06-viz"
    test3 = "trial40_M57_F29_070220-viz"
    test_directories = [test1, test2, test3]

    label_files = [
        "180101_0133S12.csv",
        "180101_0183S34D06.csv",
        "trial40_M57_F29_070220.csv",
    ]

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ensemble_directories = glob(root + "*ensemble*")
    ensemble_directories = [
        d for d in ensemble_directories if "10_random_init" in os.path.basename(d)
    ]
    # randomly selected single model from the 10 member random init

    recalls = defaultdict(lambda: defaultdict(list))
    precisions = defaultdict(lambda: defaultdict(list))

    for ensemble_directory in ensemble_directories:
        print(ensemble_directory)
        for gapfill in [True, False]:

            for iqr_threshold in cfg.iqr_thresholds:
                trues = []
                preds = []
                hits_and_misses = []

                for test_directory, label_file in zip(test_directories, label_files):
                    # load the data
                    (
                        label_to_true,
                        label_to_predicted,
                        label_to_hit_and_miss,
                    ) = load_and_compute(
                        ensemble_directory=ensemble_directory,
                        test_directory=test_directory,
                        label_file=label_file,
                        iqr_threshold=iqr_threshold,
                        root=root,
                        single_model=False,
                        fill_gaps=gapfill,
                    )
                    trues.append(label_to_true)
                    preds.append(label_to_predicted)
                    hits_and_misses.append(label_to_hit_and_miss)

                # now, we can get precision and recall
                # recall is easy: use the hit and miss dictionary
                a_recall = compute_recall(hits_and_misses, class_name="A")
                b_recall = compute_recall(hits_and_misses, class_name="B")

                a_precision = compute_precision(hits_and_misses, preds, class_name="A")

                b_precision = compute_precision(hits_and_misses, preds, class_name="B")

                bs = os.path.basename(ensemble_directory)
                if gapfill:
                    bs = "post " + bs

                recalls[bs]["A"].append(a_recall)
                recalls[bs]["B"].append(b_recall)
                precisions[bs]["A"].append(a_precision)
                precisions[bs]["B"].append(b_precision)

    for ensemble_type in recalls:
        r = recalls[ensemble_type]
        p = precisions[ensemble_type]

        (line,) = ax[0].plot(r["A"], p["A"], "-")
        ax[1].plot(r["B"], p["B"], "-", color=line.get_color(), label=ensemble_type)

    ax[0].set_xlim(0, 1.01)
    ax[1].set_xlim(0, 1.01)
    ax[1].legend()

    ax[0].set_ylim(0, 1.01)
    ax[1].set_ylim(0, 1.01)

    ax[0].set_title("A chirp")
    ax[1].set_title("B chirp")

    ax[0].set_xlabel("recall")
    ax[1].set_xlabel("recall")

    ax[0].set_ylabel("precision")
    # ax[1].set_ylabel("precision")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_color("#808080")
        a.spines["left"].set_color("#808080")

        a.set_facecolor("none")
        a.grid(alpha=0.1, color="#808080")

    plt.show()
