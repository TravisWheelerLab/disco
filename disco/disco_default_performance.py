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

    fig, ax = plt.subplots()
    ensemble_directories = glob(root + "*ensemble_10_random*")
    ensemble_directory = ensemble_directories[0]
    iqr_thresholds = np.logspace(-2, 0, num=50)[::-1]

    recalls = defaultdict(lambda: defaultdict(list))
    precisions = defaultdict(lambda: defaultdict(list))

    for iqr_threshold in cfg.iqr_thresholds:
        trues = []
        preds = []
        hits_and_misses = []

        for test_directory, label_file in zip(test_directories, label_files):
            # load the data
            label_to_true, label_to_predicted, label_to_hit_and_miss = load_and_compute(
                ensemble_directory=ensemble_directory,
                test_directory=test_directory,
                label_file=label_file,
                iqr_threshold=iqr_threshold,
                root=root,
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
    ax.legend()
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    ax.spines["left"].set_color("#808080")

    ax.set_facecolor("none")
    ax.grid(alpha=0.1, color="#808080")

    plt.show()
