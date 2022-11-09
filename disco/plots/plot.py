import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

outcome_variables = [
    "accuracy",
    "recall, A chirp",
    "recall, B chirp",
    "precision, A chirp",
    "precision, B chirp",
    "IoU, A chirp",
    "IoU, B chirp",
]


class Plotter:
    def __init__(self, csv_dirpath):
        sns.set_style("darkgrid", {"axes.facecolor": ".95"})
        csv_dir = csv_dirpath
        files = glob(os.path.join(csv_dir, "*.csv"))
        dataframes = {}

        for file in files:
            filename = os.path.basename(file).split(".")[0]
            dataframe = pd.read_csv(file, index_col=0)
            dataframe.loc["iqr threshold"] = [
                float(colname.split("/")[0]) for colname in dataframe.columns
            ]
            dataframe.loc["voting threshold"] = [
                int(colname.split("/")[1]) for colname in dataframe.columns
            ]
            if "event_wise" in filename:
                dataframe.loc["event proportion"] = [
                    int(colname.split("/")[2]) for colname in dataframe.columns
                ]
            dataframe = dataframe.T

            dataframe["accuracy"] = pd.to_numeric(dataframe["accuracy"])
            dataframe["recall, A chirp"] = pd.to_numeric(dataframe["recall, A chirp"])
            dataframe["recall, B chirp"] = pd.to_numeric(dataframe["recall, B chirp"])

            if "point_wise" in filename:
                dataframe["precision, A chirp"] = pd.to_numeric(
                    dataframe["precision, A chirp"]
                )
                dataframe["precision, B chirp"] = pd.to_numeric(
                    dataframe["precision, B chirp"]
                )
                dataframe["IoU, A chirp"] = pd.to_numeric(dataframe["IoU, A chirp"])
                dataframe["IoU, B chirp"] = pd.to_numeric(dataframe["IoU, B chirp"])

            dataframe["members"] = int(filename.split("_")[2])
            dataframe["type"] = filename.split("_")[-1]
            dataframe.loc[dataframe["type"] == "init", "type"] = "random init"
            dataframes[filename] = dataframe

        self.merged_event_wise_dfs = (
            pd.concat(
                [val for key, val in dataframes.items() if "event" in key],
                ignore_index=True,
            )
            .sort_values(by="iqr threshold", ascending=False)
            .sort_values(by=["type", "members"])
        )
        self.merged_point_wise_dfs = (
            pd.concat(
                [val for key, val in dataframes.items() if "point" in key],
                ignore_index=True,
            )
            .sort_values(by="iqr threshold", ascending=False)
            .sort_values(by=["type", "members"])
        )

    def plot_event_wise(
        self, type, outcome_variable, ax, ylims=(0, 1), palette="crest", title=None
    ):
        if type == "both":
            base_df = self.merged_event_wise_dfs
        else:
            base_df = self.merged_event_wise_dfs[
                self.merged_event_wise_dfs["type"] == type
            ]
        no_iqr_thresholding = base_df[base_df["iqr threshold"] == 1]
        no_thresholding = no_iqr_thresholding[
            no_iqr_thresholding["voting threshold"] == 0
        ]

        if type == "both":
            line_plot = sns.relplot(
                data=no_thresholding,
                x="event proportion",
                y=outcome_variable,
                col="type",
                hue="members",
                palette=palette,
                marker="o",
                legend="full",
                kind="line",
            )

            models = no_thresholding["type"].unique()
            for i, ax in enumerate(line_plot.axes.flat):
                ax.set_title(models[i])
            line_plot.figure.suptitle(
                outcome_variable + " by model type (no uncertainty thresholding)",
                size=15,
            )
            line_plot.figure.subplots_adjust(top=0.87)

        else:
            line_plot = sns.lineplot(
                data=no_thresholding,
                x="event proportion",
                y=outcome_variable,
                hue="members",
                palette=palette,
                marker="o",
                ax=ax,
            )
        if title is None:
            title = outcome_variable
        if "chirp" in title:
            plt.legend([], [], frameon=False)

        line_plot.set(
            ylim=ylims, xlabel="minimum proportion event correct", title=title
        )

        sns.set(rc={"figure.figsize": (5, 6)})
        return line_plot

    def plot_point_wise(
        self, type, outcome_variable, predictive_variable, ylims=(0, 1), palette="crest"
    ):
        if type == "both":
            base_df = self.merged_point_wise_dfs
        else:
            base_df = self.merged_point_wise_dfs[
                self.merged_point_wise_dfs["type"] == type
            ]

        if predictive_variable == "iqr threshold":
            base_df = base_df[base_df["voting threshold"] == 0]
        else:
            print("No graphs available for non-iqr predictive variables.")

        if type == "both":
            line_plot = sns.relplot(
                data=base_df,
                x=predictive_variable,
                y=outcome_variable,
                col="type",
                hue="members",
                palette=palette,
                marker="o",
                legend="full",
                kind="line",
            )

            models = base_df["type"].unique()
            for i, ax in enumerate(line_plot.axes.flat):
                ax.set_title(models[i])
            line_plot.figure.suptitle(
                "pointwise "
                + outcome_variable
                + " by model type with uncertainty thresholding",
                size=15,
            )
            line_plot.figure.subplots_adjust(top=0.87)

        else:
            line_plot = sns.lineplot(
                data=base_df,
                x=predictive_variable,
                y=outcome_variable,
                hue="members",
                palette=palette,
                marker="o",
            )
            line_plot.set(title=type)
            line_plot.set_xlim(line_plot.get_xlim()[::-1])

        line_plot.set(ylim=ylims, xlabel=predictive_variable)
        sns.set(rc={"figure.figsize": (5, 6)})
        return line_plot

    def plot_iqr_thresholding(self, type, members, palette="crest"):
        base_df = self.merged_point_wise_dfs[self.merged_point_wise_dfs["type"] == type]
        base_df = base_df[base_df["voting threshold"] == 0]
        base_df = base_df[base_df["members"] == members]
        # base_df = base_df.rename({'accuracy': 'accuracy, all classes'}, axis=1)
        base_df = base_df.drop(
            [
                "accuracy",
                "IoU, A chirp",
                "IoU, B chirp",
                "confusion_matrix",
                "confusion_matrix_nonnorm",
                "voting threshold",
                "members",
                "type",
            ],
            axis=1,
        )
        base_df = base_df.melt(id_vars=["iqr threshold"], var_name="metric")
        line_plot = sns.lineplot(
            data=base_df,
            x="iqr threshold",
            y="value",
            hue="metric",
            palette=palette,
            marker="o",
        )
        line_plot.set_xlim(line_plot.get_xlim()[::-1])
        line_plot.set(
            ylim=(0.5, 1.01), title="pointwise accuracy metrics by IQR threshold"
        )


if __name__ == "__main__":

    for i in [0, 20, 40, 80, 160, 320]:

        plotter = Plotter(csv_dirpath=f"disco/resources/disco_accuracy_csvs/snr_{i}")
        plotter.merged_event_wise_dfs = plotter.merged_event_wise_dfs[
            plotter.merged_event_wise_dfs["event proportion"] != 95
        ]

        outcome = 0
        # plotter.plot_point_wise(type="bootstrap", outcome_variable=outcome_variables[outcome], ylims=(0.3, 1.01), predictive_variable="iqr threshold")
        plotter.plot_event_wise(
            type="bootstrap",
            outcome_variable=outcome_variables[outcome],
            ylims=(0.3, 1.01),
        )
        # plotter.plot_event_wise(type="bootstrap", outcome_variable=outcome_variables[outcome], ylims=(0.3, 1.01))
        # plotter.plot_point_wise(type="bootstrap", outcome_variable=outcome_variables[outcome], ylims=(0.3, 1.01), predictive_variable="iqr threshold")
        # plotter.plot_iqr_thresholding(type="random init", members=10, palette="colorblind")
        plt.legend(title="ensemble_members")
        plt.show()
