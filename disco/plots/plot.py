import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


class Plotter:
    def __init__(self, csv_dirpath):
        sns.set_style("darkgrid", {"axes.facecolor": ".95"})
        self.csv_dir = csv_dirpath
        self.files = glob(os.path.join(self.csv_dir, "*.csv"))
        self.dataframes = {}
        for file in self.files:
            filename = os.path.basename(file).split(".")[0]
            dataframe = pd.read_csv(file, index_col=0)
            dataframe.loc["iqr threshold"] = [float(colname.split('/')[0]) for colname in dataframe.columns]
            dataframe.loc["voting threshold"] = [int(colname.split('/')[1]) for colname in dataframe.columns]
            if "event_wise" in filename:
                dataframe.loc["event proportion"] = [int(colname.split('/')[2]) for colname in dataframe.columns]
            dataframe = dataframe.T

            dataframe["accuracy"] = pd.to_numeric(dataframe["accuracy"])
            dataframe["recall, A chirp"] = pd.to_numeric(dataframe["recall, A chirp"])
            dataframe["recall, B chirp"] = pd.to_numeric(dataframe["recall, B chirp"])

            if "point_wise" in filename:
                dataframe["precision, A chirp"] = pd.to_numeric(dataframe["precision, A chirp"])
                dataframe["precision, B chirp"] = pd.to_numeric(dataframe["precision, B chirp"])
                dataframe["IoU, A chirp"] = pd.to_numeric(dataframe["IoU, A chirp"])
                dataframe["IoU, B chirp"] = pd.to_numeric(dataframe["IoU, B chirp"])

            dataframe["members"] = int(filename.split("_")[2])
            dataframe["type"] = filename.split("_")[-1]
            dataframe.loc[dataframe["type"] == "init", "type"] = "random init"
            self.dataframes[filename] = dataframe

        self.merged_event_wise_dfs = pd.concat([val for key, val in self.dataframes.items() if "event" in key],
                                               ignore_index=True).sort_values(by="iqr threshold",
                                                                              ascending=False).sort_values(
            by=["type", "members"])
        self.merged_point_wise_dfs = pd.concat([val for key, val in self.dataframes.items() if "point" in key],
                                               ignore_index=True).sort_values(by="iqr threshold",
                                                                              ascending=False).sort_values(
            by=["type", "members"])

    def plot_event_wise(self, type, outcome_variable, ylims=(0, 1), palette="crest"):
        if type == "both":
            base_df = self.merged_event_wise_dfs
        else:
            base_df = self.merged_event_wise_dfs[self.merged_event_wise_dfs["type"] == type]
        no_iqr_thresholding = base_df[base_df["iqr threshold"] == 1]
        no_thresholding = no_iqr_thresholding[no_iqr_thresholding["voting threshold"] == 0]

        if type == "both":
            line_plot = sns.relplot(data=no_thresholding,
                                    x="event proportion",
                                    y=outcome_variable,
                                    col="type",
                                    hue="members",
                                    palette=palette,
                                    marker="o",
                                    legend="full",
                                    kind="line")

            models = no_thresholding["type"].unique()
            for i, ax in enumerate(line_plot.axes.flat):
                ax.set_title(models[i])
            line_plot.figure.suptitle(outcome_variable + " by model type (no uncertainty thresholding)",
                                      size=15)
            line_plot.figure.subplots_adjust(top=0.87)

        else:
            line_plot = sns.lineplot(data=no_thresholding,
                                     x="event proportion",
                                     y=outcome_variable,
                                     hue="members",
                                     palette=palette,
                                     marker="o")
            line_plot.set(title=type)

        line_plot.set(ylim=ylims,
                      xlabel="minimum proportion event correct")
        sns.set(rc={'figure.figsize': (5, 6)})
        plt.show(line_plot)

    def plot_point_wise(self, type, outcome_variable, predictive_variable, ylims=(0, 1), palette="crest"):
        if type == "both":
            base_df = self.merged_point_wise_dfs
        else:
            base_df = self.merged_point_wise_dfs[self.merged_point_wise_dfs["type"] == type]
        if predictive_variable == "iqr threshold":
            base_df = base_df[base_df["voting threshold"] == 0]
        else:
            print("No graphs for other predictive variables yet.")
            pass

        if type == "both":
            line_plot = sns.relplot(data=base_df,
                                    x=predictive_variable,
                                    y=outcome_variable,
                                    col="type",
                                    hue="members",
                                    palette=palette,
                                    marker="o",
                                    legend="full",
                                    kind="line")

            models = base_df["type"].unique()
            for i, ax in enumerate(line_plot.axes.flat):
                ax.set_title(models[i])
            line_plot.figure.suptitle("pointwise " + outcome_variable + " by model type with uncertainty thresholding",
                                      size=15)
            line_plot.figure.subplots_adjust(top=0.87)

        else:
            line_plot = sns.lineplot(data=base_df,
                                     x=predictive_variable,
                                     y=outcome_variable,
                                     hue="members",
                                     palette=palette,
                                     marker="o")
            line_plot.set(title=type)
            line_plot.set_xlim(line_plot.get_xlim()[::-1])

        line_plot.set(ylim=ylims,
                      xlabel=predictive_variable)
        sns.set(rc={'figure.figsize': (5, 6)})
        plt.show(line_plot)

    def plot_voting(self, type, outcome_variable, members, ylims=(0, 1), palette="crest"):
        if type == "both":
            base_df = self.merged_point_wise_dfs
        else:
            base_df = self.merged_point_wise_dfs[self.merged_point_wise_dfs["type"] == type]
        base_df = base_df[base_df["iqr threshold"] == 1]
        base_df = base_df[base_df["members"] == members]

        if type == "both":
            line_plot = sns.relplot(data=base_df,
                                    x="voting threshold",
                                    y=outcome_variable,
                                    col="type",
                                    palette=palette,
                                    marker="o",
                                    legend="full",
                                    kind="line")

            models = base_df["type"].unique()
            for i, ax in enumerate(line_plot.axes.flat):
                ax.set_title(models[i])
            line_plot.figure.suptitle(
                "pointwise " + outcome_variable + " by voting threshold, " + str(members) + " members",
                size=15)
            line_plot.figure.subplots_adjust(top=0.87)

        else:
            line_plot = sns.lineplot(data=base_df,
                                     x="voting threshold",
                                     y=outcome_variable,
                                     palette=palette,
                                     marker="o")
            line_plot.set(title=type)
            line_plot.set_xlim(line_plot.get_xlim()[::-1])

        line_plot.set(ylim=ylims,
                      xlabel="voting threshold")
        sns.set(rc={'figure.figsize': (5, 6)})
        plt.show(line_plot)


plotter = Plotter(csv_dirpath="./")

outcome_variables = ["accuracy", "recall, A chirp", "recall, B chirp", "precision, A chirp", "precision, B chirp",
                     "IoU, A chirp", "IoU, B chirp"]
outcome = 4
plotter.plot_voting(type="both", outcome_variable=outcome_variables[outcome], members=10, ylims=(0.9, 1.001))