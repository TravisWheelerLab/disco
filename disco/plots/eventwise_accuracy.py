import matplotlib.pyplot as plt

from disco.plots.plot import Plotter, outcome_variables

"""
This script is intended to reproduce the event-wise accuracy figure in the manuscript.
"""
# use a signal-to-noise ratio of 0, since this indicates unnoised data.
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
plotter = Plotter(csv_dirpath=f"disco/resources/disco_accuracy_csvs/snr_0")
outcome = 0  # accuracy
overall = plotter.plot_event_wise(
    type="random init",
    outcome_variable=outcome_variables[outcome],
    ax=ax[0],
    ylims=(0.3, 1.01),
)
outcome = 1  # a chirp recall, which is the same as accuracy
a_chirp = plotter.plot_event_wise(
    type="random init",
    outcome_variable=outcome_variables[outcome],
    ax=ax[1],
    ylims=(0.3, 1.01),
)
outcome = 2  # a chirp recall, which is the same as accuracy
b_chirp = plotter.plot_event_wise(
    type="random init",
    outcome_variable=outcome_variables[outcome],
    ax=ax[2],
    ylims=(0.3, 1.01),
)

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_color("#808080")
    a.spines["left"].set_color("#808080")

ax[1].get_legend().remove()
ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)

ax[0].set_xlim(25, 100)
ax[1].set_xlim(25, 100)
ax[2].set_xlim(25, 100)

ax[0].set_ylabel("accuracy", fontsize=15)
ax[1].set_xlabel("minimum proportion of event classified correctly", fontsize=15)
alpha = 0.3
plt.suptitle("event-wise accuracy")

ax[0].grid(alpha=alpha)
ax[1].grid(alpha=alpha)
ax[2].grid(alpha=alpha)

ax[0].set_title("A+B chirp")
ax[1].set_title("A chirp")
ax[2].set_title("B chirp")

plt.show()
