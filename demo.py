# ---------------------------------------------------------------------------- #
#          Module to plot and demo results related to HMM and HsMM
# ---------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}


def hmm_hsmm_comp(data, legend_data, ground_truth=None, over_width=15, fig_height=14, save_pdf=False,
                  title_set=None):
    """
    Function: plot the comparasion results between HMM and HsMM, with possible FO, FB, VB algorithms
    Note: some of the property, such as tick font size, might need to be adjusted in typical application
    :param data: format, list of list, the inner list contain results from HMM and HsMM
    :param legend_data: same format as data but are strings, which specify the algorithms
    :param ground_truth: format1 style, list of list, the inner list is two items based
                         specify the state index and its duration
    :param over_width: the excess horizontally to accomodate the legends to avoid overlapping with data
    :param fig_height: set the figure height
    :param title_set: allow for specify title name if needed, by default there is no figure title
    """
    style = ['m--s', 'r--o', 'c--s', 'm--o']

    plot_num = len(data)

    fig = plt.figure(facecolor='white', figsize=(12, fig_height), dpi=80)
    xmajorLocator = MultipleLocator(10)
    ymajorLocator = MultipleLocator(1)
    for k in range(plot_num):
        if k == 0 and title_set is not None:
            plt.title(title_set)
        if k == 0:
            ax1 = fig.add_subplot(plot_num, 1, k + 1)
            ax = ax1
        else:
            ax = fig.add_subplot(plot_num, 1, k + 1, sharex=ax1)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if ground_truth is not None:  # the ground truth is provided
            space_s = 0.2
            for i in range(len(ground_truth)):
                # generate filled rectangle
                if i == 0:
                    x = [0, ground_truth[i][1], ground_truth[i][1], 0, 0]
                else:
                    x = [ground_truth[i - 1][1], ground_truth[i][1],
                         ground_truth[i][1], ground_truth[i - 1][1], ground_truth[i - 1][1]]
                y = [ground_truth[i][0] - space_s, ground_truth[i][0] - space_s, ground_truth[i][0] + space_s,
                     ground_truth[i][0] + space_s, ground_truth[i][0] - space_s]
                ax.fill(x, y, facecolor='k', edgecolor='r', alpha=0.35)
        for m in range(len(data[0])):
            ax.plot(data[k][m], style[m], markersize=4)
        ax.grid(axis="y")
        ax.set_xlim([-1, len(data[0][0]) + over_width])
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.legend(legend_data[k], prop=font1, loc=4)
        ax.set_ylabel('State index', fontdict=font2)
    ax.set_xlabel('Time index', fontdict=font2)

    if save_pdf:
        plt.savefig('HsMM-FR_VS_Fingerprint.pdf', format='pdf')
