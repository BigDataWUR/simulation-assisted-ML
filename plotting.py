import matplotlib
import os
import numpy as np
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def make_jointplot(predictions_df, set_name, output, model_type, location):
    """Creates plots of predictions vs actual values and saves them. 

    Parameters
    ----------
    predictions_df: pandas.core.frame.DataFrame, the predictions of the estimator for the specified training/test set
    set_name: str, indicator for the name of the figure to show if it's training or test set
    output: str, the name of the target variable column in 'predictions_df'
    model_type: str, indicator for the name of the figure (could be local, regional, global)
    location: str, the of the location for which the results refer to

    Returns
    -------
    Returns None.
    """

    plt.clf()
    axis_min = 0
    axis_max = 40

    plot = sns.jointplot(x=output, y='Predicted_'+output,
                         data=predictions_df, kind='hex')
    plot.ax_joint.plot([axis_min, axis_max], [
                       axis_min, axis_max], 'black', linewidth=1)
    plot.ax_marg_x.set_xlim([axis_min, axis_max])
    plot.ax_marg_y.set_ylim([axis_min, axis_max])
    sns.regplot(x=output, y='Predicted_'+output, data=predictions_df,
                ax=plot.ax_joint, scatter=False, color="red")
    plot.set_axis_labels('Ground truth', 'Predictions')
    plt.savefig(location + '_' + model_type +
                '_densityplot_' + set_name + '.png')
    plt.close()


def make_boxplots(known_unknown, totalDF, x_axis, irrigation=None):
    """Creates plots of monthly and yearly residuals and saves them. If irrigation is not provided the plots contain both cases.

    Parameters
    ----------
    known_unknown: str, one of 'known' 'unknown'
    totalDF: pandas.core.frame.DataFrame, contains the predictions for all models, model types, and location types and also the correspodning simulation parameters
    x_axis: str, one of 'FertMonth' 'Year'
    irrigation: int, one of '0' '1' 

    Returns
    -------
    Returns None. 
    """

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(18, 10))
    if x_axis == 'FertMonth':
        palette = 'Set3'
    else:
        palette = 'Set2'

    for index, (ax, location) in enumerate(zip(axes.flatten(), ['Waiotu', 'Ruakura', 'Wairoa', 'Marton', 'Mahana', 'Kokatahi', 'Lincoln', 'Wyndham'])):

        if irrigation is None:
            p = sns.boxplot(ax=ax, data=totalDF[(totalDF['Test type'] == known_unknown) & (
                totalDF['Location'] == location)], x=x_axis, y="Residual", hue='Metamodel type', palette=palette, showfliers=False)
        else:
            p = sns.boxplot(ax=ax, data=totalDF[(totalDF['Test type'] == known_unknown) & (totalDF['Location'] == location) & (
                totalDF['Irrigation'] == irrigation)], x=x_axis, y="Residual", hue='Metamodel type', palette=palette, showfliers=False)

        for label in p.xaxis.get_ticklabels()[::2]:
            # hide ticklabel every 2 ticks to make them less dense
            label.set_visible(False)
        ax.legend_.remove()  # remove individual legends from subplots
        p.set_title(location, fontdict={'fontsize': 16})
        p.set(ylim=(-1, 21))
        p.set(yticks=np.arange(0, 20.5, 2.5))
        p.set(xlabel='')  # With None doesn't work in servers
        p.set(ylabel='')
        ax.tick_params(labelsize=14)

    fig.text(0.52, 0.06, 'Month' if x_axis ==
             'FertMonth' else 'Year', ha='center', fontsize=18)
    fig.text(0.07, 0.5, 'Residuals', va='center',
             rotation='vertical', fontsize=18)

    # Create a legend
    axLine, axLabel = axes[0, 0].get_legend_handles_labels()

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines[-3:], labels[-3:], loc='center right', fontsize=14)

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.9,
                        wspace=0.05,
                        hspace=0.125)

    plt.savefig('boxplots_' + known_unknown + '_' + x_axis +
                ('' if irrigation is None else ('_irrigation_' + str(irrigation))) + '.png')
    plt.close()
