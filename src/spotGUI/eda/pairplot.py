import seaborn as sns
import matplotlib.pyplot as plt


def generate_pairplot(data, size, target_column, title="", show=False, sample=False):
    """Generates a pairplot of the data.

    Args:
        data (pandas.DataFrame):
            The data.
        target_column (str):
            The target column.
        title (str):
            The title of the plot.
        show (bool):
            Whether to show the plot.
        sample (bool):
            Whether a sub-sample of the data is used.
        size (int):
            The size of the full data set.

    """
    if sample:
        title = title + " (Subset of 1000 Samples from " + str(size) + " Samples)"
    else:
        title = title + " (" + str(size) + " Samples)"
    pair_plot = sns.pairplot(data, hue=target_column, corner=False)
    plt.subplots_adjust(top=0.95)
    plt.suptitle(title, fontsize=10)
    if show:
        plt.show()
    pair_plot
