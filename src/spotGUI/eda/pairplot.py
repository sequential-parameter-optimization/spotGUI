import seaborn as sns
import matplotlib.pyplot as plt


def generate_pairplot(df):
    """
    Function to generate a pairplot of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be plotted.

    Returns:
    None
    """
    sns.pairplot(df)
    plt.show()
