import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from fuzzywuzzy import process

def plot_corr_matrix(df: pd.DataFrame, columns: list):
    """
    Plots a correlation matrix for the specified DataFrame and columns.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        columns (list): The list of columns to include in the correlation matrix.

    Returns:
        None
    """
    plt.figure(figsize=(25, 25))
    plt.title("Correlation Matrix", fontsize=20) 
    sns.heatmap(
        df[columns].corr(),
        vmin=-1,
        vmax=1,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        fmt=".1f",
        annot=True
    )
    plt.show()


def plot_countplot(df: pd.DataFrame, x:str, title: str): 
    """
    Plot the distribution of fights by weight class using a count plot.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data for plotting.
    
    Returns:
    None
    """
    plt.figure(figsize=(8,5))
    plt.xticks(rotation=45)
    sns.set_theme(style="white")
    palette = [
        "#9b59b6", 
        "#3498db", 
        "#95a5a6", 
        "#e74c3c", 
        "#34495e", 
        "#2ecc71", 
        "#f1c40f", 
        "#e67e22", 
        "#d35400", 
        "#1abc9c", 
        "#27ae60", 
        "#2980b9"
    ]
    ax = sns.countplot(
        x=x, 
        hue=x, 
        data=df, 
        palette=palette, 
        legend=False
    )
    for p in ax.patches:
        height = int(p.get_height())
        ax.text(p.get_x()+p.get_width()/2., 
                height + 3,
                height, 
                ha="center")
    ax.set_title(title)
    ax.set_ylabel("")  
    ax.set_xlabel("")
    ax.grid(False)
    plt.gca().set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)  
    ax.set_yticks([])
    plt.show()

def plot_lineplot(
        df: pd.DataFrame, 
        title:str, 
        legend: list
    ): 
    plt.figure(figsize=(8,5))
    sns.set_style("white")
    df.plot(kind='line', marker='o')
    plt.title(title)
    plt.legend(legend)
    plt.xticks(df.index[::5])
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.show()

def plot_hist(
        data: pd.DataFrame, 
        x: str, 
        bins, 
        figsize: tuple = (5, 5),
        xlabel: str = '',
        ylabel: str = '',
        title: str = ''
        ):

    plt.figure(figsize=(5,5))
    sns.histplot(data=data, x=x, bins=bins)
    sns.set_style("white")

    plt.xlabel('Fight Duration (seconds)')
    plt.ylabel('')  
    plt.title('Distribution of UFC Fight Durations')

    plt.tight_layout()
    plt.show()

def plot_scatterplots(
        df: pd.DataFrame, 
        x: str,
        columns: list,
        title: str,
        nrows,
        ncols,
        figsize: tuple = (6, 40)
        ):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)
    plt.subplots_adjust(
        left=0,
        right=0.95,
        top=0.965,
        bottom=0.05,
        wspace=0.2,
        hspace=0.4
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(columns):
            break
        col = columns[i]
        sns.scatterplot(
            x=x,
            y=col,
            data=df,
            ax=ax
        )
        ax.set_title(col)
        ax.set(ylabel='')
        ax.set(xlabel='')

    plt.show()