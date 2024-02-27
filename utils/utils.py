import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from fuzzywuzzy import process

def get_odds_from_match(entry: pd.DataFrame, closest_match: str) -> float:
    """
    Return the odds for the closest match in the given entry DataFrame.
    
    Parameters:
    - entry: pd.DataFrame, the DataFrame containing match information
    - closest_match: str, the name of the closest match
    
    Returns:
    - float, the odds for the closest match
    """

    if closest_match in entry['favourite'].tolist():
        i = entry['favourite'] == closest_match
        odds = entry.loc[i, 'favourite_odds'].iloc[0]
    else:
        i = entry['underdog'] == closest_match
        odds = entry.loc[i, 'underdog_odds'].iloc[0]

    return odds

def find_closest_match(entry: pd.DataFrame, fighter_name: str) -> str:
    """
    Find the closest match to the given fighter name in the provided DataFrame.

    :param entry: A pandas DataFrame containing 'favourite' and 'underdog' 
    columns.
    
    :param fighter_name: A string representing the fighter name to find the 
    closest match for.

    :return: A string representing the closest match to the fighter name.
    """

    possible_matches = entry['favourite'].tolist() + entry['underdog'].tolist()
    closest_match, _ = process.extractOne(fighter_name, possible_matches)
    return closest_match

def get_odds_for_fighter(
        fights_odds_df: pd.DataFrame, 
        row: pd.Series, 
        fighter_name_col: str) -> float:
    """
    Get the odds for a specific fighter in a fight.

    Parameters:
        fights_odds_df (pd.DataFrame): The DataFrame containing the odds for all
        fights.

        row (pd.Series): The row containing the information about the fight.

        fighter_name_col (str): The name of the column in the DataFrame that 
        contains the fighter's name.

    Returns:
        float: The odds for the specified fighter in the fight.
    """

    odds = np.nan
    mask = (fights_odds_df.date == row.event_datetime)
    entry = fights_odds_df[mask]

    if not entry.empty:
        closest_match = find_closest_match(entry, row[fighter_name_col])
        odds = get_odds_from_match(entry, closest_match)

    return odds

def add_fighter_prefix(fighter_number: int, df: pd.DataFrame) -> dict:
    """
    Generate a dictionary with modified column names in the DataFrame.
    
    Args:
        fighter_number (int): The fighter number to use as a prefix.
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        dict: A dictionary with modified column names, where columns not in 
        ['fight_id', 'event_date'] have the fighter number as a prefix.
    """
    return { 
        col: f"f{fighter_number}_{col}" if col not in ['fight_id', 'event_date'] 
        else col 
        for col in df.columns.tolist()
    }
    
def get_rolling_features(
        df: pd.DataFrame, 
        cols: list, 
        weeks: int, 
        stat: str) -> pd.DataFrame:
    """
    Calculate rolling features over a specified number of weeks for the given 
    columns and statistic.

    Args:
      df: A pandas DataFrame containing the data

      cols: A list of column names for which rolling features are to be 
    calculated

      weeks: An integer representing the number of weeks to consider for the 
    rolling features

      stat: A string representing the statistic to calculate (e.g., "mean", 
    "sum", "count")

    Returns:
      A pandas DataFrame containing the calculated rolling features
    """
    rolling_features = [
        f"{col}_{stat}_over_{int(weeks/52)}_year" for col in cols
    ] 
    rolling_features_df = pd.DataFrame(columns=rolling_features)

    for index, row in df.iterrows():
        time_constraint = row['event_date'] - pd.Timedelta(weeks=weeks)
        mask = (
            (df['name'] == row['name']) &
            (df['event_date'] >= time_constraint)
        )
        slice_df = df.loc[
            mask & (df.index <= index), 
            ['event_date', 'name'] + cols
        ]
        for col in cols:
            if stat == "mean":
                result = slice_df[col].mean()
            elif stat == "sum":
                result = slice_df[col].sum()
            elif stat == "count":
                result = slice_df[col].count()
            else:
                result = None
            col_name = f"{col}_{stat}_over_{int(weeks/52)}_year"
            rolling_features_df.loc[index,col_name] = result
 
    return rolling_features_df

def get_stats(fighters_df: pd.DataFrame) -> pd.DataFrame:

    features = {
        'winning_streak': [],
        'wins_over_carrier': [],
        'losses_over_carrier': [],
        'draws_over_carrier': [],
        'ncs_over_carrier': []
    }
    for fighters_index, fighters_row in fighters_df.iterrows():
        mask = (fighters_df['name'] == fighters_row['name'])
        slice = fighters_df.loc[
            mask & (fighters_df.index <= fighters_index), 
            ['target']
        ]

        win_streak = 0
        wins = len(slice[slice['target'] == 'W'])
        losses = len(slice[slice['target'] == 'L'])
        draws = len(slice[slice['target'] == 'Draw'])
        ncs = len(slice[slice['target'] == 'NC'])

        for _, slice_row in slice[::-1].iterrows():
            if slice_row['target'] == 'W':
                win_streak += 1
            elif slice_row['target'] == 'Draw':
                win_streak += 0
            elif slice_row['target'] == 'NC':
                win_streak += 0
            else:
                break

        features['winning_streak'].append(win_streak)  
        features['wins_over_carrier'].append(wins)  
        features['losses_over_carrier'].append(losses)  
        features['draws_over_carrier'].append(draws)  
        features['ncs_over_carrier'].append(ncs)  

    return pd.DataFrame(features)

def get_seconds(x: str) -> int:
    """
    This function takes a string representing a time in minutes and seconds (in the format "MM:SS") and returns the total number of seconds. 
    It accepts a single parameter:
      x: a string representing the time in "MM:SS" format
    It returns an integer representing the total number of seconds, or np.nan if the input is null.
    """
    if pd.notna(x):
        min, sec = map(int, x.split(":"))
        result = min*60 + sec
    else:
        result = np.nan
    return result

def get_lan_thr_cols(
        df: pd.DataFrame, 
        columns_to_split: list) -> pd.DataFrame:
    """
    Function to split columns in a DataFrame and create a new DataFrame with the split columns for language and threshold values.
    
    Args:
    - df: A pandas DataFrame to split columns from.
    - columns_to_split: A list of column names to split.

    Returns:
    - A pandas DataFrame with the split columns for language and threshold values.
    """
    columns = (
        [f"{col}_thr" for col in columns_to_split] + 
        [f"{col}_lan" for col in columns_to_split]
    )
    lan_thr_df = pd.DataFrame(columns=columns)

    for index, row in df.loc[:, columns_to_split].iterrows():
        for feature in columns_to_split:
            if pd.notna(row[feature]):
                lan, thr = map(int, row[feature].split(" of "))
                lan_thr_df.at[index, feature + '_lan'] = lan
                lan_thr_df.at[index, feature + '_thr'] = thr
            else:
                lan_thr_df.at[index, feature + '_lan'] = np.nan
                lan_thr_df.at[index, feature + '_thr'] = np.nan

    return lan_thr_df

def get_fight_lasted(row: pd.Series) -> int:
    """
    Calculate the duration of the fight based on the input row. 

    Args:
    - row: pd.Series - The input row containing information about the fight.

    Returns:
    - int: The duration of the fight in seconds.
    """
    minutes, seconds = map(int, row.last_round_time.split(":"))
    last_round_time = minutes*60 + seconds

    if (row.time_format == 'No Time Limit') or \
        (row.time_format.startswith('Unlimited Rnd')):
        return last_round_time
    else:
        rounds_list = row.time_format.split(' ')[-1][1:-1].split('-')
        last_round_index = int(row.last_round)-1
        if last_round_index == 0:
            return last_round_time
        else:
            rounds_list = rounds_list[:last_round_index]
            result = len(rounds_list)*int(rounds_list[0])*60. + last_round_time
            return result
        
def get_weight_class(title: str) -> str:
    """
    Get the weight class from the given title.

    Args:
        title (str): The title to extract the weight class from.

    Returns:
        str: The extracted weight class or "unknown weight" if not found.
    """
    if 'weight' in title.lower():
        for word in title.lower().split(" "):
            if ('weight' in word) & (len(word) > 6):
                return word
            elif 'weight' in word:
                index = title.lower().split(" ").index(word)
                weight = " ".join(title.lower().split(" ")[index-1:index+1])
                return weight
            else:
                continue
    else:
        return "unknown weight"
    
def feet_to_inches(height: str) -> int:
    """
    Convert the given height from feet and inches to total inches.

    Args:
        height (str): The height in the format "feet inches".

    Returns:
        int: The total height in inches.
    """
    if pd.isna(height):
        return height
    new_hength = 0
    feet, inches = map(lambda x: int(x[:-1]), height.split(" "))
    new_hength = feet*12 + inches
    return new_hength


def swap_fighter_positions(
        df: pd.DataFrame, 
        columns_to_swap: dict) -> pd.DataFrame:
    """
    Swaps the positions of 'F1' and 'F2' fighters in the DataFrame based on the given columns_to_swap mapping.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_swap (dict): A dictionary mapping the columns to be swapped.
        
    Returns:
        pd.DataFrame: The DataFrame with the fighter positions swapped.
    """

    result_df = pd.DataFrame(columns=df.columns)

    mask = (df.target == 'F1') | (df.target == 'F2')

    for index, series in df[mask].iterrows():
        if index % 2 == 0:
            if df.loc[index, 'target'] != 'F1':
                series = series.rename(index=columns_to_swap)
                series['target'] = 'F1'
        else:
            if df.loc[index, 'target'] != 'F2':
                series = series.rename(index=columns_to_swap)
                series['target'] = 'F2'
        result_df.loc[index,:] = series 

    return result_df

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