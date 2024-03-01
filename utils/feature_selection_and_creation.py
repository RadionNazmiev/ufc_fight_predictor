import numpy as np
import pandas as pd


def get_stats(fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various statistics for a given DataFrame of fighter data.

    Parameters:
        fighters_df (pd.DataFrame): A DataFrame containing fighter data, with
            columns for "name", "target", and any other relevant data.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
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