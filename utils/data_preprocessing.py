import numpy as np
import pandas as pd

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

def remove_inch_sign(text):
  """Removes the inch sign ("") from the end of a string, if present.

  Args:
      text: The string to process.

  Returns:
      The string with the inch sign removed, or the original string if no inch sign is found.
  """
  if pd.notnull(text) and text[-1] == '"':
    return text[:-1]
  else:
    return text
  
def add_fighter_odds(fights_df, fights_odds_df):
  """
  Adds a new column for each fighter in fights_df containing their odds,
  obtained from fights_odds_df.

  Args:
      fights_df: A pandas dataframe containing fight information.
      fights_odds_df: A pandas dataframe containing fighter odds information.

  Returns:
      A new pandas dataframe with the added columns.
  """

  for index, row in fights_df.iterrows():
    for _, col in enumerate(['f1_name', 'f2_name']):
      odds = get_odds_for_fighter(fights_odds_df, row, col)
      fighter = col.split('_')[0]
      fights_df.at[index, f'{fighter}_odds'] = odds
  return fights_df