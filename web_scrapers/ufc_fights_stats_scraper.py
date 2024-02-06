from requests import get
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
import numpy as np
import time
from loguru import logger

PAGES = 'abcdefghijklmnopqrstuvwxyz'

FIGHTERS_COLUMNS = [      
    'name',
    'record',
    'height',
    'weight',
    'reach',
    'stance',
    'dob'
]

FIGHTS_COLUMNS = [
    'event_name',
    'event_location',
    'event_datetime',
    'fighter_1',
    'fighter_2',
    'fighter_1_result',
    'title',
    'method',
    'last_round',
    'last_round_time',
    'time_format',
    'referee',
    'details',
    'f1_total_knock_downs',
    'f2_total_knock_downs',
    'f1_total_signif_str',
    'f2_total_signif_str',
    'f1_total_total_str',
    'f2_total_total_str',
    'f1_total_take_downs',
    'f2_total_take_downs',
    'f1_total_sub_atts',
    'f2_total_sub_atts',
    'f1_total_rev',
    'f2_total_rev',
    'f1_total_ctrl',
    'f2_total_ctrl',
    'f1_total_head',
    'f2_total_head',
    'f1_total_body',
    'f2_total_body',
    'f1_total_leg',
    'f2_total_leg',
    'f1_total_distance',
    'f2_total_distance',
    'f1_total_clinch',
    'f2_total_clinch',
    'f1_total_ground',
    'f2_total_ground'
]

def get_fight_essential_details(fight_html) -> np.ndarray:
    fighter_1 = fight_html.select_one(
        'div.b-fight-details__persons.clearfix > div:nth-of-type(1) a'
        ).text.strip()
    
    fighter_2 = fight_html.select_one(
        'div.b-fight-details__persons.clearfix > div:nth-of-type(2) a'
        ).text.strip()
    
    fighter_1_result = fight_html.select_one(
        'div.b-fight-details__persons.clearfix > div:nth-of-type(1) > i'
        ).text.strip()
    
    title = fight_html.select_one(
        'i.b-fight-details__fight-title'
        ).text.strip()
    
    method = fight_html.select_one(
        'i.b-fight-details__text-item_first:nth-of-type(1) > i:nth-of-type(2)'
        ).text.strip()
    
    last_round = fight_html.select_one(
        'i.b-fight-details__text-item:nth-of-type(2)'
        ).text.split(':')[1].strip()
    
    last_round_time = fight_html.select_one(
        'i.b-fight-details__text-item:nth-of-type(3)'
        ).text.split('Time:')[1].strip()
    
    time_format = fight_html.select_one(
        'i.b-fight-details__text-item:nth-of-type(4)'
        ).text.split(':')[1].strip()
    
    referee = fight_html.select_one(
        'i.b-fight-details__text-item:nth-of-type(5) > span'
        ).text.strip()
    
    details = fight_html.select_one(
        'div.b-fight-details__content > p.b-fight-details__text:nth-of-type(2)'
        ).get_text(strip=True).split(':')[1]
     
    return np.array([
        fighter_1,
        fighter_2,
        fighter_1_result,
        title,
        method,
        last_round,
        last_round_time,
        time_format,
        referee,
        details
    ])

def get_table_data(table_data, desired_columns) -> np.ndarray:
    result = list()

    for col in desired_columns:
        for i in range(2):
            value = (
                table_data[col].select_one(f"td > p:nth-of-type({i+1})")
                    .get_text(strip=True)
            )
            result.append(value)

    return np.array(result)

def get_fight_totals(fight_html, desired_columns) -> np.ndarray:
    try:
        table_data = fight_html.select(
            'section.b-fight-details__section:nth-child(4) > table:nth-child(1)' 
            + '> tbody:nth-child(2) > tr:nth-child(1) > td'
        )
        table_data = get_table_data(table_data,desired_columns)
        logger.info("Succesfully retreived totals for the fight")

        return table_data

    except Exception as e:
        logger.info("No Totals For This Fight Found")
        return np.full((len(desired_columns)*2,), np.nan)

def get_fight_significant_strikes(fight_html, desired_columns) -> np.ndarray:
    try:
        table_data = fight_html.select(
            '.b-fight-details > table:nth-child(7) > tbody > tr > td'
        )
        table_data = get_table_data(table_data,desired_columns)

        logger.info("Succesfully retreived significant strikes for the fight")

        return table_data

    except Exception as e:
        logger.info("No Significant Strikes For This Fight Found")
        return np.full((len(desired_columns)*2,), np.nan)
      
def get_single_fight_stats(fight_html) -> (np.ndarray, np.ndarray, np.ndarray):
    essential_details = get_fight_essential_details(fight_html)

    logger.info("Succesfully retreived details for the fight " +
        f"between {essential_details[0]} and {essential_details[1]}")

    desired_total_columns=[1, 2, 4, 5, 7, 8, 9]
    totals = get_fight_totals(fight_html,desired_total_columns)
    
    desired_signif_str_columns=[3, 4, 5, 6, 7, 8]
    significant_strikes = (
        get_fight_significant_strikes(fight_html, desired_signif_str_columns)
    )

    return (essential_details, totals, significant_strikes)

def is_fight_exist(fight,fights_df) -> bool:
    fighter_1 = (
        fight.select_one('td:nth-child(2) > p:nth-child(1) > a')
        .get_text(strip=True)
    )

    fighter_2 = (
        fight.select_one('td:nth-child(2) > p:nth-child(2) > a')
        .get_text(strip=True)
    )

    logger.info(
        f"Proceeding with fight between {fighter_1} and {fighter_2}"
    )

    event_datetime = (
        fight.select_one('td:nth-child(7) > p:nth-child(2)')
            .get_text(strip=True)
    )

    is_same_event_datetime = (
        str(fights_df.event_datetime) == event_datetime

    )
    fought_before = (
        ((fights_df.fighter_1 == fighter_1) &
        (fights_df.fighter_2 == fighter_2)) |
        ((fights_df.fighter_1 == fighter_2) &
        (fights_df.fighter_2 == fighter_1))
    )

    if fights_df[fought_before & is_same_event_datetime].shape[0] > 0:
        return True
    else:
        return False
    
def get_event_location(event_link,fights_df) -> str:

    is_event_name_exist = (
        fights_df.event_name == event_link.get_text(strip=True)
    )

    if fights_df[is_event_name_exist].empty:

        response = get(event_link.get('href'))
        event_html = BeautifulSoup(response.content, 'html.parser') 
        event_location = event_html.select_one(
            'li.b-list__box-list-item:nth-of-type(2)'
            ).text.split(":")[1].strip()
    else:
        event_location = fights_df[is_event_name_exist].iloc[0].event_location

    return event_location

def get_fights_entries(fighter_html,fights_df):
    fights = (
        fighter_html.select('tbody.b-fight-details__table-body > tr')[1:]
    )

    for fight in fights:
        is_upcoming = (
            fight.select_one(
                'i:nth-child(1) > i:nth-child(1)'
                ).get_text(strip = True)
                .lower() 
                == 'next'
        )
        if is_upcoming:
            continue
        
        is_exist = is_fight_exist(fight,fights_df)
       
        if is_exist:
            continue

        event_link = (
            fight.select_one('td:nth-child(7) > p:nth-child(1) > a')
        )
        event_name = event_link.get_text(strip=True)

        event_location = get_event_location(event_link,fights_df)

        event_datetime = (
            fight.select_one('td:nth-child(7) > p:nth-child(2)')
                .get_text(strip=True)
        )

        response = get(fight.get('data-link'))
        fight_html = BeautifulSoup(response.content, 'html.parser')

        essential_details, totals, significant_strikes = (
            get_single_fight_stats(fight_html)
        )

        entry = [
            event_name,
            event_location,
            event_datetime
        ]

        entry.extend(essential_details)
        entry.extend(totals)
        entry.extend(significant_strikes)

        fights_df.loc[fights_df.shape[0],:] = entry
                
        time.sleep(0.5)

def get_fighter_entries(fighter_html,fighters_df):
    name = (
        fighter_html.select_one('span.b-content__title-highlight').text.strip()
    )
    logger.info(f"Proceeding to retrieve stats of fighter: {name}")
    record = (
        fighter_html.select_one('span.b-content__title-record')
            .text
            .strip()
            .split(': ')[1]
    )

    fighter_elements = (
        fighter_html.select('section > div > div > div:nth-of-type(1) li')
    )

    height, weight, reach, stance, dob = [
        i.get_text(strip=True).split(':')[1] for i in fighter_elements
    ]

    fighter_entry = np.array([
        name,
        record,
        height,
        weight,
        reach,
        stance,
        dob
    ])

    fighters_df.loc[fighters_df.shape[0],:] = fighter_entry
    logger.info(f"Done retrieving stats of fighter: {name}")

def get_data(pages:str = PAGES):   
    fighters_df = pd.DataFrame(columns=FIGHTERS_COLUMNS)
    fights_df = pd.DataFrame(columns=FIGHTS_COLUMNS) 

    for page in PAGES:
        logger.info(f"Proceeding with page: {page.upper()}")
        page_link = (
            f"http://ufcstats.com/statistics/fighters?char={page}&page=all"
        )
        response = get(page_link)
        fighters_html = BeautifulSoup(response.content, 'html.parser')
        fighters = (
            fighters_html.select('tr.b-statistics__table-row')[2:]
        )

        for fighter in fighters:
            response = get(fighter.select_one('a').get('href'))
            fighter_html = BeautifulSoup(response.content, 'html.parser')

            get_fighter_entries(fighter_html,fighters_df)

            get_fights_entries(fighter_html,fights_df)
        
        fighters_df.to_csv("fighters_stats.csv", sep=',', mode='w', index=False)
        fights_df.to_csv("fights_stats.csv", sep=',', mode='w', index=False)

    return (fighters_df, fights_df)

get_data()
