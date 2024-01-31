from requests import get
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
import time

BASE_URL = 'https://www.betmma.tips/'

ODDS_PAGE = BASE_URL + 'mma_betting_favorites_vs_underdogs.php?Org=1'

COLUMNS = [      
    "event",
    "date",
    "favourite",
    "favourite_odds",
    "underdog",
    "underdog_odds",
    "result"
]

DRAW = "#cccccc"
WIN = "#A2FC98"

def get_event_info(event_link) -> list:
    response = get(event_link)
    DOM = BeautifulSoup(response.content, 'html.parser')

    event_name_selector  = (
        "body > table:nth-child(1) > tr:nth-child(4) > td:nth-child(2)"
        " > table:nth-child(1) > tr:nth-child(1) > td:nth-child(1)"
        " > table:nth-child(1) > tr:nth-child(2) > td:nth-child(1)"
    )
    event = DOM.select_one(f"{event_name_selector} > h1").text.strip()
    date = DOM.select_one(f"{event_name_selector} > h2").text.strip().split("; ")[1]
    
    fights_selector  = (
        "body > table:nth-child(1) > tr:nth-child(4) > td:nth-child(2)"
        " > table:nth-child(1) > tr:nth-child(1) > td:nth-child(1)"
        " > table:nth-child(1) > tr:nth-child(4) > td:nth-child(1)"
        " > table:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > table"
    )
    fights = DOM.select(fights_selector)

    data = []

    for pane in fights:
        
        names = pane.find_all(
            "a", 
            href=lambda href: href and href.startswith("fighter_profile.php")
        )

        fighter_1, fighter_2 = [names[i].text.strip() for i in range(2)]

        odds = pane.select('tr')[4].select('tr > td')

        fighter_1_odds, fighter_2_odds = [
            float(odds[i].text.strip()[1:]) for i in range(2)
        ]

        if fighter_1_odds < fighter_2_odds:
            favourite, favourite_odds = fighter_1, fighter_1_odds
            underdog, underdog_odds = fighter_2, fighter_2_odds

        else:
            favourite, favourite_odds = fighter_2, fighter_2_odds
            underdog, underdog_odds = fighter_1, fighter_1_odds
    
                
        try: 
            result = (
                "Favourite"
                if favourite in names[2].text.strip()
                else "Underdog" 
            )
        except IndexError:
            result = "Draw"
                        
        
            
        data.append([
            event,
            date,
            favourite,
            favourite_odds,
            underdog,
            underdog_odds,
            result
        ])
    return data

def get_odds_df (page:str = ODDS_PAGE) -> pd.DataFrame:
    odds_df = pd.DataFrame(columns=COLUMNS)
    
    response = get(ODDS_PAGE)
    DOM = BeautifulSoup(response.content, 'html.parser')
    selector = (
        "body > table:nth-child(1) > tr:nth-child(4) > td:nth-child(2)"
        " > table:nth-child(1) > tr:nth-child(1) > td:nth-child(1)"
        " > table:nth-child(1) > tr:nth-child(4) > td:nth-child(1)"
    )
    event_links = DOM.select_one(selector).find_all(
        "a", 
        href=lambda href: href and href.startswith(
            "mma_event_betting_history.php?Event="
        )
    )

    for event in event_links:
        event_link = f"https://www.betmma.tips/{event.get('href')}"
        event_data = get_event_info(event_link) 
        event_df = pd.DataFrame(event_data, columns=COLUMNS)
        odds_df = pd.concat([odds_df, event_df])
        time.sleep(0.5) 
    
    return odds_df

odds_df = get_odds_df()

odds_df.to_csv('./fights_odds.csv', sep=',', index=False)