import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
data = requests.get(standings_url)

#print(data.text)

soup = BeautifulSoup(data.text)
standings_table = soup.select('table.stats_table')[0]

#print(standings_table)

links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]

#print(links)

team_urls = [f"https://fbref.com{l}" for l in links]
#print(team_urls)

team_url = team_urls[0]
data = requests.get(team_url)

matches = pd.read_html(data.text, match="Scores & Fixtures")
#print(matches[0])

soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]

#print('\n'.join(links))

data = requests.get(f"https://fbref.com{links[0]}")

shooting = pd.read_html(data.text, match="Shooting")[0]
#print(shooting.head())

shooting.columns = shooting.columns.droplevel()
#print(shooting["Date"])

team_data = matches[0].merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")

years = list(range(2024, 2020, -1))

all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()

        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue

        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(12)

match_df = pd.concat(all_matches)

#print(match_df)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("matches.csv")
        

#print('\n'.join(team_urls))