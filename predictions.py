import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

matches = pd.read_csv("matches_short.csv", index_col=0)

matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes ## Encoding a home game as 1 and an away game as 0
matches["opp_code"] = matches["opponent"].astype("category").cat.codes ## Encoding opponents as a number
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") ## Encoding hours as a number
matches["day_code"] = matches["date"].dt.day_of_week ## Encoding day of week of game as a number

matches["target"] = (matches["result"] == "W").astype("int") ## Encoding a win as 1 and loss/draw as 0

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01'] ## Training data: matches before 2022
test = matches[matches["date"] > '2022-01-01'] ## Testing data: matches after 2022
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors]) ## Making prediction

acc = accuracy_score(test["target"], preds) ## Testing accuracy
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City") ## Isolating a single team's data (e.g., Manchester City) for analysis

def rolling_averages(group, cols, new_cols):
    """
    Compute rolling averages for the last 10 matches
    """
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) ## Dropping missing values and replacing with empty
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("team") ## Dropping extra index level
matches_rolling.index = range(matches_rolling.shape[0]) ## Resetting index to maintain a continuous numerical index after applying groupby

def make_predictions(data, predictors):
    """
    Train and evaluate a RandomForest model
    """
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key ## Returns the original key if it's not found in the dictionary (prevents KeyError)


# Map different names for same team
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) ## Merging predictions to align both home and away teams in a single row for easier analysis