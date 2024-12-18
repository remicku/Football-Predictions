import predictions as p

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

'''
Scraping
'''

#print(data.text)
#print(standings_table)
#print(links)
#print(team_urls)
#print(matches[0])
#print('\n'.join(links))
#print(shooting.head())
#print(shooting["Date"])
#print(match_df)
#print('\n'.join(team_urls))

'''
Predictions
'''
#print(p.matches.dtypes)
#print(predictions.matches.dtypes)
#print(p.matches)

#print(p.acc)
#print(pd.crosstab(index = p.combined["actual"], columns=p.combined["prediction"]))
#print("precision: ", precision_score(p.test["target"], p.preds))

#print(p.matches_rolling)
#print(p.mapping["Wolverhampton Wanderers"])
print(p.merged[(p.merged["prediction_x"] == 1) & (p.merged["prediction_y"] == 0)]["actual_x"].value_counts())
#print(p.merged[(p.merged["prediction_x"] == 0) & (p.merged["prediction_y"] == 0)])