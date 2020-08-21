# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import poisson
import datetime

# Fetch Dataset : Premier League Data 2013-2018
df1 = pd.read_csv('/Users/daveyap/Desktop/train.csv')
#check
type(df1['Date'][0])
#change date data type
df1['Date'] = df1['Date'].astype('datetime64[ns]')
df2 = df1[(df1['Date'] > '2013-02-04') & (df1['Date'] < '2018-08-10')]


## epl_league_table
# all match point for 2019 premier league
epl_league_table = pd.read_csv('/Users/daveyap/Desktop/epltable.csv')

# upcoming match in 2018/2019
eplmatches = df1[(df1['Date'] >= '2018-08-01')]
eplmatches = eplmatches[['HomeTeam', 'AwayTeam']]

df1 = df1[['HomeTeam', 'AwayTeam','FTHG','FTAG']]
df1 = df1.rename(columns={'FTHG': "HomeGoals", 'FTAG': "AwayGoals"})

# count the frequency of each team
df1.groupby(["HomeTeam", "AwayTeam"]).size().reset_index(name="Time")
# change data type
df1["AwayGoals"] = df1["AwayGoals"].astype(float)
df1["HomeGoals"] = df1["HomeGoals"].astype(float)



# Calculate Team and League stats

# For each team - Average {Home_Scored, Home_Conceded, Away_Scored, Away_Conceded}
HomeTeam = df1[['HomeTeam', 'HomeGoals', 'AwayGoals']].rename(
    columns={'HomeTeam':'Team', 'HomeGoals':'Home_Scored', 'AwayGoals':'Home_Conceded'}).groupby(
    ['Team'], as_index=False)[['Home_Scored', 'Home_Conceded']].mean()

AwayTeam = df1[['AwayTeam', 'HomeGoals', 'AwayGoals']].rename(
    columns={'AwayTeam':'Team', 'HomeGoals':'Away_Conceded', 'AwayGoals':'Away_Scored'}).groupby(
    ['Team'], as_index=False)[['Away_Scored', 'Away_Conceded']].mean()

# Overall - Average {leagueHome_Scored, leagueHome_Conceded, leagueAway_Scored, leagueAway_Conceded}
leagueHome_Scored, leagueHome_Conceded = HomeTeam['Home_Scored'].mean(), HomeTeam['Home_Conceded'].mean()
leagueAway_Scored, leagueAway_Conceded = AwayTeam['Away_Scored'].mean(), AwayTeam['Away_Conceded'].mean()

TeamStrength = pd.merge(HomeTeam, 
AwayTeam, on='Team')

assert(leagueHome_Scored != 0)
assert(leagueHome_Conceded != 0)
assert(leagueAway_Scored != 0)
assert(leagueAway_Conceded != 0)

# Normalize the parameters
# For each team - {Home_Attack, HomeDefence, Away_Attack, Away_Defense}
TeamStrength['Home_Scored'] /= leagueHome_Scored
TeamStrength['Home_Conceded'] /= leagueHome_Conceded
TeamStrength['Away_Scored'] /= leagueAway_Scored
TeamStrength['Away_Conceded'] /= leagueAway_Conceded

TeamStrength.columns=['Team','Home_Attack','Home_Defense','Away_Attack','Away_Defense']
TeamStrength.set_index('Team', inplace=True)

# Overall - {overallHome_Scored, overallAway_Scored}
overallHome_Scored = (leagueHome_Scored+leagueAway_Conceded)/2
overallAway_Scored = (leagueHome_Conceded+leagueAway_Scored)/2


# Predict outcome of match and assign points to the teams
def Score_predict(home, away):
    if home in TeamStrength.index and away in TeamStrength.index:
        lambdH = TeamStrength.at[home,'Home_Attack'] * TeamStrength.at[away,'Away_Defense'] * overallHome_Scored
        lambdA = TeamStrength.at[away,'Away_Attack'] * TeamStrength.at[home,'Home_Defense'] * overallAway_Scored
        probH, probA, probT = 0, 0, 0  # Probability of Home win(H), Away win(A) or Tie(T)
        for X in range(0,11):
            for Y in range(0, 11):
                p = poisson.pmf(X, lambdH) * poisson.pmf(Y, lambdA)
                if X == Y:
                    probT += p
                elif X > Y:
                    probH += p
                else:
                    probA += p
        scoreH = 3 * probH + probT
        scoreA = 3 * probA + probT
        return (scoreH, scoreA)
    else:
        return (0, 0)



#  Simulate the matches to predict final standings
for index, row in eplmatches.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    assert(home in epl_league_table.Team.values and away in epl_league_table.Team.values)
    sH, sA = Score_predict(home, away)
    epl_league_table.loc[epl_league_table.Team == home, 'Points'] += sH
    epl_league_table.loc[epl_league_table.Team == away, 'Points'] += sA


epl_league_table.round(2)