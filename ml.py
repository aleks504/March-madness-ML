# -------------------------------------------------------
# Aleksandra Stansbury
# I used logistic regression for this assignment because I went down the machine learning route. I've been wanting to get into machine learning
# for quite some time so after a lot of research I decided to keep it simple and use logistic regression, which will allow me to use a sigmoid function to 
# calculate a percentage for each team, which then will determine win/loss. 
# -------------------------------------------------------

# importing pandas (basically excel)
import pandas as pd
# I chose sklearn for my machine learning (includes models like logistic regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Using numpy for arrays
import numpy as np

# I am first loading the data about the teams, seeds, tournament information and season information
teams        = pd.read_csv("march-machine-learning-mania-2026/MTeams.csv")
seeds        = pd.read_csv("march-machine-learning-mania-2026/MNCAATourneySeeds.csv")
tourney      = pd.read_csv("march-machine-learning-mania-2026/MNCAATourneyCompactResults.csv")
season       = pd.read_csv("march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv")

# Wins - each game has a winner (WTeamID) which we set as 1
wins = season[["Season", "WTeamID", "WScore", "LScore"]].copy()
wins.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
wins["Win"] = 1

# Losses - each game has a loser (LTeamID) which we set as 0
losses = season[["Season", "LTeamID", "LScore", "WScore"]].copy()
losses.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
losses["Win"] = 0

# Grouping into one row per team using pandas
all_games = pd.concat([wins, losses])

# Aggregate all the teams per season informationi using pandas
team_stats = all_games.groupby(["Season", "TeamID"]).agg(
    WinPct      = ("Win", "mean"),
    AvgPtDiff   = ("PointsFor", "mean"), 
    AvgPtsFor   = ("PointsFor", "mean"),
    AvgPtsAgainst = ("PointsAgainst", "mean"),
    GamesPlayed = ("Win", "count")
).reset_index()

# Averge point differential calculations here 
team_stats["AvgPtDiff"] = team_stats["AvgPtsFor"] - team_stats["AvgPtsAgainst"]

# This part I used Claude for because it was confusing but it converts the seed column into a number
# For example, a seed "W01" would be converted to a number 1
seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)


# Begining training using logistic regression model, so for each tournament game we create a row with:
# 1. Seed difference between two teams
# 2. Point diff difference
# 3. Win percentage difference
# 4. Assign a 1 if team 1 wins, 0 otherwise
def build_training_data(tourney_df, team_stats_df, seeds_df):
    rows = []

    for _, game in tourney_df.iterrows():
        season  = game["Season"]
        w_team  = game["WTeamID"]
        l_team  = game["LTeamID"]

        # Getting team stats
        w_stats = team_stats_df[(team_stats_df.Season == season) &
                                (team_stats_df.TeamID == w_team)]
        l_stats = team_stats_df[(team_stats_df.Season == season) &
                                (team_stats_df.TeamID == l_team)]

        # Getting seeds
        w_seed_row = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == w_team)]
        l_seed_row = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == l_team)]

        # Skip if data is gone for whatever reason
        if w_stats.empty or l_stats.empty or w_seed_row.empty or l_seed_row.empty:
            continue

        w_seed = w_seed_row["SeedNum"].values[0]
        l_seed = l_seed_row["SeedNum"].values[0]

        w_wp   = w_stats["WinPct"].values[0]
        l_wp   = l_stats["WinPct"].values[0]

        w_pd   = w_stats["AvgPtDiff"].values[0]
        l_pd   = l_stats["AvgPtDiff"].values[0]

        # Randomly assign teams as team 1 vs team 2
        import random
        if random.random() > 0.5:
            rows.append({
                "SeedDiff":   w_seed - l_seed,
                "WinPctDiff": w_wp - l_wp,
                "PtDiffDiff": w_pd - l_pd,
                "Label": 1 # team 1 wins
            })
        else:
            rows.append({
                "SeedDiff":   l_seed - w_seed,
                "WinPctDiff": l_wp - w_wp,
                "PtDiffDiff": l_pd - w_pd,
                "Label": 0 # team1 loses 
            })

    return pd.DataFrame(rows)


training_data = build_training_data(tourney, team_stats, seeds)

print(f"\n--- Training data: {len(training_data)} rows ---")
print(training_data.head())
print("\nLabel distribution (0=loss, 1=win):")
print(training_data["Label"].value_counts())

# Actually training the model with logistic regression function from sklearn
# I chose logistic regression because we want a yes/no answer rather than an exact score, though we could theoretically get an exact
# score if we changed it to linear regression
# For times sake I won't do that though
features = ["SeedDiff", "WinPctDiff", "PtDiffDiff"]
X = training_data[features]
y = training_data["Label"]

# Train on the older data and test on new data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression() # pick logistic regression modle
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"\n=== Model accuracy: {acc:.1%} ===")

# Feature weights 
for feat, coef in zip(features, model.coef_[0]):
    print(f"  {feat}: {coef:.3f}")


# Predicts team matchup
def predict_matchup(team1_seed, team1_winpct, team1_ptdiff,
                    team2_seed, team2_winpct, team2_ptdiff):
    X_new = np.array([[
        team1_seed  - team2_seed,
        team1_winpct - team2_winpct,
        team1_ptdiff - team2_ptdiff
    ]])
    prob = model.predict_proba(X_new)[0][1]
    return prob

# Example: #1 seed (73% WR, +12 pt diff) vs #8 seed (58% WR, +3 pt diff)
prob = predict_matchup(1, 0.73, 12, 8, 0.58, 3)
print(f"\nExample matchup — #1 vs #8 seed: {prob:.1%} chance team1 wins")
print(seeds["Season"].unique())

# Prediciting the 2025 bracket here, could also use 2026's data for this bracket when seeds come out
# Get all 2025 tournament seeds
seeds_2025 = seeds[seeds.Season == 2025].copy()
stats_2025 = team_stats[team_stats.Season == 2025].copy()

def predict_game_2025(team1_id, team2_id):
    t1_seed  = seeds_2025[seeds_2025.TeamID == team1_id]["SeedNum"].values[0]
    t2_seed  = seeds_2025[seeds_2025.TeamID == team2_id]["SeedNum"].values[0]

    t1_stats = stats_2025[stats_2025.TeamID == team1_id].iloc[0]
    t2_stats = stats_2025[stats_2025.TeamID == team2_id].iloc[0]

    X_new = np.array([[
        t1_seed            - t2_seed,
        t1_stats["WinPct"] - t2_stats["WinPct"],
        t1_stats["AvgPtDiff"] - t2_stats["AvgPtDiff"]
    ]])

    prob   = model.predict_proba(X_new)[0][1]
    winner = team1_id if prob > 0.5 else team2_id
    conf   = prob if prob > 0.5 else 1 - prob
    print(f"  Team {team1_id} vs Team {team2_id} → Team {winner} wins ({conf:.0%} confidence)")
    return winner

# Print all 2025 first round matchups
print("\n--- 2025 Tournament First Round Predictions ---")
tourney_2025 = tourney[tourney.Season == 2025]
for _, game in tourney_2025.iterrows():
    predict_game_2025(game["WTeamID"], game["LTeamID"])

# -------------------------------------------------------
# BRACKET PRINTER
# Predicts and prints the full 2025 tournament as a bracket
# -------------------------------------------------------

# First build a lookup so we can show team names instead of IDs
id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

def predict_game_silent(team1_id, team2_id):
    """Same as predict_game_2025 but returns winner without printing"""
    try:
        t1_seed  = seeds_2025[seeds_2025.TeamID == team1_id]["SeedNum"].values[0]
        t2_seed  = seeds_2025[seeds_2025.TeamID == team2_id]["SeedNum"].values[0]
        t1_stats = stats_2025[stats_2025.TeamID == team1_id].iloc[0]
        t2_stats = stats_2025[stats_2025.TeamID == team2_id].iloc[0]

        X_new = pd.DataFrame([[
            t1_seed             - t2_seed,
            t1_stats["WinPct"]  - t2_stats["WinPct"],
            t1_stats["AvgPtDiff"] - t2_stats["AvgPtDiff"]
        ]], columns=["SeedDiff", "WinPctDiff", "PtDiffDiff"])

        prob = model.predict_proba(X_new)[0][1]
        return team1_id if prob > 0.5 else team2_id
    except:
        return team1_id  # default if data missing

def predict_region_bracket(region_letter):
    """Predict all rounds for one region, returns (results, region_winner)"""
    # Get all teams in this region and sort by seed
    region_seeds = seeds_2025[seeds_2025.Seed.str[0] == region_letter].copy()
    region_seeds = region_seeds.sort_values("SeedNum")

    # Build seed -> teamID lookup
    seed_to_id = dict(zip(region_seeds["SeedNum"], region_seeds["TeamID"]))

    # Round 1 matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    matchups_r1 = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]

    # Run each round and store results
    rounds = {}

    # Round of 64
    r1 = []
    for s1, s2 in matchups_r1:
        if s1 in seed_to_id and s2 in seed_to_id:
            winner = predict_game_silent(seed_to_id[s1], seed_to_id[s2])
            r1.append((seed_to_id[s1], seed_to_id[s2], winner))
        
    rounds["Round of 64"] = r1

    # Round of 32
    r2_teams = [w for (_,_,w) in r1]
    r2 = []
    for i in range(0, len(r2_teams), 2):
        if i+1 < len(r2_teams):
            winner = predict_game_silent(r2_teams[i], r2_teams[i+1])
            r2.append((r2_teams[i], r2_teams[i+1], winner))
    rounds["Round of 32"] = r2

    # Sweet 16
    r3_teams = [w for (_,_,w) in r2]
    r3 = []
    for i in range(0, len(r3_teams), 2):
        if i+1 < len(r3_teams):
            winner = predict_game_silent(r3_teams[i], r3_teams[i+1])
            r3.append((r3_teams[i], r3_teams[i+1], winner))
    rounds["Sweet 16"] = r3

    # Elite 8
    r4_teams = [w for (_,_,w) in r3]
    r4 = []
    if len(r4_teams) >= 2:
        winner = predict_game_silent(r4_teams[0], r4_teams[1])
        r4.append((r4_teams[0], r4_teams[1], winner))
    rounds["Elite 8"] = r4

    region_winner = r4[0][2] if r4 else None
    return rounds, region_winner


def print_bracket():
    region_letters = ["W", "X", "Y", "Z"]
    region_names   = {"W": "EAST", "X": "WEST", "Y": "SOUTH", "Z": "MIDWEST"}
    round_names    = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]

    final_four = []

    for letter in region_letters:
        rounds, region_winner = predict_region_bracket(letter)
        final_four.append(region_winner)

        print(f"\n{'═'*55}")
        print(f"  {region_names[letter]} REGION")
        print(f"{'═'*55}")

        for round_name in round_names:
            games = rounds.get(round_name, [])
            if not games:
                continue
            print(f"\n  ── {round_name} ──")
            for t1, t2, winner in games:
                t1_name = id_to_name.get(t1, str(t1))
                t2_name = id_to_name.get(t2, str(t2))
                w_name  = id_to_name.get(winner, str(winner))
                # Show loser greyed out with strikethrough-style and winner highlighted
                loser   = t1 if winner == t2 else t2
                l_name  = id_to_name.get(loser, str(loser))
                print(f"    {t1_name:<22} vs  {t2_name:<22}  →  ✓ {w_name}")

        if region_winner:
            print(f"\n  🏆 {region_names[letter]} WINNER: {id_to_name.get(region_winner, str(region_winner))}")

    # Final Four
    print(f"\n\n{'═'*55}")
    print("  FINAL FOUR")
    print(f"{'═'*55}\n")

    if len(final_four) >= 4 and all(t is not None for t in final_four):
        ff1 = predict_game_silent(final_four[0], final_four[1])
        ff2 = predict_game_silent(final_four[2], final_four[3])

        print(f"  {id_to_name.get(final_four[0], '?'):<22} vs  {id_to_name.get(final_four[1], '?'):<22}  →  ✓ {id_to_name.get(ff1, '?')}")
        print(f"  {id_to_name.get(final_four[2], '?'):<22} vs  {id_to_name.get(final_four[3], '?'):<22}  →  ✓ {id_to_name.get(ff2, '?')}")

        # Championship
        print(f"\n{'═'*55}")
        print("  CHAMPIONSHIP")
        print(f"{'═'*55}\n")
        champion = predict_game_silent(ff1, ff2)
        print(f"  {id_to_name.get(ff1, '?'):<22} vs  {id_to_name.get(ff2, '?'):<22}  →  ✓ {id_to_name.get(champion, '?')}")

        print(f"\n  🏆🏆 2025 PREDICTED CHAMPION: {id_to_name.get(champion, '?')} 🏆🏆\n")


print_bracket()