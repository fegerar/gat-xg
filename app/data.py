import tqdm
import os
import pickle
from utils.dataset import download_github_directory, game2graphs

# Download if not already present
if not os.path.exists("../statsbomb"):
    download_github_directory(
        repo_url="https://github.com/statsbomb/open-data",
        directory_path="data/events",
        local_path="../statsbomb",
        branch="master"
    )

# List all files in the directory
statsbomb_dir = "../statsbomb"
statsbomb_files = [f"../statsbomb/{f}" for f in os.listdir(statsbomb_dir) if f.endswith('.json')]

# Collect all processed data
all_games_data = []

pbar = tqdm.tqdm(statsbomb_files, desc="Processing StatsBomb files", unit="file")
for file in pbar:
    game_id = os.path.basename(file).replace('.json', '')
    graphs_data = game2graphs(file)
    
    # Add game_id to each possession sequence
    for possession_data in graphs_data:
        possession_data['game_id'] = game_id
    
    all_games_data.extend(graphs_data)


print(f"Processed {len(all_games_data)} possession sequences from {len(statsbomb_files)} games")
pickle.dump(all_games_data, open("../dataset/processed_data.pkl", "wb"))



