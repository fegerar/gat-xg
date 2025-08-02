import tqdm
import os
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
statsbomb_files = [f"../statsbomb/{f}" for f in os.listdir(statsbomb_dir) if f.endswith('.json')][2] # just for testing

# pbar = tqdm.tqdm(statsbomb_files, desc="Processing StatsBomb files", unit="file")
# for file in pbar:
#     try:
#         file_path = os.path.join(statsbomb_dir, file)
#         if os.path.isfile(file_path):
#             possessions_t, average_pass_t = game2graphs(file_path)
#             possessions += possessions_t
#             average_pass += average_pass_t
#             pbar.set_postfix(file=file)
#     except:
#         failed += 1

game2graphs(statsbomb_files)
