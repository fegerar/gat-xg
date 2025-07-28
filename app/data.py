from utils.dataset import download_github_directory

download_github_directory(
    repo_url="https://github.com/statsbomb/open-data",
    directory_path="data/events",
    local_path="../statsbomb",
    branch="master"
)

