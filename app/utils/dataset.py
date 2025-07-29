import json
import subprocess
import os
import shutil
from pathlib import Path
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt


def download_github_directory(repo_url, directory_path, local_path="downloaded_dir", branch="main"):
    """
    Download large directory using git sparse-checkout (handles >1000 files).
    """
    temp_repo = "temp_repo_" + str(hash(repo_url))[:8]
    original_dir = os.getcwd()
    
    try:
        # Clean up any existing temp directory
        if os.path.exists(temp_repo):
            shutil.rmtree(temp_repo)
        
        print(f"Cloning repository...")
        # Clone with no checkout and minimal history
        subprocess.run([
            "git", "clone", 
            "--filter=blob:none",  # Don't download file contents initially
            "--no-checkout", 
            "--single-branch", 
            "--branch", branch,
            repo_url, 
            temp_repo
        ], check=True, capture_output=True, text=True)
        
        # Change to repo directory
        os.chdir(temp_repo)
        
        # Enable sparse checkout
        subprocess.run(["git", "config", "core.sparseCheckout", "true"], check=True)
        
        # Specify which directory to checkout
        sparse_checkout_path = Path(".git/info/sparse-checkout")
        with open(sparse_checkout_path, "w") as f:
            f.write(f"{directory_path}/\n")
        
        print(f"Checking out directory: {directory_path}")
        # Checkout the files
        subprocess.run(["git", "checkout"], check=True)
        
        # Move back to original directory
        os.chdir(original_dir)
        
        # Copy the directory to final location
        source_dir = os.path.join(temp_repo, directory_path)
        if os.path.exists(source_dir):
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.copytree(source_dir, local_path)
            
            # Count files
            file_count = sum(1 for _ in Path(local_path).rglob('*') if _.is_file())
            print(f"Successfully downloaded {file_count} files to: {local_path}")
        else:
            print(f"Directory '{directory_path}' not found in repository")
            
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure we're back in original directory
        os.chdir(original_dir)
        # Cleanup
        if os.path.exists(temp_repo):
            shutil.rmtree(temp_repo)


def get_event_by_id(events, id):
    return [event for event in events if event['id'] == id][0]
     

def split_ball_possessions(events):
    """
    Split ball possession events into separate sequences.
    """
    possessions = []
    current_possession = []
    current_team = None
    
    # Process ALL events in chronological order
    for event in events:
        event_type = event['type']['name']
        team = event['team']['name']
        
        # Check if this event changes possession
        if current_team is None:
            current_team = team
        
        if team != current_team:
            # Possession changed - save current possession
            if current_possession:
                possessions.append({
                    'possession': current_possession,
                    'xg': float(get_event_by_id(events, current_possession[-1]["pass"]["assisted_shot_id"])["shot"]["statsbomb_xg"]) if "shot_assist" in current_possession[-1]["pass"] else 0
                })
            current_possession = []
            current_team = team
        
        # Add pass events to current possession
        if event_type == 'Pass':
            current_possession.append(event)
    
    # Don't forget the last possession
    if current_possession:
        possessions.append({
            'possession': current_possession,
            'xg': float(get_event_by_id(events, current_possession[-1]["pass"]["assisted_shot_id"])["shot"]["statsbomb_xg"]) if "shot_assist" in current_possession[-1]["pass"] else 0
        })
    
    return possessions

def game2graphs(file_path):
    """
    Create a graph representation from game data.
    """
    # edge_index = torch.tensor()
    # edge_features = torch.tensor()
    # x = torch.tensor()
    with open(file_path, "r") as f:
        game = json.load(f)
        poss = split_ball_possessions(game)
        count_shot = 0
        for pos in poss:
            if pos["xg"] > 0.0:
                count_shot += 1
    
        return len(poss), count_shot



