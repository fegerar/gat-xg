"""
Simple training script for Progressive Graph Attention Networks (ProgressiveGAT).
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes import ProgressiveGraphSoccerDataset, ProgressiveGAT


def train_model(model, train_loader, val_loader, config):
    """Train the model with the given configuration."""
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    epochs = config['epochs']
    train_losses = []
    val_losses = []
    
    print(f"Training for {epochs} epochs on {device}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Use tqdm for training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for batch in train_pbar:
            # Move to device
            batch['xg'] = batch['xg'].to(device)
            if 'sequences' in batch:
                for i, sequence in enumerate(batch['sequences']):
                    for j, graph in enumerate(sequence):
                        batch['sequences'][i][j] = graph.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs['predictions'].squeeze(), batch['xg'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # Use tqdm for validation loop
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                # Move to device
                batch['xg'] = batch['xg'].to(device)
                if 'sequences' in batch:
                    for i, sequence in enumerate(batch['sequences']):
                        for j, graph in enumerate(sequence):
                            batch['sequences'][i][j] = graph.to(device)
                
                outputs = model(batch)
                loss = criterion(outputs['predictions'].squeeze(), batch['xg'])
                val_loss += loss.item()
                val_batches += 1
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses


def main():    
    # Default configuration
    config = {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 16,
        'device': 'auto'
    }
    
    
    with open('../config.json', 'r') as f:
        config.update(json.load(f))
    
    # Auto-detect device
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading dataset")
    
    # Load dataset
    dataset = ProgressiveGraphSoccerDataset(
        pickle_path='../dataset/processed_data.pkl',
        use_full_sequence=True,
        normalize_coordinates=True
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to avoid batch norm issues
        collate_fn=ProgressiveGraphSoccerDataset.collate_football_sequences
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True,  # Drop last incomplete batch to avoid batch norm issues
        collate_fn=ProgressiveGraphSoccerDataset.collate_football_sequences
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ProgressiveGAT(
        input_features=2,
        hidden_dim=16,  # Reduced from 64
        gat_layers=1,   # Reduced from 2 
        gat_heads=2,    # Reduced from 4
        sequence_hidden_dim=32,  # Reduced from 128
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'progressive_gat_model.pt')
    
    print("Model saved as 'progressive_gat_model.pt'")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
