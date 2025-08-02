import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider


def visualize_possession_graphs(file_path, possession_idx=0):
    """
    Create an interactive visualization of progressive graphs from a game file.
    
    Args:
        file_path: Path to the StatsBomb JSON file
        possession_idx: Index of the possession to visualize (default: 0 for first possession)
    
    Returns:
        list: List of progressive graphs for the selected possession
    """
    # Import here to avoid circular imports
    from .dataset import (split_ball_possessions, select_only_shot_possessions, 
                         progressive_graphs, get_event_by_id)
    
    with open(file_path, "r") as f:
        events = json.load(f)
        possessions = select_only_shot_possessions(split_ball_possessions(events))

    # Create interactive visualization with slider
    if possessions and possession_idx < len(possessions):
        print(f"Available possessions: {len(possessions)}")
        print(f"Visualizing possession {possession_idx + 1}")
        
        selected_possession = possessions[possession_idx]
        graphs = progressive_graphs(selected_possession)
        
        # Get shot information for visualization
        shot_info = None
        last_pass = selected_possession['possession'][-1]
        if 'pass' in last_pass and 'assisted_shot_id' in last_pass['pass']:
            try:
                shot_event = get_event_by_id(events, last_pass['pass']['assisted_shot_id'])
                shot_location = shot_event.get('location')
                shot_data = shot_event.get('shot', {})
                shot_end_location = shot_data.get('end_location')
                
                if shot_location and shot_end_location:
                    shot_info = {
                        'start': shot_location,
                        'end': [shot_end_location[0], shot_end_location[1]],  # Only x, y (ignore z)
                        'outcome': shot_data.get('outcome', {}).get('name', 'Unknown'),
                        'xg': shot_data.get('statsbomb_xg', 0)
                    }
            except:
                pass
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Initialize the plot
        def update_plot(graph_idx):
            ax.clear()
            
            if graph_idx < len(graphs):
                graph = graphs[graph_idx]
                
                # Draw football pitch
                _draw_football_pitch(ax)
                
                if graph.x.shape[0] > 0:
                    # Plot nodes (passes)
                    ax.scatter(graph.x[:, 0], graph.x[:, 1], 
                             c='red', s=100, zorder=5, alpha=0.8, 
                             edgecolors='darkred', linewidth=2)
                    
                    # Add node numbers and coordinates
                    for i, (x, y) in enumerate(graph.x.numpy()):
                        ax.annotate(f'{i+1}\n({x:.0f},{y:.0f})', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=9, 
                                  fontweight='bold', color='darkred',
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    # Plot edges (pass connections)
                    if graph.edge_index.shape[1] > 0:
                        for edge in graph.edge_index.t().numpy():
                            start_pos = graph.x[edge[0]].numpy()
                            end_pos = graph.x[edge[1]].numpy()
                            ax.annotate('', xy=end_pos, xytext=start_pos,
                                      arrowprops=dict(arrowstyle='->', color='blue', 
                                                    lw=2, alpha=0.7))
                
                # Show shot direction only on the final graph (when possession is complete)
                if shot_info and graph_idx == len(graphs) - 1:
                    _draw_shot_visualization(ax, graph, shot_info)
                
                # Set plot properties
                _set_plot_properties(ax, graph_idx, len(graphs), selected_possession, shot_info)
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Graph', 0, len(graphs)-1, 
                       valinit=0, valfmt='%d', valstep=1)
        
        # Update function for slider
        def update(val):
            graph_idx = int(slider.val)
            update_plot(graph_idx)
            plt.draw()
        
        slider.on_changed(update)
        
        # Initial plot
        update_plot(0)
        
        plt.show()
        
        return graphs
    
    return []


def explore_possessions(file_path):
    """
    Helper function to explore all shot possessions in a game file.
    Shows a summary of all possessions and allows selection.
    
    Args:
        file_path: Path to the StatsBomb JSON file
    
    Returns:
        list: List of all shot possessions
    """
    # Import here to avoid circular imports
    from .dataset import split_ball_possessions, select_only_shot_possessions, get_event_by_id
    
    with open(file_path, "r") as f:
        events = json.load(f)
        possessions = select_only_shot_possessions(split_ball_possessions(events))
    
    print(f"Found {len(possessions)} shot possessions in the game:")
    print("-" * 60)
    
    for i, possession in enumerate(possessions):
        num_passes = len(possession['possession'])
        xg = possession['xg']
        
        # Get shot information
        last_pass = possession['possession'][-1]
        shot_outcome = "Unknown"
        if 'pass' in last_pass and 'assisted_shot_id' in last_pass['pass']:
            try:
                shot_event = get_event_by_id(events, last_pass['pass']['assisted_shot_id'])
                shot_outcome = shot_event.get('shot', {}).get('outcome', {}).get('name', 'Unknown')
            except:
                pass
        
        print(f"Possession {i+1:2d}: {num_passes:2d} passes, XG: {xg:.3f}, Outcome: {shot_outcome}")
    
    print("-" * 60)
    print("Use visualize_possession_graphs(file_path, possession_idx) to visualize a specific possession")
    print("Example: visualize_possession_graphs('file.json', 0) for the first possession")
    
    return possessions


def _draw_football_pitch(ax):
    """
    Draw a football pitch on the given axis.
    
    Args:
        ax: Matplotlib axis to draw on
    """
    # Draw football pitch outline
    pitch = patches.Rectangle((0, 0), 120, 80, linewidth=2, 
                            edgecolor='green', facecolor='lightgreen', alpha=0.3)
    ax.add_patch(pitch)
    
    # Draw center line
    ax.axvline(x=60, color='white', linewidth=2)
    
    # Draw penalty areas
    penalty_left = patches.Rectangle((0, 18), 18, 44, linewidth=2, 
                                   edgecolor='white', facecolor='none')
    penalty_right = patches.Rectangle((102, 18), 18, 44, linewidth=2, 
                                    edgecolor='white', facecolor='none')
    ax.add_patch(penalty_left)
    ax.add_patch(penalty_right)
    
    # Draw goal areas
    goal_left = patches.Rectangle((0, 30), 6, 20, linewidth=2, 
                                edgecolor='white', facecolor='none')
    goal_right = patches.Rectangle((114, 30), 6, 20, linewidth=2, 
                                 edgecolor='white', facecolor='none')
    ax.add_patch(goal_left)
    ax.add_patch(goal_right)
    
    # Draw goals
    ax.plot([0, 0], [36, 44], color='white', linewidth=4)  # Left goal
    ax.plot([120, 120], [36, 44], color='white', linewidth=4)  # Right goal


def _draw_shot_visualization(ax, graph, shot_info):
    """
    Draw shot visualization including shot location, direction, and movement.
    
    Args:
        ax: Matplotlib axis to draw on
        graph: Current graph with pass information
        shot_info: Dictionary containing shot information
    """
    # Plot shot starting position
    ax.scatter(shot_info['start'][0], shot_info['start'][1], 
             c='orange', s=150, marker='*', zorder=6, 
             edgecolors='darkorange', linewidth=2, 
             label='Shot Location')
    
    # Connect last pass to shot location if they're different
    if graph.x.shape[0] > 0:
        last_pass_pos = graph.x[-1].numpy()  # Last pass position
        shot_start_pos = shot_info['start']
        
        # Only draw connection if shot is from a different location than last pass
        distance = ((last_pass_pos[0] - shot_start_pos[0])**2 + 
                   (last_pass_pos[1] - shot_start_pos[1])**2)**0.5
        
        if distance > 5:  # Only if shot is more than 5 units away from last pass
            ax.annotate('', 
                      xy=shot_start_pos, 
                      xytext=last_pass_pos,
                      arrowprops=dict(
                          arrowstyle='->', 
                          color='purple',
                          lw=2, 
                          alpha=0.6,
                          linestyle='--'
                      ),
                      zorder=4)
            ax.text((last_pass_pos[0] + shot_start_pos[0])/2,
                   (last_pass_pos[1] + shot_start_pos[1])/2 - 2,
                   'Move to shot', fontsize=8, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    # Plot shot direction arrow
    ax.annotate('', 
              xy=shot_info['end'], 
              xytext=shot_info['start'],
              arrowprops=dict(
                  arrowstyle='->', 
                  color='red' if shot_info['outcome'] == 'Goal' else 'orange',
                  lw=4, 
                  alpha=0.9,
                  connectionstyle="arc3,rad=0"
              ),
              zorder=7)
    
    # Add shot end marker
    end_color = 'green' if shot_info['outcome'] == 'Goal' else 'red'
    ax.scatter(shot_info['end'][0], shot_info['end'][1], 
             c=end_color, s=100, marker='X', zorder=6, 
             edgecolors='darkred', linewidth=2)
    
    # Add shot info text with coordinates for debugging
    ax.text(shot_info['start'][0], shot_info['start'][1] - 4, 
           f"Shot: {shot_info['outcome']}\n({shot_info['start'][0]:.1f}, {shot_info['start'][1]:.1f})", 
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))


def _set_plot_properties(ax, graph_idx, total_graphs, possession, shot_info):
    """
    Set plot properties including axis limits, labels, title, etc.
    
    Args:
        ax: Matplotlib axis
        graph_idx: Current graph index
        total_graphs: Total number of graphs
        possession: Possession data
        shot_info: Shot information dictionary
    """
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)
    ax.set_xlabel('X Coordinate (pitch length)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pitch width)', fontsize=12)
    
    # Enhanced title with shot information
    title = f'Progressive Graph {graph_idx + 1}/{total_graphs} - Possession with {len(possession["possession"])} passes (XG: {possession["xg"]:.3f})'
    if shot_info and graph_idx == total_graphs - 1:
        title += f'\nShot: {shot_info["outcome"]} (XG: {shot_info["xg"]:.3f})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend if shot is shown
    if shot_info and graph_idx == total_graphs - 1:
        ax.legend(loc='upper left', fontsize=10)


def create_simple_possession_plot(file_path, possession_idx=0):
    """
    Create a simple static plot of a possession without interactive elements.
    Useful for debugging or quick visualization.
    
    Args:
        file_path: Path to the StatsBomb JSON file
        possession_idx: Index of the possession to visualize
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Import here to avoid circular imports
    from .dataset import (split_ball_possessions, select_only_shot_possessions, 
                         get_event_by_id)
    
    with open(file_path, "r") as f:
        events = json.load(f)
        possessions = select_only_shot_possessions(split_ball_possessions(events))
    
    if not possessions or possession_idx >= len(possessions):
        print(f"No possession found at index {possession_idx}")
        return None
    
    possession = possessions[possession_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw pitch
    _draw_football_pitch(ax)
    
    # Plot passes
    for i, pass_event in enumerate(possession['possession']):
        location = pass_event.get('location', [])
        if len(location) >= 2:
            ax.scatter(location[0], location[1], c='red', s=100, zorder=5)
            ax.annotate(f'{i+1}', (location[0], location[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Connect passes
    locations = [event.get('location', []) for event in possession['possession'] 
                if len(event.get('location', [])) >= 2]
    
    for i in range(len(locations) - 1):
        start = locations[i]
        end = locations[i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, alpha=0.7)
    
    # Add shot information
    last_pass = possession['possession'][-1]
    if 'pass' in last_pass and 'assisted_shot_id' in last_pass['pass']:
        try:
            shot_event = get_event_by_id(events, last_pass['pass']['assisted_shot_id'])
            shot_location = shot_event.get('location')
            shot_data = shot_event.get('shot', {})
            shot_end_location = shot_data.get('end_location')
            
            if shot_location and shot_end_location:
                # Shot start
                ax.scatter(shot_location[0], shot_location[1], 
                         c='orange', s=150, marker='*', zorder=6)
                
                # Shot end
                end_color = 'green' if shot_data.get('outcome', {}).get('name') == 'Goal' else 'red'
                ax.scatter(shot_end_location[0], shot_end_location[1], 
                         c=end_color, s=100, marker='X', zorder=6)
                
                # Shot arrow
                ax.arrow(shot_location[0], shot_location[1], 
                        shot_end_location[0] - shot_location[0], 
                        shot_end_location[1] - shot_location[1],
                        head_width=2, head_length=2, fc='orange', ec='orange', linewidth=3)
        except:
            pass
    
    # Set properties
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)
    ax.set_xlabel('X Coordinate (pitch length)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pitch width)', fontsize=12)
    ax.set_title(f'Possession {possession_idx + 1} - {len(possession["possession"])} passes (XG: {possession["xg"]:.3f})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig
