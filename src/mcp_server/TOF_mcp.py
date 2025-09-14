from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import sys
import matplotlib.pyplot as plt
import io
import os
import numpy as np
import base64

# --- Physical Constants ---
# Mass of a Rubidium-87 atom in kg
MASS_RB87 = 1
# Boltzmann constant in J/K
BOLTZMANN_CONSTANT = 1

# Set up logging and MCP server
mcp = FastMCP()

def load_and_filter_data(radius_threshold=1.5):
    """
    Loads particle trajectory data (pos & vel) and filters based on a 3D radius.

    Args:
        radius_threshold (float): The maximum 3D radius for filtering.

    Returns:
        tuple: A tuple of filtered (x, y, z, vx, vy, vz) coordinate arrays.
               Returns (None, ..., None) if the file fails to load.
    """
    full_data_path = r"C:\Users\Buantum\Desktop\lucien_mcp\capture\saved_trajectories.npy"
    
    if not os.path.exists(full_data_path):
        print(f"Error: Data file not found.")
        return None, None, None, None, None, None

    try:
        data = np.load(full_data_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None, None, None

    # --- MODIFIED: Extract positions and velocities ---
    # Assuming columns are [x, y, z, vx, vy, vz, ...]
    x_pos, y_pos, z_pos = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
    
    total_points = len(x_pos)
    print(f"Loaded data for {total_points} particles.")

    # --- Filter data based on a 3D radius ---
    print(f"Filtering data based on 3D radius r < {radius_threshold}...")
    radius_squared = x_pos**2 + y_pos**2 + z_pos**2
    filter_mask = radius_squared < radius_threshold**2
    
    # Apply the same mask to all arrays
    filtered_x, filtered_y, filtered_z = x_pos[filter_mask], y_pos[filter_mask], z_pos[filter_mask]
    filtered_vx, filtered_vy, filtered_vz = vx[filter_mask], vy[filter_mask], vz[filter_mask]

    points_after_filter = len(filtered_x)
    
    if total_points > 0:
        percentage = points_after_filter / total_points * 100
        print(f"Filtering left {points_after_filter} data points (kept {percentage:.2f}%).")
    
    return filtered_x, filtered_y, filtered_z, filtered_vx, filtered_vy, filtered_vz

def plot_density_map(x_data, y_data, xlabel, ylabel, title):
    """
    Generates a 2D column density plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    print(f"Generating density plot for: {title}")
    BINS = 80
    _, _, _, im = ax.hist2d(x_data, y_data, bins=BINS, cmap='viridis', cmin=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Counts per Bin (Column Density)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    print("Plotting complete.")
    return fig, ax

@mcp.tool()
def capture(imaging_mode: str) -> ImageContent:
    """
    Captures the state of the atoms and generates an image based on the specified imaging mode.

    Args:
        imaging_mode (str): 'horizontal' for Y-Z plane or 'vertical' for X-Y plane.

    Returns:
        ImageContent: The generated image.
    """
    # MODIFIED: Unpack the new return values, ignoring velocities with '_'
    filtered_x, filtered_y, filtered_z, _, _, _ = load_and_filter_data(radius_threshold=1.5)

    if filtered_x is None:
        raise ValueError("Failed to load or filter data. Cannot generate image.")

    if imaging_mode == 'horizontal':
        x_plot, y_plot = filtered_y, filtered_z
        xlabel, ylabel, title = 'Y Position', 'Z Position', 'Column Density in Y-Z Plane (Horizontal)'
    elif imaging_mode == 'vertical':
        x_plot, y_plot = filtered_x, filtered_y
        xlabel, ylabel, title = 'X Position', 'Y Position', 'Column Density in X-Y Plane (Vertical)'
    else:
        raise ValueError(f"Invalid imaging_mode: '{imaging_mode}'. Use 'horizontal' or 'vertical'.")

    fig, _ = plot_density_map(x_plot, y_plot, xlabel, ylabel, title)
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return ImageContent(type="image", data=img_base64, mimeType="image/png")

@mcp.tool()
def calculate_atom_properties() -> TextContent:
    """
    Calculates the number of atoms and temperature from the filtered data.

    Returns:
        TextContent: A string containing the calculated properties.
    """
    # Step 1: Load all filtered data, including velocities
    data_arrays = load_and_filter_data(radius_threshold=1.5)
    filtered_x, _, _, filtered_vx, filtered_vy, filtered_vz = data_arrays

    if filtered_x is None:
        return TextContent(type="text", text="Error: Could not load data to calculate properties.")

    # Step 2: Calculate atom number
    atom_number = len(filtered_x)
    if atom_number == 0:
        return TextContent(type="text", text="Atom Number: 0\nTemperature: N/A (no atoms)")

    # Step 3: Calculate temperature
    # T = (m * <v^2>) / (3 * kB)
    # <v^2> is the mean of the squared speeds
    squared_speeds = filtered_vx**2 + filtered_vy**2 + filtered_vz**2
    mean_squared_speed = np.mean(squared_speeds)
    
    temperature_kelvin = (MASS_RB87 * mean_squared_speed) / (3 * BOLTZMANN_CONSTANT)
    
    # Convert to a more readable unit, e.g., microkelvins (uK)
    temperature_microkelvin = temperature_kelvin 

    # Step 4: Format the output string
    result_string = (
        f"**Atom Cloud Properties**\n"
        f"---------------------------\n"
        f"Filtered Atom Number: {atom_number}\n"
        f"Temperature: {temperature_microkelvin:.2f} µK"
    )
    
    return TextContent(type="text", text=result_string)

# --- Main application logic ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Starts the MCP server."""
    logger.info('Starting QuantumSim MCP server...')
    mcp.run('streamable-http')

if __name__ == "__main__":
    main()