"""
Visual grid rendering utilities for ARC puzzles.
Generates actual PNG images of grids with proper ARC color palette.
"""

from typing import List, Optional
import base64
from io import BytesIO

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ARC official color palette (RGB values)
ARC_COLOR_PALETTE = {
    0: (0, 0, 0),           # Black
    1: (0, 116, 217),       # Blue
    2: (255, 65, 54),       # Red
    3: (46, 204, 64),       # Green
    4: (255, 220, 0),       # Yellow
    5: (170, 170, 170),     # Gray
    6: (240, 18, 190),      # Magenta
    7: (255, 133, 27),      # Orange
    8: (127, 219, 255),     # Light Blue
    9: (135, 12, 37),       # Maroon
}

def render_grid_to_image(
    grid: List[List[int]],
    cell_size: int = 24,
    target_min_size: int = 512,
    target_max_size: int = 1024,
    dynamic_scale: bool = False,
) -> Optional[bytes]:
    """
    Render a grid as a PNG image with proper ARC colors.

    Args:
        grid: 2D list of integers (0-9)
        cell_size: Base size of each cell in pixels (default 24)
        target_min_size: Minimum desired dimension for the output image
        target_max_size: Maximum desired dimension for the output image
        dynamic_scale: If True, auto-scale to target_min_size/target_max_size.
            If False (default), keep fixed per-cell size across all grids.
    
    Returns:
        PNG image as bytes, or None if PIL not available
    """
    if not PIL_AVAILABLE:
        return None
    
    if not grid or not grid[0]:
        return None
    
    rows = len(grid)
    cols = len(grid[0])
    
    max_dim = max(rows, cols)
    if dynamic_scale:
        # Dynamic cell size calculation (legacy behavior)
        calculated_cell_size = cell_size
        if max_dim * calculated_cell_size < target_min_size:
            scale = target_min_size / (max_dim * calculated_cell_size)
            calculated_cell_size = int(calculated_cell_size * scale)
        if max_dim * calculated_cell_size > target_max_size:
            scale = target_max_size / (max_dim * calculated_cell_size)
            calculated_cell_size = int(calculated_cell_size * scale)
        cell_size = max(16, calculated_cell_size)
    else:
        # Fixed per-cell rendering so all puzzle grids share identical pixel size.
        cell_size = max(8, int(cell_size))
    
    # Create image
    img_width = cols * cell_size
    img_height = rows * cell_size
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            color_idx = grid[r][c]
            color = ARC_COLOR_PALETTE.get(color_idx, (128, 128, 128))
            
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Draw filled rectangle
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(50, 50, 50)) # Darker outline for better contrast
    
    # Convert to PNG bytes
    buffer = BytesIO()
    # Fireworks/Kimi has intermittent decode issues with some default Pillow PNG
    # compression streams. A low compression level produces broadly compatible PNGs.
    img.save(buffer, format='PNG', compress_level=1)
    return buffer.getvalue()


def render_grid_to_base64(
    grid: List[List[int]],
    cell_size: int = 24,
    dynamic_scale: bool = False,
) -> Optional[str]:
    """
    Render a grid as a base64-encoded PNG image.

    Args:
        grid: 2D list of integers (0-9)
        cell_size: Size of each cell in pixels (default 24)
        dynamic_scale: If True, auto-scale image dimensions; default False.

    Returns:
        Base64-encoded PNG string, or None if PIL not available
    """
    img_bytes = render_grid_to_image(
        grid,
        cell_size=cell_size,
        dynamic_scale=dynamic_scale,
    )
    if img_bytes is None:
        return None
    
    return base64.b64encode(img_bytes).decode('utf-8')



