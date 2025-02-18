import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

# Define the color map
digit_to_word_plus = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
arc_color_map = LinearSegmentedColormap.from_list(
    name='arc_colors',
    colors=digit_to_word_plus,
)

def generate_input_type_ids_multi(grid, visualize=False):
    """
    Generates input_type_ids for the given grid based on detected object contours.
    Automatically handles multiple object colors using OpenCV contour detection.

    Params:
    - grid: 2D numpy array (H x W) representing the grid.
    - visualize: If True, intermediate steps will be displayed using matplotlib.

    Returns:
    - input_type_ids: 2D numpy array of the same shape as grid, where each detected
      object is assigned a unique ID starting from 1, while background remains 0.
    """
    input_type_ids = np.zeros_like(grid, dtype=np.int32)
    current_object_id = 1  # Start object IDs from 1

    # Identify the most frequent color in the grid as background
    background_value = np.bincount(grid.flatten()).argmax()

    # Get unique values except the assumed background
    unique_values = np.unique(grid)
    unique_values = unique_values[unique_values != background_value]

    for value in unique_values:
        # Create a binary mask: 255 for cells with 'value', 0 otherwise
        binary_image = np.where(grid == value, 255, 0).astype(np.uint8)

        if visualize:
            plt.figure()
            plt.title(f'Binary Image for Object Value {value}')
            plt.imshow(binary_image, cmap='gray')
            plt.show()

        # Find contours of these regions
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if visualize:
            # Draw contours on a blank image for visualization
            contour_image = np.zeros_like(binary_image)
            cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
            plt.figure()
            plt.title(f'Contours for Object Value {value}')
            plt.imshow(contour_image, cmap='gray')
            plt.show()

        # Assign unique object IDs
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Keep bounding box within grid bounds
            x_end = min(x + w, input_type_ids.shape[1])
            y_end = min(y + h, input_type_ids.shape[0])

            # Assign a unique ID for all pixels in this bounding box
            input_type_ids[y:y_end, x:x_end] = current_object_id
            current_object_id += 1

    if visualize:
        plt.figure()
        plt.title('input_type_ids')
        plt.imshow(input_type_ids, cmap='tab20')
        plt.colorbar()
        plt.show()

    return input_type_ids

def paint_grid_with_boxes(grid, input_type_ids):
    """
    Visualize the grid with bounding boxes drawn around detected objects.

    Params:
    - grid: 2D numpy array (H x W) representing the grid (0..9).
    - input_type_ids: 2D numpy array of same shape, containing object IDs >=1.
    """
    df = pd.DataFrame(grid)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df, annot=True, fmt="d", linewidths=.5,
        xticklabels=False, yticklabels=False, cbar=False,
        cmap=arc_color_map, vmin=0, vmax=9
    )

    object_ids = np.unique(input_type_ids)
    object_ids = object_ids[object_ids != 0]  # Exclude background=0

    for obj_id in object_ids:
        positions = np.argwhere(input_type_ids == obj_id)
        min_row, min_col = positions.min(axis=0)
        max_row, max_col = positions.max(axis=0)

        rect = patches.Rectangle(
            (min_col, min_row),
            max_col - min_col + 1,
            max_row - min_row + 1,
            linewidth=2,
            edgecolor=digit_to_word_plus[obj_id % len(digit_to_word_plus)],
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()

def test_opencv_contour():
    """
    Example usage to demonstrate the generate_input_type_ids_multi function.
    """
    grid = np.array([
        [5, 5, 5, 5, 5, 5],
        [5, 6, 6, 5, 8, 5],
        [5, 5, 8, 8, 8, 5],
        [5, 5, 8, 5, 8, 5],
        [5, 5, 5, 5, 5, 5]
    ], dtype=np.int32)

    input_type_ids = generate_input_type_ids_multi(grid, visualize=True)
    paint_grid_with_boxes(grid, input_type_ids)
