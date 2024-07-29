import open3d as o3d
import json
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv

# which data split we're using ('train', 'test', 'val')
data_split = 'val'

# Load the 3D mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/outputs/hook/alex-silhouette-model/2024-07-27_172756_l1_pure_hook/exports/poisson_mesh.ply")
# mesh = o3d.io.read_triangle_mesh("L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/outputs/hook/alex-silhouette-model/2024-07-27_221252_l1_oSig_hook/exports/poisson_mesh.ply")
mesh.compute_vertex_normals()

# Set the mesh color to white
mesh.paint_uniform_color([1.0, 1.0, 1.0])  # RGB values for white

transforms_folder_path = "L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/data/working/binarized_images_lowres/hook/"
transforms_file_path = transforms_folder_path + "transforms_"+data_split+".json"
# Load the camera positions from a JSON file
with open(transforms_file_path, "r") as f:
    camera_data = json.load(f)


# Create a folder for saving images if it doesn't exist
output_folder = "export_mesh_renders/" + data_split
os.makedirs(output_folder, exist_ok=True)


# Extract camera frames from JSON
frames = camera_data["frames"]
frames_short = frames#[:3]
camera_angle_x = camera_data["camera_angle_x"]


# Set desired window size
window_width = 1200
window_height = 1200

# Create a visualizer window once
vis = o3d.visualization.Visualizer()
vis.create_window(width=window_width, height=window_height)

# Set the background color to black
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # RGB values for black
# opt.light_on = False  # Turn off lighting to avoid shading effects

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Get the view control to set camera parameters
ctr = vis.get_view_control()


crop_percentage = .268

for idx, frame in enumerate(frames_short):
    transform_matrix = np.array(frame["transform_matrix"])

    # Extract the camera position and orientation
    camera_position = transform_matrix[:3, 3]
    front = transform_matrix[:3, 2]
    up = transform_matrix[:3, 1]
    lookat = camera_position + front

    # Set the view control parameters
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(.1)  # Set zoom level based on FOV

    # Update the geometry and renderer
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()


    # Capture the current view as an image
    image = vis.capture_screen_float_buffer(False)
    image_np = np.asarray(image)
    image_np = (255 * image_np).astype(np.uint8)  # Convert to uint8

    # Convert to PIL Image
    pil_image = Image.fromarray(image_np)

    # Calculate the crop box
    width, height = pil_image.size
    left = crop_percentage * width
    top = crop_percentage * height
    right = (1 - crop_percentage) * width
    bottom = (1 - crop_percentage) * height
    crop_box = (left, top, right, bottom)

    # Crop the image
    cropped_image = pil_image.crop(crop_box)

    # Save the current view as an image
    image_path = os.path.join(output_folder, f"frame_{idx}_shaded_pure.png")
    cropped_image.save(image_path)
    print(f"Saved image: {image_path}")

    # Render and wait for user to press Enter before moving to the next frame
    print(f"Showing frame {idx}, press Enter in the console to continue to the next view...")
    input()  # Wait for user input to proceed



# Destroy the window after all iterations
vis.destroy_window()