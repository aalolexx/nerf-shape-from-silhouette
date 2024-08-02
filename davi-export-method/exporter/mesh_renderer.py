import open3d as o3d
import json
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv
import math

# which data split we're using ('train', 'test', 'val')
data_split = 'val'

# Load the 3D mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/outputs/hook/alex-silhouette-model/2024-07-27_221252_l1_oSig_hook/exports/poisson_mesh.ply")
mesh.compute_vertex_normals()

# Set the mesh color to white
mesh.paint_uniform_color([1.0, 1.0, 1.0])  # RGB values for white

input_data_folder_path = "L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/data/working/binarized_images_lowres/hook/"
transforms_file_path = input_data_folder_path + "transforms_"+data_split+".json"
# Load the camera positions from a JSON file
with open(transforms_file_path, "r") as f:
    camera_data = json.load(f)

# Directory containing the corresponding original images
corresponding_images_folder = "L:/Master/TU/3_HTCV_Proj/nerf-shape-from-silhouette/data/working/binarized_images_lowres/hook/"
corresponding_images_folder = input_data_folder_path + data_split

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
opt.light_on = False  # Turn off lighting to avoid shading effects

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Get the view control to set camera parameters
ctr = vis.get_view_control()

zoom_level = 1 / np.tan(camera_angle_x / 2)


# Define a range of crop percentages to test
crop_rates = np.linspace(0, 0.4, 40)  # Test 30 rates from 0% to 30%
best_crop_rate_bin = []

# Define the standard resolution for IoU calculation
standard_resolution = (600, 600)

iou_bin = []
dice_bin = []
ssim_bin = []
psnr_bin = []

for idx, frame in enumerate(frames_short):
    transform_matrix = np.array(frame["transform_matrix"])

    # Extract the camera position and orientation
    camera_position = transform_matrix[:3, 3]
    front = transform_matrix[:3, 2]
    up = transform_matrix[:3, 1]
    lookat = camera_position + front

    # # Debug: Print the camera parameters
    # print(f"Frame {idx}:")
    # print(f"Camera Position: {camera_position}")
    # print(f"Front: {front}")
    # print(f"Up: {up}")
    # print(f"Lookat: {lookat}")

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

    # Load the corresponding image from the other folder
    corresponding_image_filename = f"r_{idx:03d}.png"
    corresponding_image_path = os.path.join(corresponding_images_folder, corresponding_image_filename)
    if not os.path.exists(corresponding_image_path):
        print(f"Corresponding image not found for idx {idx}: {corresponding_image_filename}")
        continue
    
    corresponding_image = Image.open(corresponding_image_path).convert('1')

    # Resize corresponding image to the standard resolution
    corresponding_image_resized = corresponding_image.resize(standard_resolution)
    corresponding_np = np.array(corresponding_image_resized)

    best_iou = 0
    best_crop_image = None
    best_crop_rate = 0

    # Iterate over different crop rates
    for crop_percentage in [.268]:
        # Calculate the crop box
        width, height = pil_image.size
        left = crop_percentage * width
        top = crop_percentage * height
        right = (1 - crop_percentage) * width
        bottom = (1 - crop_percentage) * height
        crop_box = (left, top, right, bottom)

        # Crop the image
        cropped_image = pil_image.crop(crop_box)

        # Resize the cropped image to the standard resolution
        cropped_image_resized = cropped_image.resize(standard_resolution)
        
        # Convert to binary format (1 for white, 0 for black)
        cropped_binary = cropped_image_resized.convert('1')
        cropped_np = np.array(cropped_binary)
        # Calculate IoU
        intersection = np.logical_and(cropped_np, corresponding_np).sum()
        union = np.logical_or(cropped_np, corresponding_np).sum()
        iou = intersection / union if union != 0 else 0

        if iou > best_iou:
            best_iou = iou
            best_crop_image = cropped_image_resized
            best_binary_cropped_np = cropped_np
            best_crop_rate = crop_percentage
            # print("best_binary_cropped_np")
            # print(best_binary_cropped_np[0][0])


            # Create overlay image
            overlay = np.zeros((*standard_resolution, 3), dtype=np.uint8)
            overlay[np.logical_and(cropped_np, corresponding_np)] = [255,255,255] #[0, 255, 0]   # White for overlap
            overlay[np.logical_and(cropped_np, np.logical_not(corresponding_np))] = [255, 0, 0]  # Red for FP
            overlay[np.logical_and(corresponding_np, np.logical_not(cropped_np))] = [0, 0, 255]  # Blue for FN
            best_overlay = Image.fromarray(overlay)

    ## calculate some more metrics
    
    # Convert images to numpy arrays for SSIM calculation
    best_crop_np = np.array(best_crop_image.convert('L'))
    corresponding_np_resized = np.array(corresponding_image_resized.convert('L'))

    # Calculate SSIM
    ssim_index = ssim(best_crop_np, corresponding_np_resized)
    print(f"SSIM for frame {idx}: {ssim_index:.4f}")

    # Calculate intersection and sum of areas
    # print(np.array(best_crop_image)[0][0])
    # print(corresponding_np[0][0])
    sum_of_areas = best_binary_cropped_np.sum() + corresponding_np.sum()
    # print(sum_of_areas)

    # Calculate Dice Coefficient
    dice_index = 2 * intersection / sum_of_areas if sum_of_areas != 0 else 0
    print(f"Dice Index for frame {idx}: {dice_index:.4f}")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((best_crop_np - corresponding_np_resized) ** 2)
    print(f"MSE for frame {idx}: {mse:.4f}")

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 10 * math.log10(max_pixel ** 2 / mse)
    print(f"PSNR for frame {idx}: {psnr:.4f}")

    best_crop_rate_bin.append(best_crop_rate)
    iou_bin.append(best_iou)
    dice_bin.append(dice_index)
    ssim_bin.append(ssim_index)
    psnr_bin.append(psnr)


    # Save the best cropped image
    if best_crop_image is not None:
        cropped_image_path = os.path.join(output_folder, f"frame_{idx}_cropped_best.png")
        best_crop_image.save(cropped_image_path)
        print(f"Saved best cropped image: {cropped_image_path}, Best IoU: {best_iou:.4f}, Best Crop Rate: {best_crop_rate:.2%}")
        
    # Save the overlay image
    if best_overlay is not None:
        overlay_image_path = os.path.join(output_folder, f"frame_{idx}_overlay.png")
        best_overlay.save(overlay_image_path)
        print(f"Saved overlay image: {overlay_image_path}")


    
    # # Save the current view as an image
    # image_path = os.path.join(output_folder, f"frame_{idx}.png")
    # vis.capture_screen_image(image_path)
    # print(f"Saved image: {image_path}")

    # Render and wait for user to press Enter before moving to the next frame
    print(f"Showing frame {idx}, press Enter in the console to continue to the next view...")
    # input()  # Wait for user input to proceed

print("best crop rates:")
print(best_crop_rate_bin)
print("average crop rate:")
print(np.mean(best_crop_rate_bin))
print("ious:")
# print(iou_bin)
print("average IoU:")
print(np.mean(iou_bin))
print("dices:")
# print(dice_bin)
print("average dice:")
print(np.mean(dice_bin))
print("ssims:")
# print(ssim_bin)
print("average SSIM:")
print(np.mean(ssim_bin))
print("psnrs:")
print(psnr_bin)
print("average PSNR:")
print(np.mean(psnr_bin))

# Define the output file path
output_scores_file = os.path.join(output_folder, "scores.csv")

# Write the data to a CSV file
with open(output_scores_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Index", "IoU", "Dice", "SSIM"])  # Header row
    for idx, (iou, dice, ssim) in enumerate(zip(iou_bin, dice_bin, ssim_bin)):
        writer.writerow([idx, f"{iou:.3f}", f"{dice:.3f}", f"{ssim:.3f}"])

print(f"Scores saved to {output_scores_file}")

# Destroy the window after all iterations
vis.destroy_window()