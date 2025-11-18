import os
import shutil
import math

# Path to your directory that contains the 3 main folders
base_path = "C:\\Users\\Mahesh Porla\\Downloads\\foodproject"

# List your main folders
main_folders = ["./training_dataset", "./testing_dataset", "./validation_dataset"]

# Number of groups to create inside each main folder
num_groups = 11

for main_folder in main_folders:
    folder_path = os.path.join(base_path, main_folder)
    subfolders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

    total_subfolders = len(subfolders)
    group_size = total_subfolders // num_groups  # base number of folders per group
    remainder = total_subfolders % num_groups  # remaining folders (if not divisible)

    print(f"\nProcessing {main_folder}: {total_subfolders} folders -> {num_groups} groups")

    start_idx = 0
    for i in range(num_groups):
        group_name = f"Group_{i + 1}"
        group_path = os.path.join(folder_path, group_name)
        os.makedirs(group_path, exist_ok=True)

        # Distribute folders evenly; leftover goes to last group
        if i < num_groups - 1:
            end_idx = start_idx + group_size
        else:
            end_idx = total_subfolders  # last group gets the remaining folders (including 34th)

        group_subfolders = subfolders[start_idx:end_idx]

        for subfolder in group_subfolders:
            src = os.path.join(folder_path, subfolder)
            dest = os.path.join(group_path, subfolder)
            shutil.move(src, dest)

        print(f"  → {group_name}: {len(group_subfolders)} folders moved")
        start_idx = end_idx
