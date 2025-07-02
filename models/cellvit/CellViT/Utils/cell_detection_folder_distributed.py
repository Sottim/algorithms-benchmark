'''
    A new command-line argument --process_percent is added, which can take values 0 (for processing the first 50% of folders), 50 (for processing the last 50% of folders), or 100 (for processing all folders).
    The sorted list of directories sorted_dirs is obtained, and the total number of directories total_dirs is calculated.
    Based on the value of process_percent, the start_index and end_index for the loop iteration are determined. If process_percent is 0, start_index is 0, and end_index is the ceiling of total_dirs / 2. If process_percent is 50, start_index is the ceiling of total_dirs / 2, and end_index is total_dirs. If process_percent is 100 (the default), start_index is 0, and end_index is total_dirs.
    The loop iterates over the directories from start_index to end_index (excluding end_index), and the rest of the code remains the same.

'''


import os
import subprocess
import argparse
from math import ceil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Python script for all folders under a parent folder.")
    parser.add_argument("--data_folder", required=True, help="Path to the parent folder.")
    parser.add_argument("--out_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--batch_size", default=2, help="Batch Size")
    parser.add_argument("--script_path", default="/home/KutumLabGPU/Documents/oralcancer/CellViT/cell_segmentation/inference/cell_detection.py", help="Path to the Python script to be executed.")
    parser.add_argument("--process_percent", type=int, choices=[0, 50], default=100, help="Percentage of folders to process (0 for first 50%, 50 for last 50%, 100 for all folders)")

    args = parser.parse_args()

    data_folder = args.data_folder
    script_path = args.script_path
    output_folder = args.out_folder
    batch_size = args.batch_size
    process_percent = args.process_percent

    error_paths = []

    sorted_dirs = sorted(os.listdir(data_folder))
    total_dirs = len(sorted_dirs)

    if process_percent == 0:
        start_index = 0
        end_index = ceil(total_dirs / 2)
    elif process_percent == 50:
        start_index = ceil(total_dirs / 2)
        end_index = total_dirs
    else:
        start_index = 0
        end_index = total_dirs

    for i, data_path in enumerate(sorted_dirs[start_index:end_index], start=start_index):
        if os.path.isdir(os.path.join(data_folder, data_path)):
            script_command = f"python3 {script_path} --model ./CellViT-SAM-H-x40.pth --gpu 0 --batch_size {batch_size} process_patches --patch_path {os.path.join(data_folder, data_path)} --save_path {os.path.join(output_folder, data_path)}"

            # Run the command and capture the result
            result = subprocess.run(script_command, shell=True)

            # Check the exit code
            if result.returncode != 0:
                print(f"Error: Command failed with exit code {result.returncode}. Recording path.")
                error_paths.append(os.path.join(data_folder, data_path))

    print(error_paths)

    # Save the list of error paths to a text file
    with open("error_paths.txt", "w") as file:
        for path in error_paths:
            file.write(path + "\n")