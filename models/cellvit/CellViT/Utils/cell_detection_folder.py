import os
import subprocess
import argparse      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Python script for all folders under a parent folder.")
    parser.add_argument("--data_folder", required=True, help="Path to the parent folder.")
    parser.add_argument("--out_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--batch_size", default=2, help="Batch Size")
    parser.add_argument("--script_path", default="/home/KutumLabGPU/Documents/oralcancer/CellViT/cell_segmentation/inference/cell_detection.py", help="Path to the Python script to be executed.")
    args = parser.parse_args()

    data_folder = args.data_folder
    script_path = args.script_path
    output_folder = args.out_folder
    batch_size = args.batch_size

    error_paths = []

    for i, data_path in enumerate(sorted(os.listdir(data_folder))):
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
