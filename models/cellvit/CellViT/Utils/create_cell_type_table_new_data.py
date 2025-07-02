import json
import pandas as pd
import os
import argparse
import warnings

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data
        
def create_table(data, label=None, patient_id=None):
    stats = pd.DataFrame()
    for patch, instances in data.items():
        patch_parts = patch.split("-")
        patient_id = f"{patient_id}" 
        patch_name = f"{patch}"
        stat = {
            "Paitent-ID": patient_id,
            "Patch": patch_name,
            "Label": label,
            "Neoplastic": 0,
            "Inflammatory": 0,
            "Connective": 0,
            "Dead": 0,
            "Epithelial": 0,
            "UNK": 0,
        }
        for instance in instances:
            stat[data[patch][instance]["type"]] += 1
        if label is None:
            stat.pop("Label")
        stat_df = pd.DataFrame(stat, index=['Patch'])
        stats = pd.concat([stats, stat_df], ignore_index=True)
        stats.reset_index()

    return stats


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", type=str, default=None, help="Path to Parent Folder")
    parser.add_argument("--json_file", type=str, default=None, help="Path to Json File")
    parser.add_argument("--combine", type=bool, default=False)

    args = parser.parse_args()
    parent_folder = args.parent_folder

    if args.combine == True:
        collated_df = pd.DataFrame()

    if parent_folder is not None:
        folders = sorted(os.listdir(parent_folder))
        
        for folder in folders:
            if os.path.isdir(os.path.join(parent_folder, folder)):
                json_path = os.path.join(parent_folder, folder, "instance_segmentation.json")
                print(json_path)
                type_ = parent_folder.split("/")[-2]
                df = create_table(load_json(json_path), label=type_, patient_id=folder)
                if args.combine == True:
                    collated_df = pd.concat([collated_df, df], ignore_index=True)
                    collated_df.reset_index()
                
                else:
                    save_path = os.path.join(parent_folder, folder, f"cell_stats_{parent_folder.split('/')[-1]}.csv")
                    df.to_csv(save_path, index=False)

        if args.combine == True:
            save_path = os.path.join(parent_folder, f"cell_stats_{parent_folder.split('/')[-1]}_combined.csv")
            collated_df.to_csv(save_path, index=False)
            print(f"File save to {save_path}")
    
    elif args.json_file is not None:
        df = create_table(load_json(args.json_file))
        dir_name = os.path.dirname(args.json_file)
        file_name = args.json_file.split("/")[-1][:-4] + ".csv"
        save_path = os.path.join(dir_name, file_name)
        df.to_csv(save_path, index=False)
