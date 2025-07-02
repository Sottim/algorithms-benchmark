import subprocess

types_ = ["NORMAL", "OSMF", "OSCC/WD", "OSCC/MD", "OSCC/PD"]
splits_ = ["train", "test", "validation"]

for split in splits_:
    for type_ in types_:
        command = f"python Utils/cell_detection_folder.py --data_folder=/media/KutumLabGPU/split_data_png_new_padded/{split}/{type_} --out_folder=/media/KutumLabGPU/split_data_png_new_padded_out_latest_dist/{split}/{type_}"
        subprocess.run(command, shell=True)
