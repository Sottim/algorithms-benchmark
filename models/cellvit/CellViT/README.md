## Code Setup & Run

To fix the dependencies run the following command under the project-root directory:

```
conda env create -f environment.yml
```
This will create an environment called `cellvit_env`. Activate this environment using the following command:
```
conda activate cellvit_env
```
Now download the checkpoint `CellViT-SAM-H-x40.pth` following the guidelines from [CellViT Repo](https://github.com/TIO-IKIM/CellViT/) and put it under the project-root directory.

To run the inference use the following command:
```
python /home/KutumLabGPU/Documents/oralcancer/CellViT/cell_segmentation/inference/cell_detection.py --model ./CellViT-SAM-H-x40.pth --gpu 0 --batch_size 2 process_patches --patch_path <dir_containing_images> --save_path <ouput_path> --is_stain_normalize 1
```
Pass `1` to  the flag `--is_stain_normalize` to apply stain normalization or `0` otherwise.

## Directory Structure
```
project-root/
├── input_dir/
│   ├── image_1.png
│   ├── image_2.png
│   ├── ...
│   └── image_n.png
├── environment.yml
├── ouput_dir/
    ├── mask_barplots/
    │    ├── image_1.png
    │    ├── image_1.png
    │    ├── ...
    │    └── image_n.png
    │    
    ├── pred_binary_mask/
    │        ├── image_1.png
    │        ├── image_1.png
    │        ├── ...
    │        └── image_n.png
    ├── pred_semantic_mask/
    │            ├── image_1.png
    │            ├── image_1.png
    │            ├── ...
    │            └── image_n.png
    ├── instance_segmentation.josn
    ├── cell_embeddings.h5
```
`mask_barplots: barplots of cell distribution per patch from inference`

`pred_binary_mask: instance segmentation masks`

`pred_semantic_mask: semantic segmentation masks`

`instance_segmentation.josn: information about the masks in json format and also contains probability distribution of predictions per cells`

`cell_embedding.h5: contails cell embedding vectors from CellViT encoder`

For more information about CellViT refer to ther official [repo](https://github.com/TIO-IKIM/CellViT/).
