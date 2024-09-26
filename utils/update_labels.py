import os
import SimpleITK as sitk
import numpy as np
"""
description: This file is used to convert the original labels to the form of labels that accepted by the new version (nnU-Net v2) of the network.
author: Yifan HE
date: 2024/9/20
"""

def update_labels(input_file: str, output_file: str) -> None:
    """
    Update labels[0, 1, 2, 4] to [0, 1, 2, 3] (nnU-Net v2)
    """
    img = sitk.ReadImage(input_file)
    img_array = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_array)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('Unexpected label found: {}'.format(u))

    new_labels = np.zeros_like(img_array)
    new_labels[img_array == 4] = 3  # 4 -> 3
    new_labels[img_array == 2] = 1  # 2 -> 1
    new_labels[img_array == 1] = 2  # 1 -> 2

    new_img = sitk.GetImageFromArray(new_labels)
    new_img.CopyInformation(img)

    sitk.WriteImage(new_img, output_file)
    print(f"Converted labels saved to: {output_file}")

def convert_labels_in_folder(input_folder: str, output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            update_labels(input_file_path, output_file_path)

if __name__ == '__main__':
    input_folder = input("Origin Path: ")  
    output_folder = input("Output Path: ") 

    convert_labels_in_folder(input_folder, output_folder)
