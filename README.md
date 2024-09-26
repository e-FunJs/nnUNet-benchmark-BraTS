# nnUNet-benchmark-BraTS
Thanks for reading this file! In this file, I'll introduce you to how to run nnUNetv2 for **BraTS**.<br>
## 1. Installation 
Please use a recent version of Python(**3.9** or newer is guaranteed to work).  
You need to follow the subsequent commands for installation.
  ```bash
  git clone https://github.com/MIC-DKFZ/nnUNet.git
  cd nnUNet
  pip install -e .
  ```
Please note that to avoid potential version conflicts, I recommend creating a new conda environment.  
```bash
conda create -n env_name python=3.9
```
Whether to install the **hiddenlayer** or not depends on you. But I still recommend downloading.
```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
```
## 2. Dataset preparation
If you clone my GitHub repository, you will find 3 empty folders: `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. Please do not change the names or locations of these folders arbitrarily, as nnU-Net has strict requirements for data organization, and modifications are not recommended. Once you have determined the locations and names of these three folders, **you need to add their paths to the environment variables**. <br>
I recommend that you locate the `.bashrc` file and add the following content:
```bash
export nnUNet_raw="/root/autodl-tmp/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/root/autodl-tmp/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/root/autodl-tmp/nnUNet/nnUNet_results"
```
Please determine the content based on your actual situation.   
Please download the dataset from [Here](https://drive.google.com/drive/folders/1O51sltrjFbGnOfM3vxezJOk7FtBSjUQg?usp=drive_link). <br>
Next, we can prepare to start training!
## 3. Training
Please confirm that you have downloaded the dataset I prepared. You may find 4 folders `Task001_BraTS`, `Dataset725_BraTS`, `BraTS2021_Training_Data`, and `Dataset666_BraTS2021`, where `Task001_BraTS` and `BraTS2021_Training_Data` are the original datasets, while `Dataset725_BraTS` and `Dataset666_BraTS2021` are the converted datasets. <br>
Although I have provided datasets that have already been converted, I still recommend that you do not use them directly. This is because the converted datasets I provided are 2 fragments extracted from the whole datasets.<br>
### 3.1. MSD dataset conversion   
`Task001_BraTS` is used for category-based training and `BraTS2021_Training_Data` is used for region-based training.<br>
- If you want to conduct category-based training, please run ```nnUNetv2_convert_MSD_dataset -i your_dataset_path -overwrite_id custom id```    
- If you want to conduct region-based training, you can find `convert_BraTS2021. py` I wrote in `utils` folder. You just need to make a simple change to the file's read path, very simple!
  
When you complete the conversion, a new folder will appear in `nnUNet_raw`, with your custom ID included in the folder name. This ID will be the only stuff recognized by the program.
This is the format you should obtain(For instance):
```
    nnUNet_raw/
    ├── Dataset725_BraTS
        │── nnUNetTrainer__nnUNetPlans__2d
             ├── imagesTr
             |    ├── BRATS_001_0000.nii.gz
             |    ├── BRATS_001_0001.nii.gz
             |    ├── BRATS_001_0002.nii.gz
             |    ├── BRATS_001_0003.nii.gz
             |    ├── BRATS_002_0000.nii.gz
             |    ├── ...
             ├── labelsTr
             |    ├── BRATS_001.nii.gz
             |    ├── BRATS_002.nii.gz
             |    ├── BRATS_003.nii.gz
             |    ├── ...
             ├── imagesTs(optional)
             |    ├── ...
             └── dataset.json

```
In fact, the results generated by subsequent processing also roughly conform to this format, except for differences between files.    
### 3.2. Experiment planning and preprocessing
With just 1 command, nnUNet can extract the information and make plans.The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
nnUNetv2_plan_and_preprocess -d ID(Just customized) --verify_dataset_integrity
```
I recommend `--verify_dataset_integrity` whenever it's the first time  you run this command because this command can check for some of the most common error sources.  
Once the planning and processing are completed, you can see the results in `nnUNet_preprocessed` folder.  
### 3.3. Model training
Before starting the training, please find `nnunetv2/training/nnUNetTrainer/NnUNetTrainer.py` file. You can modify some parameters used during training (approximately at line 150) in this file.  
nnU-Net trains all U-Net configurations using 5-fold cross-validation. This allows nnU-Net to determine the post-processing and ensemble of the training dataset.  
Please run the following command:  
```bash
  nnUNetv2_train ID(costom) UNet_Configuration(2d/3d_fullres) FOLD(0/1/2/3/4) --npz
```
For BraTS, in general, the effect of 3d_fullres is better than 2d. As for me, I only ran 3d_fullres (because training is really time-consuming!)    
If the training is interrupted, don't worry, you can continue training with the latest checkpoint:
```bash
  nnUNetv2_train ID(costom) UNet_Configuration(2d/3d_fullres) FOLD(0/1/2/3/4) --npz --c
```
The trained model will be written to the `nnUNet_results` folder. A picture will appear in the folder to record your training process. In addition, a `summary.json` file will appear where you can see all the metrics. I have added some new metrics, and you can find the process of obtaining these metrics in `nnunetv2/evaluation/evaluate_predictions. py`.<br>
[Here](https://imgur.com/a/uNNfkEm) is a visualization process image from my certain training session.
### 3.4. Determination of the optimal U-Net configuration   
In order to proceed with this phase, you need to ensure that you have completed at least 5 training sessions for a certain UNet configuration such as 3d_fullres (different folds).  
Run this command:   
```
nnUNetv2_find_best_configuration ID(custom) -c (e.g., 2d/3d_fullres/2d 3d_fullres if you have trained on both of them) -f 0 1 2 3 4
```
### 4. Inference and Post-Processing
Run these commands:
```bash
  nnUNetv2_predict -d ID(custom) -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
  nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /root/autodl-tmp/nnUNet/nnUNet_results/Dataset666_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /root/autodl-tmp/nnUNet/nnUNet_results/Dataset666_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

```
These are just examples. In fact, when you run **the command in 3.4**, the terminal will tell you what to do. You only need to provide the input and output paths for it. For inference, the input path should be **imageTs**.  

---
This is the entire process, which can be submitted to the official website for verification. If you encounter any problems, you can check the **report** or send me a private message.
