# Report-Taming-Transformers
This is a reimplementation from scratch of the VQ-GAN / transformer framework from "Taming Transofmers" to generate high resolution images

# Run & Reproduce Results

## 0: Download Animal Faces (AFHQ) dataset
Find the link to donwload the dataset (kaggle)
Download packages `pip install pytorch torchvision lpips panda`
With the right CUDA version for you gpu

## 1: Train VQGAN
Change the parameter in `train_stage1_vqgan.py` with the right hyperparameters for the VQGAN and the right path for datasets, loging and checkpoints.
Run the program, it trains and checkpoints in `checkopints`!

## 2: Convert dataset
Change the path & parameters in `precompute_codebook.py`
Run the program, it creates the dataset of Tensor for the Transformers

## 3: Train Transformers
Change the path & parameter in `train_stage2_transformer.py`
Run the program to train the Transformers.
It checkpoins in `checkpoints/`


## 4: Samples new images
Change the model & path in `samples.py`
Run the models it create an images !