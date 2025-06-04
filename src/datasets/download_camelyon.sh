#!/usr/bin/env bash

# Create the folder if it doesn't exist
mkdir -p data/pcam
cd data/pcam

# Download files into datasets/pcam
echo "Downloading Camelyon dataset..."
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz

# Unzip all downloaded .gz files
echo "Unzipping downloaded files..."
gunzip ./*.gz

# Remove the .gz files after unzipping
echo "Cleaning up .gz files..."
rm -f ./*.gz

# Return to the original directory
cd ../..

# Print a message indicating completion
echo "Camelyon dataset downloaded and extracted successfully into datasets/pcam."