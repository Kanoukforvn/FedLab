import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from fedlab.utils.dataset.partition import ISICPartitioner
from fedlab.utils.dataset import functional as F
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Load the ISIC 2019 dataset
data_root = "path/to/isic-2019"  # Change this to the path where the dataset is located
metadata_csv = os.path.join(data_root, "ISIC_2019_Training_Metadata.csv")
groundtruth_csv = os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv")
input_folder = os.path.join(data_root, "ISIC_2019_training_input")

class ISIC2019Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, groundtruth_csv, input_folder, transform=None):
        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        # Load ground truth labels
        self.groundtruth = pd.read_csv(groundtruth_csv)
        # Combine metadata and ground truth labels
        self.data = pd.merge(self.metadata, self.groundtruth, on="image", how="inner")
        # Set input folder and transform
        self.input_folder = input_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]["image"]
        image_path = os.path.join(self.input_folder, image_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        label = self.data.iloc[idx]["MEL"]  # Assuming "MEL" is the label column
        if self.transform:
            image = self.transform(image)
        return image, label

# Create ISIC 2019 dataset instance
isic2019_dataset = ISIC2019Dataset(metadata_csv, groundtruth_csv, input_folder)

# Set parameters for partitioning
num_clients = 100
num_classes = 2  # Change this to the number of classes in your dataset
seed = 2021

# Perform partitioning
isic_partitioner = ISICPartitioner(isic2019_dataset.targets, 
                                    num_clients,
                                    partition='noniid-#label',  # Adjust partitioning scheme if needed
                                    dir_alpha=1,
                                    major_classes_num=1,
                                    verbose=True,
                                    seed=seed)

# Save partitioned dataset
torch.save(isic_partitioner.client_dict, "isic2019_partition.pkl")

# Generate partition report
csv_file = "isic2019_partition_report.csv"
F.partition_report(isic2019_dataset.targets, isic_partitioner.client_dict, 
                   class_num=num_classes, 
                   verbose=False, file=csv_file)

# Read partition report into a DataFrame
isic_partition_df = pd.read_csv(csv_file, header=0)
isic_partition_df = isic_partition_df.set_index('cid')
col_names = [f"class-{i}" for i in range(num_classes)]
for col in col_names:
    isic_partition_df[col] = (isic_partition_df[col] * isic_partition_df['TotalAmount']).astype(int)

# Select first 10 clients for bar plot
isic_partition_df[col_names].iloc[:10].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Sample count')
plt.ylabel('Client ID')
plt.savefig("isic2019_partition_barplot.png", dpi=400, bbox_inches='tight')
