import tifffile as tiff
import numpy as np
import os

# Define input and output directories
input_directory = '/gpfs/scratch/ss14424/Brain/BrainData/BRAIN_IMC_MaskTif'
output_directory = '/gpfs/scratch/ss14424/Brain/BrainData/BRAIN_IMC_MaskTif_csd_2'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define the target channel structure
target_channels = [
    'CD117', 'CD11c', 'CD14', 'CD163', 'CD16', 'CD20', 'CD31', 'CD3', 'CD4', 'CD68', 'CD8a', 'CD94',
    'DNA1', 'FoxP3', 'GFAP', 'HLA-DR', 'MPO', 'Olig2', 'P2PY12', 'Sox2', 'Sox9', 'MeLanA', 'PMEL', 'PanCK'
]

# Define the channel mapping for Glioma and BrM images
glioma_channels = {
    'CD117': 1, 'CD11c': 2, 'CD14': 3, 'CD163': 4, 'CD16': 5, 'CD20': 6, 'CD31': 7, 'CD3': 8, 'CD4': 9,
    'CD68': 10, 'CD8': 11, 'CD94': 12, 'DNA1': 13, 'FoxP3': 14, 'GFAP': 15, 'HLA-DR': 16, 'MPO': 17,
    'Olig2': 18, 'P2PY12': 19, 'Sox2': 20, 'Sox9': 21
}

brm_channels = {
    'CD117': 1, 'CD11c': 2, 'CD14': 3, 'CD163': 4, 'CD16': 5, 'CD20': 6, 'CD31': 7, 'CD3': 8, 'CD4': 9,
    'CD68': 10, 'CD8a': 11, 'CD94': 12, 'DNA1': 13, 'FoxP3': 14, 'GFAP': 15, 'HLA-DR': 16, 'MPO': 17,
    'MeLanA': 18, 'P2PY12': 19, 'PMEL': 20, 'PanCK': 21
}

# Process each file in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.tif'):
        file_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)
        
        # Check if the image already exists in the target folder
        if os.path.exists(output_path):
            print(f"Skipping image {file_name}: already exists")
            continue

        # Load the image
        image = tiff.imread(file_path)
        
        # Check if the image has 21 channels
        if image.shape[0] != 21:
            print(f"Error: {file_name} does not have 21 channels. Skipping this image.")
            continue

        # Create an empty array to hold the consolidated image
        consolidated_image = np.zeros((len(target_channels), *image.shape[1:]), dtype=image.dtype)

        # Determine if the image is Glioma or BrM based on the filename
        if file_name.startswith('Glioma'):
            source_channels = glioma_channels
        elif file_name.startswith('BrM'):
            source_channels = brm_channels
        else:
            print(f"Error: {file_name} has an unknown image type. Skipping this image.")
            continue

        # Reorganize the channels according to the target structure
        for i, channel_name in enumerate(target_channels):
            if channel_name in source_channels:
                channel_index = source_channels[channel_name] - 1  # Convert to zero-based index
                consolidated_image[i] = image[channel_index]
            else:
                consolidated_image[i] = np.zeros_like(image[0])  # Empty channel

        # Save the consolidated image
        tiff.imwrite(output_path, consolidated_image)
        print(f"Consolidated image saved to {output_path}")
