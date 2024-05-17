import os
import zipfile

# Define the path to the folder containing the zip files
folder_path = "/scratch3/Siberia"

# Initialize the lists
with_scratch2 = []
without_scratch2 = []

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is a zip file
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Get the first folder in the zip file
            first_folder = zip_ref.namelist()[0].split('/')[0]
            
            # Check if the first folder is "scratch2"
            if first_folder == "scratch2":
                with_scratch2.append(file_name)
            else:
                without_scratch2.append(file_name)

# Write the lists to text files
with open("/scratch2/biomass_estimation/code/patches/with_scratch2.txt", "w") as file:
    file.write("\n".join(with_scratch2))

with open("/scratch2/biomass_estimation/code/patches/without_scratch2.txt", "w") as file:
    file.write("\n".join(without_scratch2))