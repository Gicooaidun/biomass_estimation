import csv
import os
import shutil

filename = 'code/newer/overlapping_tiles.csv'
geo_tiff_column_index = 1  # Assuming the GeoTiff column is at index 2

filenames = []

with open(filename, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        if row[geo_tiff_column_index] not in filenames:
            filenames.append(row[geo_tiff_column_index])

# print(filenames)

shutil.copy('code/newer/beginning_bash.txt', 'code/newer/downloads.sh')
copy_filename = 'code/newer/downloads.sh'
with open(copy_filename, 'a') as copy_file:
    copy_file.write('\n')
    for filename in filenames:
        copy_file.write('https://data.ornldaac.earthdata.nasa.gov/protected/above/Boreal_AGB_Density_ICESat2/data/'+filename + '\n')
    copy_file.write('EDSCEOF\n')


# execute the script from where you want the files to be downloaded to with 
# 	chmod 777 download.sh
# 	./download.sh
# 	(((enter credentials and wait)))