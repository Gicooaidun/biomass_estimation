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


with open('code/newer/links.txt', 'w') as file:
    for filename in filenames:
        file.write('https://data.ornldaac.earthdata.nasa.gov/protected/above/Boreal_AGB_Density_ICESat2/data/' + filename + '\n')