import geopandas as gpd
import os
import rasterio

# Path to the GEDI shapefile
gedi_path = 'GEDI_analysis/GEDI_data/L4A_Siberia.shp'
# Path to the folder containing the icesat tif files
icesat_path = 'GEDI_analysis/reprojected_icesat_to_gedi_crs'


# Open the shapefile
gedi =  gpd.read_file(gedi_path, rows=100000)
# print(src.crs) #epsg:4326
# Iterate over the subset
i = 1
for index, row in gedi.iterrows():
    if index % 1000 == 0:
        print(f"Processing row {index}")
    # Extract lat and lon from the shapefile row
    lat = row['lat_lowest']
    lon = row['lon_lowest']
    # print(f"Shapefile row {index} coordinates: {lat}, {lon}")

    # Iterate over the tif files in the folder
    for filename in os.listdir(icesat_path):
        if filename.endswith('.tif'):
            tif_path = os.path.join(icesat_path, filename)
            
            # Open the tif file
            with rasterio.open(tif_path) as icesat:
                # print(icesat.bounds[1])
                if lat >= icesat.bounds[1] and lat <= icesat.bounds[3] and lon >= icesat.bounds[0] and lon <= icesat.bounds[2]:
                    print(f"Shapefile row {index} intersects with tif file {filename}")
                    print(f"Shapefile row {index} coordinates: {lat}, {lon}")
                    print(f"Tif file {filename} bounds: {icesat.bounds}")
                    print("=====================================")

                # if statement to print all attributes of the first icesat file
                # if index == 0 and i == 1:
                #     i+=1
                #     # Print dataset attributes
                #     print("\nDataset Attributes:")
                #     for attr_name in dir(icesat):
                #         if not attr_name.startswith("_"):  # Exclude private attributes
                #             print(f"{attr_name}: {getattr(icesat, attr_name)}")
                #             pass