import geopandas as gpd
import os
import json

sentinel_path = "S2_tiles_Siberia_polybox/"
icesat_path = "boreal_agb_density_ICESat2_tiles_shp/"

s2_filepath = os.path.join(sentinel_path, "S2_tiles_Siberia_all.geojson")
s2_data = gpd.read_file(s2_filepath)
s2_data = s2_data.to_crs(epsg=4326)


icesat_filepath = os.path.join(icesat_path, "Boreal_AGB_Density_ICESat2_tiles.shp")
icesat_data = gpd.read_file(icesat_filepath)

print(s2_data.crs)
print(icesat_data.crs)

print(s2_data.head())
print(icesat_data.head())

correspondences = {}
for _, s2_row in s2_data.iterrows():
    for _, icesat_row in icesat_data.iterrows():
        if s2_row.geometry.intersects(icesat_row.geometry):
            if s2_row['Name'] not in correspondences:
                correspondences[s2_row['Name']] = [icesat_row.tile_num]
            else:
                correspondences[s2_row['Name']].append(icesat_row.tile_num)

# Specify the file path
file_path = "correspondences_id.json"

# Write the dictionary to a JSON file
with open(file_path, 'w') as file:
    json.dump(correspondences, file, indent=4)

print("done")

