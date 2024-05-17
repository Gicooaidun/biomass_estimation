import geopandas as gpd
import pandas as pd
import time
import random
import fiona

random_numbers = [random.randint(1, 1000000) for _ in range(10000)]
random_numbers = sorted(random_numbers)

def records(filename, list):
    list = sorted(list) # if the elements of the list are not sorted
    with fiona.open(filename) as source:
        for i, feature in enumerate(source[:max(list)+1]):
            if i in list:
                yield feature

starttime = time.time()
#reading in GEDI points
# gedi_df = gpd.read_file('GEDI_analysis/GEDI_data/4190and4240gedi.gpkg', driver='GPKG', rows = 10000, index = True)
# gedi_df = gpd.read_file('GEDI_analysis/GEDI_data/clipped_gedi.gpkg', driver='GPKG', index = True)
gedi_df = gpd.GeoDataFrame.from_features(records("GEDI_analysis/GEDI_data/clipped_gedi.gpkg", random_numbers))
endtime = time.time()
print(f"Reading in the GEDI shapefile took {endtime - starttime} seconds")
# print(gedi_df.head())
#reading in ICESat2 shapefile
# icesat_df = gpd.read_file('GEDI_analysis/GEDI_data/4190and4240icesat.shp', driver='ESRI Shapefile', index = True)
icesat_df = gpd.read_file('GEDI_analysis/GEDI_data/clipped_icesat.shp', driver='ESRI Shapefile', index = True)
# print(icesat_df.head())

# overlapping_tiles = pd.DataFrame()
overlapping_tiles_list = []
#iterating over each Sentinel 2 tile and finding the ICESat2 tiles that overlap with it
for index, gedi_point in gedi_df.iterrows():
    if index % 1000 == 0:
        print(f"Processing row {index}")
    gedi_geometry = gedi_point.geometry
    overlapping_icesat_tiles = icesat_df[icesat_df.geometry.intersects(gedi_geometry)].copy()
    overlapping_icesat_tiles.drop(columns=['geometry'], inplace=True)

    # adding the name of the gedi tile to the ICESat2 tiles that overlap with it
    if not overlapping_icesat_tiles.empty:
        overlapping_icesat_tiles.loc[:, 'gedi shot number'] = gedi_point.shot_numbe
        # overlapping_icesat_tiles.loc[:, 'gedi Tile Geometry'] = gedi_geometry

    # overlapping_tiles = pd.concat([overlapping_tiles, overlapping_icesat_tiles])
    overlapping_tiles_list.append(overlapping_icesat_tiles)


# overlapping_tiles_df = gpd.GeoDataFrame(overlapping_tiles, columns=['gedi Tile Name', 'Overlapping ICESat Tiles'])
df = pd.concat(overlapping_tiles_list)

df.to_csv('gedi_icesat_matches_v3.csv', index=False)
print("file saved")