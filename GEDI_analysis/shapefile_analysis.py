import geopandas as gpd

# Path to the shapefile
shapefile_path = 'GEDI_analysis/GEDI_data/L4A_Siberia.shp'



# # Open the shapefile
# src =  gpd.read_file(shapefile_path, rows=10)
# # print(src.crs) #epsg:4326
# # Iterate over the subset
# for index, row in src.iterrows():
#     if index == 0:
#         print(row)
#         print("=====================================")




#reading in Sentinel 2 shapefile
gedi_df = gpd.read_file('GEDI_analysis/GEDI_data/L4A_Siberia.shp', driver='ESRI Shapefile', index = True, rows = slice(0, 100))
# print(gedi_df.head())
#reading in ICESat2 shapefile
icesat_df = gpd.read_file('notebook/boreal_agb_density_ICESat2_tiles_shp/Boreal_AGB_Density_ICESat2_tiles.shp', driver='ESRI Shapefile', index = False)
# print(icesat_df.head())
for index, row in icesat_df.iterrows():
    if index == 0:
        print(row)
        print("=====================================")


