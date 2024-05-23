import geopandas as gpd

# Specify the path to your shapefile
shapefile_path = '/scratch2/biomass_estimation/code/GEDI_analysis/GEDI_data/L4A_Siberia.shp'

# Read the shapefile into a GeoDataFrame
gdf = gpd.read_file(shapefile_path, rows = 100)

# View the head of the data
print(gdf.head())
# Print all columns
print(gdf.columns)