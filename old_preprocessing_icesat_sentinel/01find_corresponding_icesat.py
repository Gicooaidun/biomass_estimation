import geopandas as gpd
import pandas as pd
s2_df = gpd.read_file('S2_tiles_Siberia_polybox/S2_tiles_Siberia_all.geojson', driver='GeoJSON', index = False)
# print(s2_df.head())
icesat_df = gpd.read_file('boreal_agb_density_ICESat2_tiles_shp/Boreal_AGB_Density_ICESat2_tiles.shp', driver='ESRI Shapefile', index = False)
# print(icesat_df.head())



        
overlapping_tiles = pd.DataFrame()
for s2_row in s2_df.iterrows():
    s2_tile = s2_row[1]
    s2_geometry = s2_tile.geometry
    overlapping_icesat_tiles = icesat_df[icesat_df.geometry.intersects(s2_geometry)]
    overlapping_icesat_tiles['S2 Tile Name'] = s2_tile.Name
    # overlapping_icesat_tiles['S2 Tile Geometry'] = s2_geometry
    overlapping_tiles = pd.concat([overlapping_tiles, overlapping_icesat_tiles])

# overlapping_tiles_df = gpd.GeoDataFrame(overlapping_tiles, columns=['S2 Tile Name', 'Overlapping ICESat Tiles'])
overlapping_tiles.to_csv('code/newer/overlapping_tiles.csv', index=False)