#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 10:06:54 2025

@author: rowandavies
"""

#Question 1
#Time series challenge: Predict the weekly number of expected service requests per 
#hex that will be created each week using sr_hex.csv, for 4 weeks past 
#the end of the dataset.


import pandas as pd
import matplotlib.pyplot as plt
#pip install folium
#pip install folium==0.14.0
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
import os
import requests
import json
import geopandas as gpd
from shapely.geometry import Point


#Download data###################################################################################
#Set working dir
os.chdir("/Users/rowandavies/desktop/ds_code_challenge")
# Download geojson hex data
response = requests.get("https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/city-hex-polygons-8.geojson")
geojson_hex_8 = response.json()  

#data = pd.read_csv("https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/sr_hex_truncated.csv")
data_1 = pd.read_csv("https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/sr_hex.csv.gz")
print(data_1.head())


#Basic data vis to understand more about the data.################################################
#Note: we have 1 years worth of service request data. 
#Initial thoughts: there will be temporal trends in the data, both monthly (seasonal) and daily (days of the week).

print(data_1['creation_timestamp'].dtype)
data_1['creation_timestamp'] = pd.to_datetime(data_1['creation_timestamp'])

# Create a new column with just the date (optional)
data_1['creation_date'] = data_1['creation_timestamp'].dt.date
# Plot histogram
plt.figure(figsize=(12,6))
plt.hist(data_1['creation_date'], bins=(12*52), edgecolor='black')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Histogram of Creation Timestamps by Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data_1['day_of_week'] = data_1['creation_timestamp'].dt.day_name()
# Count the number of rows per day
counts = data_1['day_of_week'].value_counts()
# Reorder to Monday-Sunday
ordered_counts = counts.reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
# Plot
plt.figure(figsize=(8,5))
ordered_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Records')
plt.title('Number of Records per Day of the Week')
plt.tight_layout()
plt.show()
#Prectictably, the weekend have fewet service requests


data_1['month_of_year'] = data_1['creation_timestamp'].dt.month_name()
month_counts = data_1['month_of_year'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
plt.figure(figsize=(10,5))
month_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Month of the Year')
plt.ylabel('Number of Records')
plt.title('Number of Records per Month')
plt.tight_layout()
plt.show()
#Predictably, there is a monthly trent too

#There is also a spatial component to this data
# Drop rows with missing coordinates
data_1_clean = data_1.dropna(subset=['latitude', 'longitude'])
# Clean missing coordinates
data_1_clean = data_1.dropna(subset=['latitude', 'longitude'])
# Prepare heat data
heat_data = data_1_clean[['latitude', 'longitude']].values.tolist()
# Compute map center
center_lat = data_1_clean['latitude'].mean()
center_lon = data_1_clean['longitude'].mean()
# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
# Add heatmap layer
HeatMap(heat_data, radius=10).add_to(m)
# Display in notebook
#Save to HTML
m.save('heatmap.html')
#This map help visualise the data.


#Let's try to understand more about the hex data
gdf = gpd.GeoDataFrame.from_features(geojson_hex_8["features"])
fig, ax = plt.subplots(figsize=(10,10))
gdf.plot(ax=ax, edgecolor="black", facecolor="none")
plt.title("H3 Level 8 Hexagons")
plt.show()
#Clean hexagones relevant to the city of cape town.

#Let's attribute the hex data to the service request data frame to understand more about the service calls in different hex regions.
# Remove rows without lat/lon
#We remove all rows without geographical data. 
data_1_hex = data_1.dropna(subset=['latitude', 'longitude']).copy()
data_check = data_1[
    data_1['latitude'].isna() | data_1['longitude'].isna()
]
# Create geometry column
data_1_hex["hex"] = data_1_hex.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
# Convert to GeoDataFrame
gdf_points = gpd.GeoDataFrame(data_1_hex, geometry="hex", crs="EPSG:4326")
gdf_points.plot(figsize=(10,10), alpha=0.5, markersize=2)
plt.title("Geographic Distribution of Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#Plot Hexs with service calls
# Create figure and axis
fig, ax = plt.subplots(figsize=(12,12))
# Plot hex polygons first
gdf.plot(
    ax=ax,
    edgecolor="black",
    facecolor="none",
    linewidth=0.5,
    alpha=0.8
)
# Plot latitude/longitude points from data_1
plt.scatter(
    data_1["longitude"],
    data_1["latitude"],
    color="red",
    alpha=0.5,
    s=5,
    label="Service Requests"
)
# Labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Service Requests and H3 Level 8 Hexagons")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

joined = gpd.sjoin(
    gdf_points,
    gdf[["geometry", "index"]],
    how="left",
    predicate="within"
)


joined = joined.rename(columns={"index": "hex_index"})
joined["hex_index"] = joined["hex_index"].fillna(0)
data_1_hex = joined

hex_counts = data_1_hex["hex_index"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12,6))
hex_counts.plot(kind="bar", edgecolor="black")
plt.xlabel("H3 Hex Index")
plt.ylabel("Number of Records")
plt.title("Number of Service Requests per H3 Hex")
plt.tight_layout()
plt.show()

# Count records per hex
hex_counts = joined["hex_index"].value_counts()
hex_counts = hex_counts[hex_counts.index != 0]
hex_counts_df = hex_counts.reset_index()
hex_counts_df.columns = ["hex_index", "count"]
# Ensure hex index column is named consistently
gdf = gdf.rename(columns={"index": "hex_index"})
# Merge counts into polygons
gdf_counts = gdf.merge(hex_counts_df, on="hex_index", how="left")
gdf_counts["count"] = gdf_counts["count"].fillna(0)
# Plot choropleth
fig, ax = plt.subplots(figsize=(12,12))
gdf_counts.plot(
    column="count",
    ax=ax,
    cmap="OrRd",
    edgecolor="black",
    linewidth=0.2,
    legend=True
)
plt.title("Service Request Count per Hex")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()


# Compute center of map
center_lat = gdf_counts.geometry.centroid.y.mean()
center_lon = gdf_counts.geometry.centroid.x.mean()
# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
# Convert GeoDataFrame to GeoJSON
geojson_data = gdf_counts.to_json()
# Add choropleth
folium.Choropleth(
    geo_data=geojson_data,
    name='choropleth',
    data=gdf_counts,
    columns=['hex_index', 'count'],
    key_on='feature.properties.hex_index',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Service Request Count'
).add_to(m)
# Optional: Add layer control
folium.LayerControl().add_to(m)
# Save to HTML
m.save("service_requests_hex_map.html")

#Thus ends our data visualisation and attribution. 
#We have serive call orderred into geographical hexes


