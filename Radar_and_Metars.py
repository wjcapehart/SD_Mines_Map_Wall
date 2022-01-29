#!/usr/bin/env python
# coding: utf-8

# # Radar
# 
# Creates an Animated Plot for Radar and Station Models

# In[1]:


####################################################
####################################################
####################################################
#
# Libraries
#

from datetime            import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy             as np

import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import metpy.calc        as mpcalc

from   metpy.plots       import colortables, USCOUNTIES, StationPlot, sky_cover,current_weather
from   metpy.units       import units

import matplotlib.patches as patches

from siphon.cdmr         import Dataset
from siphon.radarserver  import get_radarserver_datasets, RadarServer
from siphon.catalog      import TDSCatalog

import siphon.catalog    as siphcat  
import siphon.ncss       as siphncss

import os                as os
import platform          as platform

import pathlib           as pathlib


import urllib.request
import shutil
import metpy.io          as mpio



import timezonefinder    as tzf
import pytz              as pytz

import pandas            as pd

import airportsdata as airpt

import IPython.display as idisplay

#
####################################################
####################################################
####################################################


# In[2]:


####################################################
####################################################
####################################################
#
# System Control
#

os.system("rm -v ./radar_temp_files/*")

time_now   = datetime.utcnow()
time_start = time_now - timedelta(hours=2)

print(time_start)
print(time_now)


metar_time_spread = 45

siphon_time_series       = pd.date_range(time_start- timedelta(hours=1), time_now,freq='H')
siphon_pulls_YYYYMMDD_HH = siphon_time_series.strftime("%Y%m%d_%H00")

print(siphon_pulls_YYYYMMDD_HH)


geospatial_lat_min =  42.05982
geospatial_lat_max = 46.19018

geospatial_lon_min = -105.70986
geospatial_lon_max =  -99.950134



station_id = "RAP"
radar_id   = "UDX"



metar_collection = 'https://thredds.ucar.edu/thredds/catalog/nws/metar/ncdecoded/catalog.xml?dataset=nws/metar/ncdecoded/Metar_Station_Data_fc.cdmr'
synop_collection = "https://thredds.ucar.edu/thredds/catalog/nws/synoptic/ncdecoded/catalog.xml?dataset=nws/synoptic/ncdecoded/Surface_Synoptic_Point_Data_fc.cdmr" 

norm, cmap = colortables.get_with_steps("NWSStormClearReflectivity", 
                                        -20, 
                                        0.5)


#
####################################################
####################################################
####################################################


# In[3]:


####################################################
####################################################
####################################################
#
# Crack Open Airport Databases
#

airport_database_IATA = airpt.load("IATA")
airport_database_ICAO = airpt.load("ICAO")

#
# Get the Local Time Zone Information
#

tf     = tzf.TimezoneFinder()
tz     = tf.certain_timezone_at(lng = airport_database_IATA[station_id]['lon'], 
                                lat = airport_database_IATA[station_id]['lat'])

#
####################################################
####################################################
####################################################


# In[4]:


####################################################
####################################################
####################################################
#
# Pull Text-Based METARS from UNIDATA NOAAPORT Experimental Site
#

# https://thredds-test.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar/metar_20210924_0000.txt

cat = siphcat.TDSCatalog('https://thredds-test.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml')



first = True
for datehour in siphon_pulls_YYYYMMDD_HH:
    

    
    metar_url  = "https://thredds-test.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar/metar_"+datehour+".txt"
    metar_file = "./radar_temp_files/metar_"+datehour+".txt"
    
    
    path_to_file = pathlib.Path(metar_file)
    
    print(path_to_file, path_to_file.is_file())

    
    
 
    print("downloading "+ metar_url)
    with urllib.request.urlopen(metar_url) as response, open(metar_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
            
    print("cracking "+metar_file)
    try:
        indata = mpio.metar.parse_metar_file(metar_file)
        if first:
            first = False
            metar_dataframe = indata
        else:
            metar_dataframe = metar_dataframe.append(indata)
            metar_dataframe = metar_dataframe.drop_duplicates()
    except ValueError:
        print("BALLS! Parse Error")
        error_404 = True
        pass


metar_dataframe = metar_dataframe[(metar_dataframe['latitude']  > geospatial_lat_min) & 
                                  (metar_dataframe['latitude']  < geospatial_lat_max) &
                                  (metar_dataframe['longitude'] > geospatial_lon_min) &
                                  (metar_dataframe['longitude'] < geospatial_lon_max) ]

metar_dataframe = metar_dataframe.drop(["station_id",
                                        "current_wx2_symbol",
                                        "current_wx3_symbol",
                                        "current_wx1",
                                        "current_wx2",
                                        "current_wx3",
                                        "altimeter",
                                        "remarks",
                                        "wind_direction",
                                        "wind_speed",
                                        "low_cloud_type", 
                                        "low_cloud_level", 
                                        "medium_cloud_type",
                                        "medium_cloud_level", 
                                        "high_cloud_type", 
                                        "high_cloud_level",
                                        "highest_cloud_type", 
                                        "highest_cloud_level"], axis=1)

metar_dataframe = metar_dataframe.sort_values("date_time").reset_index()

metar_dataframe = metar_dataframe.rename(columns={'cloud_coverage': 'cloud_eights', 
                                                  'visibility':'visibility_sm',
                                                  'station_id': 'ICAO_id'})

metar_dataframe['cloud_eights']          = (8 * metar_dataframe['cloud_eights']/10).fillna(10).values.astype(int)
metar_dataframe['air_temperature']       = ( metar_dataframe['air_temperature'] * 9/5) + 32
metar_dataframe['dew_point_temperature'] = ( metar_dataframe['dew_point_temperature'] * 9/5) + 32
metar_dataframe['visibility_sm']         = np.round(metar_dataframe['visibility_sm'] / 1609.34,decimals=1)


#
####################################################
####################################################
####################################################


# In[5]:


####################################################
####################################################
####################################################
#
# Radar Plotting Function
#


def make_radar_station_map(ds):
    
    radar    = Dataset(ds.access_urls['CdmRemote'])


    rng      = radar.variables['gate'][:] 
    az       = radar.variables['azimuth'][:]
    ref      = radar.variables['BaseReflectivityDR'][:]
    time_utc = datetime.strptime(radar.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ")

    time_utc_p = time_utc + timedelta(minutes=metar_time_spread/2)
    time_utc_m = time_utc - timedelta(minutes=metar_time_spread/2)
    
    
    
    metar_dataframe['staleness'] = (time_utc-metar_dataframe['date_time'])/ np.timedelta64(1, 'm')

    
    local_metars = metar_dataframe[(metar_dataframe['date_time'] > time_utc_m) &
                                   (metar_dataframe['date_time'] < time_utc_p) ].copy()
    
    
    alpha_array = (np.abs(local_metars['staleness'])/metar_time_spread*2).to_numpy()
    alpha_array[ alpha_array<0] = 0
    alpha_array[ alpha_array>1] = 1


    
    
    local_metars['alpha'] = alpha_array


    local_metars = local_metars.sort_values("alpha").reset_index()
    
    

    valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H:%M:%S %Z")
    local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(valid_time + "  (" + local_time+")")


    x   = rng * np.sin(np.deg2rad(az))[:, None]
    y   = rng * np.cos(np.deg2rad(az))[:, None]
    ref = np.ma.array(ref, mask=np.isnan(ref))




    RadarLatitude      = radar.RadarLatitude
    RadarLongitude     = radar.RadarLongitude
    geospatial_lat_min = radar.geospatial_lat_min
    geospatial_lat_max = radar.geospatial_lat_max
    geospatial_lon_min = radar.geospatial_lon_min
    geospatial_lon_max = radar.geospatial_lon_max

    #
    ###################################




 


    #
    ###################################


    ###################################
    #
    # Render Image
    #

    # fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    


    
    ax = plt.subplot(111, 
                     projection=ccrs.Stereographic(central_latitude    = RadarLatitude, 
                                                   central_longitude   = RadarLongitude, 
                                                   false_easting       = 0.0, 
                                                   false_northing      = 0.0, 
                                                   true_scale_latitude = None, 
                                                   globe=None))

    plt.suptitle(radar.ProductStationName + " ["+radar.ProductStation +"] " +radar.keywords_vocabulary,
                fontsize=20)

    ax.set_title(valid_time + "  (" + local_time+")")


    ax.set_extent([geospatial_lon_min, 
                   geospatial_lon_max, 
                   geospatial_lat_min, 
                   geospatial_lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(feature    = cfeature.STATES)
    ax.add_feature(feature    = cfeature.COASTLINE)
    ax.add_feature(feature    = USCOUNTIES, 
                   linewidths = 0.5,
                   edgecolor  = 'black',
                   facecolor  = 'none')
    filled_cm = ax.pcolormesh(x, 
                              y, 
                              ref,
                              norm = norm, 
                              cmap = cmap)
    ax.set_aspect('equal', 'datalim')
    plt.colorbar(filled_cm, 
                 label  = "Reflectivity (dbZ)",
                 shrink = 0.8,
                 pad    = 0.012)


    # Metar Plots
    print("number of obs",len(local_metars))
    for i in range(0,len(local_metars)) :
        single_row = local_metars.loc[i]
    
        stationplot = StationPlot(ax, 
                                  single_row['longitude'], 
                                  single_row['latitude'], 
                                  transform = ccrs.PlateCarree(),
                                  fontsize  = 12,
                                  alpha     = single_row["alpha"])

        stationplot.plot_parameter('NW', 
                                   np.array([single_row['air_temperature']]), 
                                   color='black')
        stationplot.plot_parameter('SW', 
                                   np.array([single_row['dew_point_temperature']]), 
                                   color='black')
        stationplot.plot_parameter('NE', 
                                   np.array([single_row['air_pressure_at_sea_level']]),  
                                   color='black')
        stationplot.plot_parameter('SE',
                                   np.array([single_row['staleness']]),
                                   color='grey')

        stationplot.plot_symbol('C', 
                                np.array([single_row['cloud_eights']]), 
                                sky_cover)
        stationplot.plot_symbol('W', 
                                np.array([single_row['current_wx1_symbol']]), 
                                current_weather)

        stationplot.plot_text((2, 0), 
                              np.array([single_row['ICAO_id']]), 
                              color='black')
        stationplot.plot_barb(np.array([single_row['eastward_wind']]), 
                              np.array([single_row['northward_wind']]))
        
        del single_row
        
    del local_metars

    plt.tight_layout()

    #
    ###################################


#
####################################################
####################################################
####################################################


# In[6]:


####################################################
####################################################
####################################################
#
# Retrieve Radar Data
#

ds  = get_radarserver_datasets('http://thredds.ucar.edu/thredds/')
url = ds['NEXRAD Level III Radar from IDD'].follow().catalog_url
rs  = RadarServer(url)

query = rs.query()

query.stations(radar_id).time_range(time_start,
                                    time_now).variables('N0Q')

rs.validate_query(query)

catalog = rs.get_catalog(query)

datasets_sorted = sorted(catalog.datasets)

list(datasets_sorted)

figure_counter    = 1
number_of_figures = len(sorted(catalog.datasets))



for name in datasets_sorted:
    ds = catalog.datasets[name]
    print(ds)
    
#
####################################################
####################################################
####################################################


# In[7]:


####################################################
####################################################
####################################################
#
# Create Individual Map Files
#

counter = 0


figure_counter = 1
number_of_figures = len(sorted(catalog.datasets))
for name in sorted(catalog.datasets):
    percent_done = figure_counter / number_of_figures
    ds = catalog.datasets[name]
    print(ds)
    print(percent_done)
    fig, ax = plt.subplots(1, 1, 
                           figsize=(9, 8))#,
                           #facecolor = 'white')
    



    
    make_radar_station_map(ds)
    plt.savefig("./radar_temp_files/Radar_Loop_Image_"+str(counter).zfill(2)+".png")
    counter = counter + 1
    figure_counter = figure_counter + 1


    plt.close()
    print("=====================")

#
####################################################
####################################################
####################################################


# In[8]:


##################################################
#
# Convert PNGs into an Animated GIF
#

os.system("convert -delay 25 " + 
          "./radar_temp_files/Radar_Loop_Image_*.png"  + 
          " " + 
          "./graphics_files/RealTime_Radar_Loop.gif")


#
##################################################


# In[ ]:




