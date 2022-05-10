#!/usr/bin/env python
# coding: utf-8

# # Radar
# 
# Creates an Animated Plot for Radar and Station Models

# In[ ]:


####################################################
####################################################
####################################################
#
# Libraries
#

from datetime            import datetime, timedelta

import numpy              as np
import matplotlib.pyplot  as plt


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


#
####################################################
####################################################
####################################################


# In[ ]:


####################################################
####################################################
####################################################
#
# System Control
#

os.system("rm -v ./temp_files_radar/*")

time_now   = datetime.utcnow()
time_start = time_now - timedelta(hours=3)

print(time_start)
print(time_now)


metar_time_spread = 59

siphon_time_series       = pd.date_range(time_start- timedelta(hours=1), time_now,freq='H')
siphon_pulls_YYYYMMDD_HH = siphon_time_series.strftime("%Y%m%d_%H00")

print(siphon_pulls_YYYYMMDD_HH)


geospatial_lat_min =  42.05982
geospatial_lat_max = 46.19018

geospatial_lon_min = -105.70986
geospatial_lon_max =  -99.950134

RadarLatitude   =   44.125
RadarLongitude  = -102.83


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


# In[ ]:


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


# In[ ]:


tz


# In[ ]:


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
    metar_file = "./temp_files_radar/metar_"+datehour+".txt"
    
    
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
            metar_dataframe = pd.concat(objs = [metar_dataframe,indata],
                                        axis =                  "index")
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


# In[ ]:


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
                                    time_now).variables('N0B') # n0q

rs.validate_query(query)

catalog = rs.get_catalog(query)

datasets_sorted = sorted(catalog.datasets)

list(datasets_sorted)

figure_counter    = 1
number_of_figures = len(sorted(catalog.datasets))

number_of_figures = len(sorted(catalog.datasets))
print("Number of Radar Images=",number_of_figures)

for name in datasets_sorted:
    ds = catalog.datasets[name]
    print(ds)
    
#
####################################################
####################################################
####################################################



# In[ ]:


####################################################
####################################################
####################################################
#
# Create Individual Map Files for Radar
#

counter = 0

number_of_figures = len(sorted(catalog.datasets))

if (len(sorted(catalog.datasets)) > 0) :

    figure_counter = 1
    for name in sorted(catalog.datasets):
        percent_done = figure_counter / number_of_figures
        ds = catalog.datasets[name]
        print(ds)
        print(percent_done)
        fig, ax = plt.subplots(1, 1, 
                               figsize=(9, 8),
                               facecolor = 'white')

        radar    = Dataset(ds.access_urls['CdmRemote'])

        rng      = radar.variables['gate'][:] 
        az       = radar.variables['azimuth'][:]
        ref      = radar.variables['BaseReflectivityDR'][:]
        time_utc = datetime.strptime(radar.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ")


        metar_dataframe['staleness']     = (time_utc-metar_dataframe['date_time'])/ np.timedelta64(1, 'm')
        metar_dataframe['abs_staleness'] =  np.abs( metar_dataframe['staleness'] )

        local_metars = metar_dataframe[(metar_dataframe['abs_staleness'] < metar_time_spread/2) ].copy()
        mystations   = local_metars["ICAO_id"].unique()

        first = True

        for mystation in mystations:
            deleteme  = local_metars[local_metars["ICAO_id"] == mystation].copy().sort_values("abs_staleness")
            deleteme2 = deleteme[deleteme["abs_staleness"]   == np.min(deleteme["abs_staleness"]) ]
            if (first) :
                first = False
                recent_local_metars = deleteme2
            else:
                recent_local_metars = pd.concat([recent_local_metars,deleteme2])

            del deleteme2
            del deleteme

        del local_metars
        del mystations

        recent_local_metars = recent_local_metars.sort_values(["abs_staleness"],ascending=False).reset_index()

        alpha_array = 1-(np.abs(recent_local_metars['abs_staleness'])/metar_time_spread*2).to_numpy()
        alpha_array[ alpha_array<0] = 0
        alpha_array[ alpha_array>1] = 1

        recent_local_metars['alpha'] = alpha_array

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

        ###################################
        #
        # Render Image
        #

        ax = plt.subplot(111, 
                         projection=ccrs.Stereographic(central_latitude    = RadarLatitude, 
                                                       central_longitude   = RadarLongitude, 
                                                       false_easting       = 0.0, 
                                                       false_northing      = 0.0, 
                                                       true_scale_latitude = None, 
                                                       globe=None))

        plt.suptitle(radar.ProductStationName + " ["+radar.ProductStation +"] " +radar.keywords_vocabulary,
                    fontsize=20, color="black")

        ax.set_title(valid_time + "  (" + local_time+")",
                        fontsize=15, color="black")

        ax.set_extent([geospatial_lon_min, 
                       geospatial_lon_max, 
                       geospatial_lat_min, 
                       geospatial_lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(feature    = cfeature.STATES,
                       edgecolor  = 'black',
                       facecolor  = 'none')
        ax.add_feature(feature    = cfeature.COASTLINE,
                       edgecolor  = 'black',
                       facecolor  = 'none')
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

        color_bar = plt.colorbar(filled_cm, 
                     label  = "Reflectivity (dbZ)",
                     shrink = 0.8,
                     pad    = 0.012)
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')           
        plt.setp(cbytick_obj, color='black')

        # Metar Plots

        print("number of obs",len(recent_local_metars))

        for i in range(0,len(recent_local_metars)) :
            single_row = recent_local_metars.loc[i]

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
                                       color='black')

            stationplot.plot_symbol('C', 
                                    np.array([single_row['cloud_eights']]), 
                                    sky_cover,color='black')
            stationplot.plot_symbol('W', 
                                    np.array([single_row['current_wx1_symbol']]), 
                                    current_weather,color='black')

            stationplot.plot_text((2, 0), 
                                  np.array([single_row['ICAO_id']]), 
                                  color='black')
            stationplot.plot_barb(np.array([single_row['eastward_wind']]), 
                                  np.array([single_row['northward_wind']]),
                                 color='black')

            del single_row

        del recent_local_metars 
        
        
        #########################################
        #
        # Insert a Clock
        #
        
        axins = fig.add_axes(rect     =    [0.015,
                                            0.785,
                                            0.12*8/9,
                                            0.12],
                              projection  =  "polar")
        
        time_for_clock = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).time()

        hour   = time_for_clock.hour
        minute = time_for_clock.minute
        second = time_for_clock.second
        
        circle_theta  = np.deg2rad(np.arange(0,360,0.01))
        circle_radius = circle_theta * 0 + 1
        
        if (hour > 12) :
            hour = hour - 12
        
        angles_h = 2*np.pi*hour/12+2*np.pi*minute/(12*60)+2*second/(12*60*60)
        angles_m = 2*np.pi*minute/60+2*np.pi*second/(60*60)
        
        print(time_for_clock)
        print(hour,   np.rad2deg(angles_h))
        print(minute, np.rad2deg(angles_m))

        
        plt.setp(axins.get_yticklabels(), visible=False)
        plt.setp(axins.get_xticklabels(), visible=False)
        axins.spines['polar'].set_visible(False)
        axins.set_ylim(0,1)
        axins.set_theta_zero_location('N')
        axins.set_theta_direction(-1)
        axins.set_facecolor("white")
        axins.grid(False)
        
        axins.plot([angles_h,angles_h], [0,0.6], color="black", linewidth=1.5)
        axins.plot([angles_m,angles_m], [0,0.95], color="black", linewidth=1.5)
        axins.plot(circle_theta, circle_radius, color="darkgrey", linewidth=1)

        
        #
        #########################################


        #. plt.tight_layout()
        plt.subplots_adjust(left   = 0.01, 
                                right  = 0.99, 
                                top    = 0.91, 
                                bottom = .01, 
                                wspace = 0)

        rect = patches.Rectangle(xy        = (0, 0),
                                 width     = percent_done,
                                 height    = 0.01, 
                                 edgecolor = 'black', 
                                 facecolor = "black",
                                 transform = ax.transAxes)
        ax.add_patch(rect)



        plt.savefig("./temp_files_radar/Radar_Loop_Image_"+str(counter).zfill(3)+".png")
        counter = counter + 1
        figure_counter = figure_counter + 1


        plt.close()
        print("=====================")

#
####################################################
####################################################
####################################################


# In[ ]:


####################################################
####################################################
####################################################
#
# Create Individual Map Files for No Radar
#

counter = 0

if (len(sorted(catalog.datasets)) == 0) :

    radarless_time_series       = pd.date_range(time_start-timedelta(hours=1), time_now,freq='5min')

    number_of_figures = len(radarless_time_series)

    figure_counter = 1
    for time in radarless_time_series:
        percent_done = figure_counter / number_of_figures

        print(time, percent_done)
        fig, ax = plt.subplots(1, 1, 
                               figsize=(9, 8),
                               facecolor = 'white')


        time_utc = time


        metar_dataframe['staleness']     = (time_utc-metar_dataframe['date_time'])/ np.timedelta64(1, 'm')
        metar_dataframe['abs_staleness'] =  np.abs( metar_dataframe['staleness'] )

        local_metars = metar_dataframe[(metar_dataframe['abs_staleness'] < metar_time_spread/2) ].copy()
        mystations   = local_metars["ICAO_id"].unique()

        first = True

        for mystation in mystations:
            deleteme  = local_metars[local_metars["ICAO_id"] == mystation].copy().sort_values("abs_staleness")
            deleteme2 = deleteme[deleteme["abs_staleness"]   == np.min(deleteme["abs_staleness"]) ]
            if (first) :
                first = False
                recent_local_metars = deleteme2
            else:
                recent_local_metars = pd.concat([recent_local_metars,deleteme2])

            del deleteme2
            del deleteme

        del local_metars
        del mystations

        recent_local_metars = recent_local_metars.sort_values(["abs_staleness"],ascending=False).reset_index()

        alpha_array = 1-(np.abs(recent_local_metars['abs_staleness'])/metar_time_spread*2).to_numpy()
        alpha_array[ alpha_array<0] = 0
        alpha_array[ alpha_array>1] = 1

        recent_local_metars['alpha'] = alpha_array

        valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H:%M:%S %Z")
        local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        print(valid_time + "  (" + local_time+")")





        #
        ###################################

        ###################################
        #
        # Render Image
        #

        ax = plt.subplot(111, 
                         projection=ccrs.Stereographic(central_latitude    = RadarLatitude, 
                                                       central_longitude   = RadarLongitude, 
                                                       false_easting       = 0.0, 
                                                       false_northing      = 0.0, 
                                                       true_scale_latitude = None, 
                                                       globe=None))

        plt.suptitle("Western South Dakota Surface Obs (Radar Services Down)",
                     fontsize=20, color="black")

        ax.set_title(valid_time + "  (" + local_time+")",
                        fontsize=15, color="black")

        ax.set_extent([geospatial_lon_min, 
                       geospatial_lon_max, 
                       geospatial_lat_min, 
                       geospatial_lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(feature    = cfeature.STATES,
                       edgecolor  = 'black',
                       facecolor  = 'none')
        ax.add_feature(feature    = cfeature.COASTLINE,
                       edgecolor  = 'black',
                       facecolor  = 'none')
        ax.add_feature(feature    = USCOUNTIES, 
                       linewidths = 0.5,
                       edgecolor  = 'black',
                       facecolor  = 'none')

        ax.set_aspect('equal', 'datalim')

 
        # Metar Plots

        print("number of obs",len(recent_local_metars))

        for i in range(0,len(recent_local_metars)) :
            single_row = recent_local_metars.loc[i]

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
                                       color='black')

            stationplot.plot_symbol('C', 
                                    np.array([single_row['cloud_eights']]), 
                                    sky_cover,color='black')
            stationplot.plot_symbol('W', 
                                    np.array([single_row['current_wx1_symbol']]), 
                                    current_weather,color='black')

            stationplot.plot_text((2, 0), 
                                  np.array([single_row['ICAO_id']]), 
                                  color='black')
            stationplot.plot_barb(np.array([single_row['eastward_wind']]), 
                                  np.array([single_row['northward_wind']]),
                                 color='black')

            del single_row

        del recent_local_metars 
        
        #########################################
        #
        # Insert a Clock
        #
        
        axins = fig.add_axes(rect     =    [0.015,
                                            0.785,
                                            0.12*8/9,
                                            0.12],
                              projection  =  "polar")
        
        time_for_clock = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).time()

        hour   = time_for_clock.hour
        minute = time_for_clock.minute
        second = time_for_clock.second
        
        circle_theta  = np.deg2rad(np.arange(0,360,0.01))
        circle_radius = circle_theta * 0 + 1
        
        if (hour > 12) :
            hour = hour - 12
        
        angles_h = 2*np.pi*hour/12+2*np.pi*minute/(12*60)+2*second/(12*60*60)
        angles_m = 2*np.pi*minute/60+2*np.pi*second/(60*60)
        
        print(time_for_clock)
        print(hour,   np.rad2deg(angles_h))
        print(minute, np.rad2deg(angles_m))

        
        plt.setp(axins.get_yticklabels(), visible=False)
        plt.setp(axins.get_xticklabels(), visible=False)
        axins.spines['polar'].set_visible(False)
        axins.set_ylim(0,1)
        axins.set_theta_zero_location('N')
        axins.set_theta_direction(-1)
        axins.set_facecolor("white")
        axins.grid(False)
        
        axins.plot([angles_h,angles_h], [0,0.6], color="black", linewidth=1.5)
        axins.plot([angles_m,angles_m], [0,0.95], color="black", linewidth=1.5)
        axins.plot(circle_theta, circle_radius, color="darkgrey", linewidth=1)


        
        #
        #########################################


        #. plt.tight_layout()
        plt.subplots_adjust(left   = 0.01, 
                                right  = 0.99, 
                                top    = 0.91, 
                                bottom = .01, 
                                wspace = 0)
        
        rect = patches.Rectangle(xy        = (0, 0),
                                 width     = percent_done,
                                 height    = 0.01, 
                                 edgecolor = 'black', 
                                 facecolor = "black",
                                 transform = ax.transAxes)
        ax.add_patch(rect)

        plt.savefig("./temp_files_radar/Radar_Loop_Image_"+str(counter).zfill(3)+".png")
        counter = counter + 1
        figure_counter = figure_counter + 1


        plt.close()
        print("=====================")

#
####################################################
####################################################
####################################################


# In[ ]:


##################################################
#
# Convert PNGs into an Animated GIF
#

os.system("convert -delay 25 " + 
          "./temp_files_radar/Radar_Loop_Image_*.png"  + 
          " " + 
          "./graphics_files/RealTime_Radar_Loop.gif")


#
##################################################


# In[ ]:




