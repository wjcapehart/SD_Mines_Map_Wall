#!/usr/bin/env python
# coding: utf-8

# # TBW Radar Loop for the Map Wall
# 
# Creates an Animated Plot for Radar and Station Models
# 
# This script has been cloned repeatedly to manage other radar sites
# 
# An image sample is shown here:
# ![KUNR_Radar_2023-12-25_2014Z.gif](http://kyrill.ias.sdsmt.edu/wjc/eduresources/KUNR_Radar_2023-12-25_2014Z.gif)

# ## Libraries
# 
# * Classic General Use Python Libraries
#   * [Numpy](https://numpy.org): The fundamental package for scientific computing with Python
#   * [Matplotlib](https://matplotlib.org): Comprehensive library for creating static, animated, and interactive visualizations in Python
#   * [Pandas](https://pandas.pydata.org): Open source data analysis and manipulation tool for Tables
# * Unidata Python Libraries 
#   * [Unidata's Siphon](https://unidata.github.io/siphon/latest/): A collection of Python utilities for downloading data from remote data services
#   * [Unidata's MetPy](https://unidata.github.io/MetPy/latest/): Tools for reading, visualizing, and performing calculations with weather data
# * Niche Python Libraries
#   * [AirportsData](https://github.com/mborsetti/airportsdata/blob/main/README.rst): Extensive database of location and timezone data for nearly every airport and landing strip in the world.
#   * [TimezoneFinder](https://timezonefinder.readthedocs.io/en/latest/): a fast and lightweight python package for looking up the corresponding timezone for given coordinates on earth entirely offline.
#   * [JobLib](https://joblib.readthedocs.io): For simple parallel computing management
#   * [Cartopy](https://scitools.org.uk/cartopy/docs/latest/): Geospatial data processing in order to produce maps and other geospatial data analyses.
# 

# In[1]:


####################################################
####################################################
####################################################
#
# Libraries
#

# Internal Python Libraries

from   datetime          import datetime, timedelta, timezone
import os                as     os
import platform          as     platform
import pathlib           as     pathlib
import urllib.request
import shutil

# Classic General Use Libraries

import numpy              as np
import matplotlib.pyplot  as plt
import pandas             as pd

# Specialized Niche Libraries

import airportsdata       as airpt
import timezonefinder     as tzf

from   joblib             import Parallel, delayed


import cartopy.crs       as ccrs
import cartopy.feature   as cfeature

# Unidata Libraries

import metpy.io           as mpio
import metpy.calc         as mpcalc
import matplotlib.patches as patches

from   metpy.plots        import colortables, USCOUNTIES, StationPlot, sky_cover,current_weather
from   metpy.units        import units
from   matplotlib.colors  import ListedColormap


import siphon.catalog     as siphcat  
import siphon.ncss        as siphncss

from   siphon.cdmr        import Dataset
from   siphon.radarserver import get_radarserver_datasets, RadarServer
from   siphon.catalog     import TDSCatalog

#
####################################################
####################################################
####################################################


# ## Geospatial Information (User Customization Area)
# 
# Script can be customized for a given location so that the parent METAR region address and Radar ID can be kept separate.
# 
# The user must modify both.  
# 
# The user should also 

# In[2]:


####################################################
####################################################
####################################################
#
#  Spatial Information for Specific Stations
#

# Station ID for METARS Addressing; Radar ID for Radar Data

station_id = "TPA"
radar_id   = "TBW"


#
#  Radar Domain Center
# 

Fixed_RadarLatitude      =   27.705
Fixed_RadarLongitude     = -82.402

#
#  Assigned Radar Domain Extents
# 

Fixed_geospatial_lat_min =    25.63982
Fixed_geospatial_lat_max =   29.77018

Fixed_geospatial_lon_min =  -84.73692
Fixed_geospatial_lon_max =  -80.06708

RadarLatitude      = Fixed_RadarLatitude
RadarLongitude     = Fixed_RadarLongitude

geospatial_lat_min = Fixed_geospatial_lat_min
geospatial_lat_max = Fixed_geospatial_lat_max
geospatial_lon_max = Fixed_geospatial_lon_max
geospatial_lon_min = Fixed_geospatial_lon_min
        
#
####################################################
####################################################
####################################################


# ## Colormaps, Colors and Branding
# 
# South Dakota Mines has distrinct branding including including [prescribed colors](https://brand.sdsmt.edu/visual-design/colors/) accessable via HEX codes. 
# 
# These colors are applied to the [matplotlib.rcParam](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) customizations.
# 
# Unidata provides a swell set of meteorology-based [color tables](https://unidata.github.io/MetPy/latest/api/generated/metpy.plots.ctables.html).  This script uses the a classic storm reflectivity field. 
# 
# 
# We're using the "NWSStormClearReflectivity"
# 
# ![metpy-plots-ctables](https://unidata.github.io/MetPy/latest/_images/metpy-plots-ctables-1.png)
# 
# This table is then modified using a simple alpha transparency to fade out lower dBz levels.
# 
# ![Faded NWSStormClearReflectivity](http://kyrill.ias.sdsmt.edu/wjc/eduresources/NWSStormClearReflectivity.png)

# In[3]:


####################################################
####################################################
####################################################
#
# Mines Colors and Fonts
#

Mines_Blue = "#002554"


plt.rcParams.update({'text.color'      : Mines_Blue,
                     'axes.labelcolor' : Mines_Blue,
					 'axes.edgecolor'  : Mines_Blue,
					 'xtick.color'     : Mines_Blue,
					 'ytick.color'     : Mines_Blue})

#
# Radar Color Map
#

alpha_factor               =     0.05
alpha_color_table_fraction =     0.70
radar_floor                = -9999.99

# Choose colormap
norm, cmap = colortables.get_with_steps(name = "NWSStormClearReflectivity", 
                                        start = -20.0, 
                                        step  =   0.5)
n_colors = cmap.N

# Get the colormap colors

cmap = cmap(np.arange(cmap.N))


# Set alpha

cmap[0:int(n_colors*alpha_color_table_fraction),-1] = np.linspace(0, 1, int(n_colors*alpha_color_table_fraction))

# Create new colormap

cmap = ListedColormap(cmap)

#
####################################################
####################################################
####################################################


# ## File Control Management, House Keeping and Time Management
# 
# Directories are set.  
# 
# Temporary temporary directories are clear.  
# 
# Siphon and HTTPS extraction times from UCAR are collected
# 
# Currently, we only do a three-hour loop with the assumption that we should expect a radar image every four minutes.  If not available, these will be replaced by a plain map of the radar coverage area.
# 

# In[4]:


####################################################
####################################################
####################################################
#
#  File Control and Cleanup
#

MAINDIR = os.getcwd() + "/"
print(MAINDIR)

os.system("rm -v ./temp_files_radar/*")

metar_collection = 'https://thredds.ucar.edu/thredds/catalog/nws/metar/ncdecoded/catalog.xml?dataset=nws/metar/ncdecoded/Metar_Station_Data_fc.cdmr'
synop_collection = "https://thredds.ucar.edu/thredds/catalog/nws/synoptic/ncdecoded/catalog.xml?dataset=nws/synoptic/ncdecoded/Surface_Synoptic_Point_Data_fc.cdmr" 

#  Time Control

time_now   =  datetime.now(tz=timezone.utc)
time_start = time_now - timedelta(hours=3)

print(time_start)
print(time_now)

radar_delta_t = 4 # min

metar_time_spread = 60

siphon_time_series       = pd.date_range(time_start- timedelta(hours=1), time_now,freq='h')
siphon_pulls_YYYYMMDD_HH = siphon_time_series.strftime("%Y%m%d_%H00")

print(siphon_pulls_YYYYMMDD_HH)

#
####################################################
####################################################
####################################################


# ## Airport Database Access
# 
# To pull the latitude and longitude of METAR search area, the latitude and longitude of the stations are pulled from [Mike Borsetti's airport database](https://github.com/mborsetti/airportsdata/blob/main/README.rst).  
# 
# In retrospect, we may not even need this if we have a manually set radar site ID since we now have the option to plot the stations on the radar map should the radar images be missing.
# 
# The local timezone is pulled from the using the station_id.  This uses [Jannik Kissinger's timezone finder](https://timezonefinder.readthedocs.io/en/latest/) to link the station_id's latitude and longitude to a timezone.

# In[5]:


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


# ## Siphon/HTTP Pulls for the METARs
# 
# METARs are pulled from Unidata's NOAAPORT text data.
# 
# This is looped over the 4-hr product period.

# In[6]:


####################################################
####################################################
####################################################
#
# Pull Text-Based METARS from UNIDATA NOAAPORT Experimental Site
#

# cat = siphcat.TDSCatalog('https://thredds-dev.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml')

#
# Extraction File Text Model
#
# https://thredds-test.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar/metar_20210924_0000.txt


first = True
for datehour in siphon_pulls_YYYYMMDD_HH:

    # URLs & Local Work File Names
    
    metar_url  = "https://thredds-dev.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar/metar_"+datehour+".txt"
    metar_file = "./temp_files_radar/metar_"+datehour+".txt"
    
    path_to_file = pathlib.Path(metar_file)
    
    print(path_to_file, path_to_file.is_file())

    # Pull File 
    
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

# Cookiecut the Radar Domain Metars and Clean Data Table

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

# Convert Units for Metars (cloud cover, visibility, temps etc)

metar_dataframe['cloud_eights']          = (8 * metar_dataframe['cloud_eights']/10).fillna(10).values.astype(int)
metar_dataframe['air_temperature']       = ( metar_dataframe['air_temperature'] * 9/5) + 32
metar_dataframe['dew_point_temperature'] = ( metar_dataframe['dew_point_temperature'] * 9/5) + 32
metar_dataframe['visibility_sm']         = np.round(metar_dataframe['visibility_sm'] / 1609.34,decimals=1)

#
####################################################
####################################################
####################################################


# ## Pulling Radar Data Times and Radar Objects
# 
# Pulling all radar data using the 
# 
# Many thanks to Ryan May at Unidata on the tips here!
# 
# [https://stackoverflow.com/questions/69923496/sorting-a-tdscatalog-list-for-loops-and-animations](https://stackoverflow.com/questions/69923496/sorting-a-tdscatalog-list-for-loops-and-animations)
# 

# In[7]:


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

catalog         = rs.get_catalog(query)

datasets_sorted = sorted(catalog.datasets)

list(datasets_sorted)

figure_counter    = 1
number_of_figures = len(sorted(catalog.datasets))

number_of_figures = len(sorted(catalog.datasets))
print("Number of Radar Images=",number_of_figures)

dates_for_radar  = []

print("start time: ",time_start)

for name in datasets_sorted:
    
    datetime_string = pd.to_datetime(datetime.strptime(name[15:28], "%Y%m%d_%H%M")).tz_localize(tz="UTC")
    dates_for_radar.append(datetime_string)

    if (name == datasets_sorted[0]):
        print(" - ",datetime_string, dates_for_radar[-1]-time_start)
    else:
        print(" - ",datetime_string, (dates_for_radar[-1]-dates_for_radar[-2]))
        
print("  end time: ",time_now)

#
####################################################
####################################################
####################################################


# ## The BIG RADAR PLOTTING FUNCTION
# 
# We use a parallel loop through the available satellite times to speed product generation.  This requires the graph products to be put into a plotting function.
# 
# We will now outline the procedure here in slow motion.
# 
# **For each name index object for the radar data...**
# 1.  Restate the specialized colors
# 2.  Touch the object on the remote THREDDS server
# 3.  Create the figure "workstation"/wks (and no I will never stop calling it that.)
# 4.  Conditional Check (to see if the object is truly accessible - this often isn't the case when the service address is heavily throttled such as during a hurricane landfall).
#     1. If go, then...
#         1. Pull the radar object...
#         2. Extract range, reflectance, azimuth, radar time,
#         3. Overlay a Hanning window over the graphics space because squares on the mapwall are make things look ugly.
#     2. If an error is found on the reading then...
#         1. Extract an error statement and record the time of the errant record.
# 5.  Assign a "staleness" parameter for each METAR record in the pandas dataframe for this timestep.  This will be used for "fading" station models in the radar plot as they are further and further from the radar observation time and make them darker and darker as you approach the radar observation time.
# 6.  Now plot the map
#     1.  Drop the Axes with map projection data
#     2.  Load the title data as well as the map vectors
#     3.  If there is an available radar image, plot it.  Otherwise, move on to the next step...
#     4.  Plot the station models
#     5.  Add a clock in the upper corner and a status bar at the bottom
#     6.  Save it to a indexed file.
# 7.  And we're done!
# 

# In[8]:


####################################################
####################################################
####################################################
#
# The Big Radar Plotting Function
#

def radar_plotting_func(name_index):

    ###################################
    #
    # Repating Mines Colors and Fonts
    #
    
    Mines_Blue = "#002554"

    plt.rcParams.update({'text.color'      : Mines_Blue,
						 'axes.labelcolor' : Mines_Blue,
						 'axes.edgecolor'  : Mines_Blue,
						 'xtick.color'     : Mines_Blue,
						 'ytick.color'     : Mines_Blue})

    #
    ###################################
    
    ###################################
    #
    # Create the Status Bar Completeness Calculation
    #
    
    percent_done = (name_index+1.) / number_of_figures

    print("image# = ", (name_index+1), 
          "/",number_of_figures, 
          " (", (percent_done*100), "%)")

    #
    ###################################
    
    ###################################
    #
    # Touch the dataset.
    #
    
    name = sorted(catalog.datasets)[name_index]
    ds   = catalog.datasets[name]
    print(ds)

    #
    ###################################
    
    ###################################
    #
    # Enable the Figure Workspace
    #
    
    fig = plt.figure(figsize   =  (9, 8),
                     facecolor = 'white')

    #
    ###################################
    
    ###################################
    #
    # Crack the Dataset If there is a failure, swear like mom (and record the time)
    #
    
    try:

        # Crack the dataset and pull range, azimuth and reflectivity and make a mapping grid

        radar    = Dataset(ds.access_urls['CdmRemote'])

        rng      = radar.variables['gate'][:] 
        az       = radar.variables['azimuth'][:]
        ref      = radar.variables['BaseReflectivityDR'][:]
        ref      = np.ma.array(ref, mask=np.isnan(ref))
        time_utc = datetime.strptime(radar.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ")
        x        = rng * np.sin(np.deg2rad(az))[:, None]
        y        = rng * np.cos(np.deg2rad(az))[:, None]
        
        ref[ ref< radar_floor] = np.nan

        ny = ref.shape[0]
        nx = ref.shape[1]      
        alpha2d = np.sqrt(np.outer(np.abs(np.hanning(ny)),np.abs(np.hanning(nx))))
        alpha2d = np.where(alpha2d>alpha_factor,alpha_factor,alpha2d)
        alpha2d = alpha2d / alpha_factor

    except:
        print("poopie: ", name, name[15:-5])
                                            #Level3_UDX_N0B_20220819_0415.nids
        time_utc = datetime.strptime(name[15:-5],"%Y%m%d_%H%M")

    #
    ###################################
    
    ###################################
    #
    # Pull the observation window for this image's window and also record relative staleness for plotting.
    #
    
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

    alpha_array                 = 1-(np.abs(recent_local_metars['abs_staleness'])/metar_time_spread*1.2).to_numpy()

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
    # Reset the Spatial Extents for either radar/station or just-station plot.
    #
    
    try:
        noradar = False
        RadarLatitude      = radar.RadarLatitude
        RadarLongitude     = radar.RadarLongitude
        geospatial_lat_min = radar.geospatial_lat_min
        geospatial_lat_max = radar.geospatial_lat_max
        geospatial_lon_min = radar.geospatial_lon_min
        geospatial_lon_max = radar.geospatial_lon_max
        print("RadarLatitude      = ", RadarLatitude)
        print("RadarLongitude     = ", RadarLongitude)
        print("geospatial_lat_min = ", geospatial_lat_min)
        print("geospatial_lat_max = ", geospatial_lat_max)
        print("geospatial_lon_max = ", geospatial_lon_min)
        print("geospatial_lon_min = ", geospatial_lon_max)
    except:
        noradar = True
        RadarLatitude= Fixed_RadarLatitude
        RadarLongitude= Fixed_RadarLongitude
        geospatial_lat_min= Fixed_geospatial_lat_min
        geospatial_lat_max= Fixed_geospatial_lat_max
        geospatial_lon_max= Fixed_geospatial_lon_max
        geospatial_lon_min= Fixed_geospatial_lon_min

    #
    ###################################

    ###################################
    #
    # Plot Radar Image
    #

    ax = plt.subplot(1,1,1, 
                     projection=ccrs.Stereographic(central_latitude    = RadarLatitude, 
                                                   central_longitude   = RadarLongitude, 
                                                   false_easting       = 0.0, 
                                                   false_northing      = 0.0, 
                                                   true_scale_latitude = None, 
                                                   globe=None),
                     frameon   = False)

    #
    # Set the Spatial Bounds for the Map Plot
    #

    ax.set_extent([geospatial_lon_min, 
                   geospatial_lon_max, 
                   geospatial_lat_min, 
                   geospatial_lat_max], crs=ccrs.PlateCarree())

    ax.set_aspect(aspect     =   'equal', 
                  adjustable = 'datalim')

    #
    # Always label your plots.. (note the divergent tities)
    #
    
    try:
        plt.suptitle(radar.ProductStationName + " [" +
                     radar.ProductStation     + "] " +
                     radar.keywords_vocabulary,
                     fontsize =         20, 
                     color    = Mines_Blue)
    except:
        plt.suptitle("Surface Obs (Radar Services Down)",
                     fontsize = 20, 
                     color    = Mines_Blue)

    ax.set_title(valid_time + 
                 "  (" +
                 local_time+")",
                 fontsize =         15, 
                 color    = Mines_Blue)

    #
    # Add Map Vector Elements
    #

    ax.add_feature(feature    = cfeature.STATES,
                   edgecolor  = Mines_Blue,
                   facecolor  = 'none')
    ax.add_feature(feature    = cfeature.COASTLINE,
                   edgecolor  = Mines_Blue,
                   facecolor  = 'none')
    ax.add_feature(feature    = USCOUNTIES, 
                   linewidths = 0.5,
                   edgecolor  = Mines_Blue,
                   facecolor  = 'none')
    
    try:

        #
        # Start with the Radar Mesh
        #
        
        filled_cm = ax.pcolormesh(x, 
                                  y, 
                                  ref,
                                  norm = norm, 
                                  cmap = cmap)
        
        color_bar = plt.colorbar(filled_cm, 
                                 label  = "Reflectivity (dbZ)",
                                 shrink = 0.8,
                                 pad    = 0.012)
        
        cbytick_obj = plt.getp(obj      = color_bar.ax.axes, 
                               property =     'yticklabels')   
        
        #plt.setp(obj   = cbytick_obj, 
        #         color = Mines_Blue)
        
        noradar = False

    except:

        print("blank map")
        noradar = True

    #
    # Loop over METAR Station Models
    #
    
    print(         "number of obs",
          len(recent_local_metars))

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
                                   color = Mines_Blue)
        stationplot.plot_parameter('SW', 
                                   np.array([single_row['dew_point_temperature']]), 
                                   color = Mines_Blue)
        stationplot.plot_parameter('NE', 
                                   np.array([single_row['air_pressure_at_sea_level']]),  
                                   color = Mines_Blue)
        stationplot.plot_parameter('SE',
                                   np.array([single_row['staleness']]),
                                   color = Mines_Blue)

        stationplot.plot_symbol('C', 
                                np.array([single_row['cloud_eights']]), 
                                sky_cover,
                                color = Mines_Blue)
        stationplot.plot_symbol('W', 
                                np.array([single_row['current_wx1_symbol']]), 
                                current_weather,
                                color = Mines_Blue)

        stationplot.plot_text((2, 0), 
                              np.array([single_row['ICAO_id']]), 
                              color = Mines_Blue)
        stationplot.plot_barb(np.array([single_row['eastward_wind']]), 
                              np.array([single_row['northward_wind']]),
                              color = Mines_Blue)

        del single_row

    del recent_local_metars 

    #
    ###################################

    ###################################
    #
    # Insert a Clock
    #

    axins = fig.add_axes(rect        = [0.015,
                                        0.785,
                                        0.12*8/9,
                                        0.12],
                         projection  =  "polar")

    time_for_clock = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).time()

    hour   = time_for_clock.hour
    minute = time_for_clock.minute
    second = time_for_clock.second

    if ((hour >= 6) and (hour < 18)):
        Clock_Color = Mines_Blue
        Clock_BgndC = "white"           
    else:
        Clock_Color = "white"
        Clock_BgndC = Mines_Blue               

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
    axins.set_facecolor(Clock_BgndC)
    axins.grid(False)

    axins.plot([angles_h,angles_h], [0,0.60], color=Clock_Color, linewidth=1.5)
    axins.plot([angles_m,angles_m], [0,0.95], color=Clock_Color, linewidth=1.5)
    axins.plot(circle_theta, circle_radius, color="darkgrey", linewidth=1)

    #
    ###################################

    ###################################
    #
    # Create a status bar at the bottom of the map axes
    #

    if (not noradar):
        plt.subplots_adjust(left   = 0.01, 
                                right  = 0.99, 
                                top    = 0.91, 
                                bottom = .01, 
                                wspace = 0)
    else:
        ax.set_position([0.01, 0.01, 0.82124, 0.9])

    rect = patches.Rectangle(xy        = (0, 0),
                             width     = percent_done,
                             height    = 0.01, 
                             edgecolor = Mines_Blue, 
                             facecolor = Mines_Blue,
                             transform = ax.transAxes)
    ax.add_patch(rect)

    #
    ###################################

    ###################################
    #
    # And we're done!
    #

    plt.savefig(fname       = "./temp_files_radar/Radar_Loop_Image_"+str(name_index).zfill(3)+".png",
                facecolor   = 'white', 
                transparent =   False)

    plt.close()
    
    print("=====================")

    #
    ###################################

#
####################################################
####################################################
####################################################
  


# ## In-script Test of the Big Plotting Function
# 
# If the first instance of the big plotting function returns an error, return a diagnostic message which indicates that there are no radar files available during the current run.
# 

# In[9]:


#try: 
radar_plotting_func(0)
#except:
#    print("## No radar files to plot")


# ## Run the Radar Plotting Tasks in Parallel
# 
# Use th [Joblib.Parallel()](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html) class to run the radar script in parallel for speedy results.

# In[10]:


####################################################
####################################################
####################################################
# 
# Create Individual Map Files for Radar
#


number_of_figures = len(sorted(catalog.datasets))

if (number_of_figures > 0) :
    

    n_jobs = 8
    
    
    start_parallel = datetime.now(tz=timezone.utc)
    print("Starting Parallel : ",start_parallel)


    Parallel(n_jobs=n_jobs)(delayed(radar_plotting_func)(name_I) for name_I in range(number_of_figures))
    
    end_parallel = datetime.now(tz=timezone.utc)
    print("Ending Parallel = ", end_parallel)
    print("  Parallel Time = ", (end_parallel - start_parallel))



print("Done")
#
####################################################
####################################################
#####################################################


# ## Plain Station Plot Full Series
# 
# In the event that there are *zero* radar plots.  The Big Plot Function will not be executed.  A faster simpler plot series is below that follows the same tasks in the Big Plot Function with the exception of pulling the radar data and plotting it.  The only thing that will be rendered will be the station plots using the same transparent=stale trick.

# In[11]:


#####################################################
####################################################
#
# Create Individual Map Files for No Radar
#

####################################################
####################################################
####################################################
#
# Mines Colors and Fonts
#

Mines_Blue = "#002554"


plt.rcParams.update({'text.color'      : Mines_Blue,
                     'axes.labelcolor' : Mines_Blue,
					 'axes.edgecolor'  : Mines_Blue,
					 'xtick.color'     : Mines_Blue,
					 'ytick.color'     : Mines_Blue})


#
####################################################
####################################################
####################################################


if (len(sorted(catalog.datasets)) == 0) :

    radarless_time_series       = pd.date_range(time_start-timedelta(hours=1), time_now,freq='5min')

    number_of_figures = len(radarless_time_series)

    for time_index in range(len(radarless_time_series)):
        
        time = radarless_time_series[time_index]
        percent_done = (time_index + 1) / number_of_figures

        print(time, percent_done)


        fig = plt.figure(figsize   =  (9, 8),
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

        ax = plt.subplot(1,1,1,
                         frame_on=False,
                         projection=ccrs.Stereographic(central_latitude    = RadarLatitude, 
                                                       central_longitude   = RadarLongitude, 
                                                       false_easting       = 0.0, 
                                                       false_northing      = 0.0, 
                                                       true_scale_latitude = None, 
                                                       globe=None))

        plt.suptitle("Western South Dakota Surface Obs (Radar Services Down)",
                     fontsize=20, color=Mines_Blue)

        ax.set_title(valid_time + "  (" + local_time+")",
                        fontsize=15, color=Mines_Blue)

        ax.set_extent([geospatial_lon_min, 
                       geospatial_lon_max, 
                       geospatial_lat_min, 
                       geospatial_lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(feature    = cfeature.STATES,
                       edgecolor  = Mines_Blue,
                       facecolor  = 'none')
        ax.add_feature(feature    = cfeature.COASTLINE,
                       edgecolor  = Mines_Blue,
                       facecolor  = 'none')
        ax.add_feature(feature    = USCOUNTIES, 
                       linewidths = 0.5,
                       edgecolor  = Mines_Blue,
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
                                       color=Mines_Blue)
            stationplot.plot_parameter('SW', 
                                       np.array([single_row['dew_point_temperature']]), 
                                       color=Mines_Blue)
            stationplot.plot_parameter('NE', 
                                       np.array([single_row['air_pressure_at_sea_level']]),  
                                       color=Mines_Blue)
            stationplot.plot_parameter('SE',
                                       np.array([single_row['staleness']]),
                                       color=Mines_Blue)

            stationplot.plot_symbol('C', 
                                    np.array([single_row['cloud_eights']]), 
                                    sky_cover,color=Mines_Blue)
            stationplot.plot_symbol('W', 
                                    np.array([single_row['current_wx1_symbol']]), 
                                    current_weather,color=Mines_Blue)

            stationplot.plot_text((2, 0), 
                                  np.array([single_row['ICAO_id']]), 
                                  color=Mines_Blue)
            stationplot.plot_barb(np.array([single_row['eastward_wind']]), 
                                  np.array([single_row['northward_wind']]),
                                 color=Mines_Blue)

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
        
        axins.plot([angles_h,angles_h], [0,0.6], color=Mines_Blue, linewidth=1.5)
        axins.plot([angles_m,angles_m], [0,0.95], color=Mines_Blue, linewidth=1.5)
        axins.plot(circle_theta, circle_radius, color=Mines_Blue, linewidth=1)


        
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
                                 edgecolor = Mines_Blue, 
                                 facecolor = Mines_Blue,
                                 transform = ax.transAxes)
        ax.add_patch(rect)

        

        plt.savefig("./temp_files_radar/Radar_Loop_Image_"+str(time_index).zfill(3)+".png",
                        facecolor   = 'white', 
                        transparent =   False)

        plt.close()
        print("=====================")

#
####################################################
####################################################
#####################################################


# ## Making an Animated GIF
# 
# Rather than using Python's animation feature (which is wanting in some areas, we will use the classic [ImageMagik "convert" utility](https://imagemagick.org/script/convert.php) to create a high-quality animated gif. 
# 

# In[12]:


##################################################
#
# Convert PNGs into an Animated GIF
#

print("creating " + MAINDIR + "./processing_radar_gif.sh")
with open(MAINDIR + "./processing_radar_gif.sh", 'w') as f:
    print("#!/bin/bash", file =  f)
    print(". /home/wjc/.bashrc", file = f)
    print("cd " + MAINDIR, file =  f) 
    print("convert -delay 25 " + 
          "./temp_files_radar/Radar_Loop_Image_*.png"  + 
          " " + 
          "./graphics_files/staging_area/RealTime_Radar_Loop.gif", file =  f) 
    print("mv -fv ./graphics_files/staging_area/RealTime_Radar_Loop.gif ./graphics_files/", file =  f) 
    print("echo MAIN:RADAR::: We^re Outahere Like Vladimir", file =  f) 

os.system("chmod a+x " + MAINDIR + "./processing_radar_gif.sh")
os.system(MAINDIR + "./processing_radar_gif.sh > ./processing_radar_gif.LOG 2>&1 ")
os.system("date")
print()

#
##################################################


# # Finally...
# As they used to say with the MM5 "decks"...

# In[13]:


print("We're Out a Here Like Vladimir!")


# In[ ]:




