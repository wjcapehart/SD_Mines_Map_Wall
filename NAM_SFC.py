#!/usr/bin/env python
# coding: utf-8

# # NAM SFC

# In[ ]:


####################################################
####################################################
####################################################
#
# Libraries
#

import numpy             as np
import datetime          as dt
import matplotlib        as mpl
import matplotlib.pyplot as plt
import ftplib            as ftplib
import urllib.request    as urllibreq
import datetime          as datetime
import os                as os
import platform          as platform
import socket            as socket
import xarray            as xr
import netCDF4           as nc4
import metpy             as metpy
import pathlib           as pathlib
import numpy             as np
import cftime            as cftime
import netCDF4           as nc4
import metpy             as metpy
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import pandas            as pd
import pyproj            as pyproj
import scipy.ndimage     as ndimage

def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=20,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, size=12, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=transform)

#
####################################################
####################################################
####################################################


# 

# In[ ]:


####################################################
####################################################
####################################################
#
# File Control
#

png_processing_directory = "./temp_files_namsfc/"

gif_file_name = "./graphics_files/NAM_SFC.gif"

png_file_root = png_processing_directory + "NAM_SFC_"

os.system("rm -v "+ png_processing_directory +"*")


#
####################################################
####################################################
####################################################


# ### Timings for each run
# 
# The Realtime WRF is generated every 6 hr at best.  The model takes 3 hr to 
# 
# | Model Product Time (UTC) | Wallclock Start Time (UTC) |
# |:------------------------:|:--------------------------:|
# |        00 UTC            |        03 UTC              |
# |        06 UTC            |        09 UTC              |
# |        12 UTC            |        15 UTC              |
# |        18 UTC            |        21 UTC              |

# In[ ]:


####################################################
####################################################
####################################################
#
# Identify Specific Run by Wall Clock Window
#

lag_hours = 3

current_datetime = datetime.datetime.utcnow()


current_datetime_lag3 = current_datetime - datetime.timedelta(hours=lag_hours)


if (current_datetime.day == current_datetime_lag3.day):
    if (current_datetime_lag3.hour < 6):
        fx_hour =  0
    elif (current_datetime_lag3.hour < 12):
        fx_hour =  6
    elif (current_datetime_lag3.hour < 18):
        fx_hour = 12
    else:
        fx_hour = 18

    model_start_datetime = datetime.datetime(year  = current_datetime_lag3.year,
                                             month = current_datetime_lag3.month, 
                                             day   = current_datetime_lag3.day, 
                                             hour  = fx_hour)     
else:
    fx_hour = 18
    model_start_datetime = datetime.datetime(year  = current_datetime_lag3.year,
                                             month = current_datetime_lag3.month, 
                                             day   = current_datetime_lag3.day, 
                                             hour  = fx_hour)

    


print("           Current Time ", current_datetime)
print("NAM Forecast Start Time ", model_start_datetime)





model_thredds_retrieval_date   = model_start_datetime.strftime("%Y%m%d_%H00")

nam_opendap_url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NAM/CONUS_40km/conduit/NAM_CONUS_40km_conduit_" + model_thredds_retrieval_date + ".grib2"
                  #https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NAM/CONUS_40km/conduit/NAM_CONUS_40km_conduit_20220124_0000.grib2
print(" ")

print(nam_opendap_url)

#
####################################################
####################################################
####################################################


# ## Crack open GRIB array with Xarray

# In[ ]:


####################################################
####################################################
####################################################
#
# Crack open the UCAR thredds NOMADS array.
#

nam_model = xr.open_dataset(nam_opendap_url)

nam_model = nam_model.metpy.parse_cf()



data_crs = nam_model.metpy_crs.metpy.cartopy_crs


eastings2d, northings2d = np.meshgrid(nam_model["x"],nam_model["y"])

pyproj_nam = pyproj.Proj(data_crs)

lon2d, lat2d = pyproj_nam(eastings2d,
                          northings2d,
                          inverse=True)




#
####################################################
####################################################
####################################################


# ## Panel Displays
# 
# 500-hPa Vorticity/Heights
# 
# * 500-hPa Absolute Vorticity
# * 500-hPa Isobaric Heights
# 
# Thickness / MSLP
# 
# * Mean Sea Level Pressure
# * 1000-500 hPa Thickness
# 
# 850-hPa Heights and Humidity
# 
# * 850-hPA Heights
# * 850-hPA Relative Humidity
# 
# Precipitation / Vertical Velocity
# 
# * 12-hr Precipitation
# * 700-hPa Vertical Velocity
# 

# ## Fetch Data for Panel Displays

# In[ ]:


####################################################
####################################################
####################################################
#
# Extract Map Fields
#

k_0500hpa_vort = 2

k_1000hpa_height = 38
k_0850hpa_height = 32
k_0700hpa_height = 26
k_0500hpa_height = 18

# MSLP - Thickness

mslp                = nam_model[ "Pressure_reduced_to_MSL_msl"]
mslp.values         = mslp.values / 100
mslp.attrs["units"] = "hPa"

heights_500        = nam_model["Geopotential_height_isobaric"][:,k_0500hpa_height,:,:]
heights_500.values = heights_500.values / 10.
heights_500.attrs["units"] = "dam"

thickness        = heights_500.copy() 
thickness.values = heights_500.values - nam_model["Geopotential_height_isobaric"][:,k_1000hpa_height,:,:].values/10
thickness.attrs["long_name"] = "1000-500 hPa Thickness"
thickness.attrs["units"]     = "dam"




#
####################################################
####################################################
####################################################


# In[ ]:


####################################################
####################################################
####################################################
#
# Plot Sample Map for Records
#

start_time = nam_model["reftime"].values
time_dim   = mslp.dims[0]
times_utc  = mslp.coords[time_dim].to_numpy()
fxx        = (times_utc-start_time)/ np.timedelta64(1, 'h')


for i in range(len(times_utc)) :

    tz='America/Denver'
    time_utc   = times_utc[i]
    valid_time = pd.to_datetime(start_time).tz_localize(tz="UTC").strftime("%Y-%m-%d %H00 %Z")
    local_time = pd.to_datetime(times_utc[i]).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H00 %Z")

    plot_label = "NAM Sea-Level Pressure & 1000-500-hPa Thickness"
    time_label = valid_time + " F" + str(int(fxx[i])).zfill(2)+  " (" + local_time + ")"
    print(time_label)

    fig = plt.figure(figsize   = (9.5, 8), 
                     facecolor = 'white')
    
    plt.suptitle(plot_label,
                     fontsize = 20, 
                     color    = "black")
  
    # add a plot element just one field

                       # nrows, ncols, index [which oddly starts with one (go fig)],
    ax = fig.add_subplot(    1,     1,     1, 
                         projection = data_crs)
    

   

    # of you use the coastlines and add_feature you can see it does 
    #   the full cone and our place on it


    ax.coastlines(resolution = 'auto',
                  linewidths =  0.75)


    ax.add_feature(cfeature.STATES.with_scale('110m'), 
                       linewidths = 0.5,
                       facecolor  = 'none' , 
                       edgecolor  = 'black')


    contourf_levels = np.arange(start =  480,
                                stop  =  613,
                                step  =    6)

    contourf_plot = thickness[i,:,:].plot.contourf(cmap        = plt.cm.turbo,
                                                   extend      = 'both',
                                                   levels      = contourf_levels,
                                                   cbar_kwargs = {"orientation" : "horizontal",
                                                                  "pad"         : 0.01,
                                                                  "shrink"      : 0.75,
                                                                  "aspect"      :   30})    



    contour_plot2 = thickness[i,:,:].plot.contour(colors     =        "white",
                                            linewidths =           2.5, 
                                            levels     = np.array([540]))


    contour_plot3 = thickness[i,:,:].plot.contour(colors     =        "white",
                                            linewidths =           0.75, 
                                            levels     = contourf_levels)


        
    ax.clabel(contour_plot2, fontsize="xx-large")

  
    

    
    contour_levels = np.arange(start =  900, 
                               stop  = 1090, 
                               step  =    4)
    
    smoothed = mslp[i,:,:].copy()
    
    smoothed.values = ndimage.gaussian_filter( mslp[i,:,:].values, sigma=0.75, order=0)

    contour_plot = smoothed[:,:].plot.contour(colors     =        "black",
                                            linewidths =           0.75, 
                                            levels     = contour_levels)


    ax.clabel(contour_plot)

    # Use definition to plot H/L symbols
    plot_maxmin_points(lon2d, lat2d, smoothed, 'max', 50, symbol='H', color='k',  transform=ccrs.PlateCarree())
    plot_maxmin_points(lon2d, lat2d, smoothed, 'min', 25, symbol='L', color='k', transform=ccrs.PlateCarree())

    
    ax.set_title(time_label,
                        fontsize=15, color="black")


    
    plt.tight_layout()

    plt.savefig(png_file_root+str(i).zfill(2)+".png")
    plt.close()



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
          png_file_root + "*.png"  + 
          " " + 
          gif_file_name)


#
##################################################


# In[ ]:




