#!/usr/bin/env python
# coding: utf-8

# # NAM Prec

# In[1]:


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



#
####################################################
####################################################
####################################################


# In[2]:


####################################################
####################################################
####################################################
#
# File Control
#

png_processing_directory = "./temp_files_namprec/"

gif_file_name = "./graphics_files/NAM_Prec.gif"

png_file_root = png_processing_directory + "NAM_Prec_"

os.system("rm -v "+ png_processing_directory +"*")








#
####################################################
####################################################
####################################################


# In[3]:


###################################################
#
# NWS Rainfall Color Table.
#

nws_precip_colors = [
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6",  # 8.00 - 10.00 inches
    "#fdfdfd"]  # 10.00+

precip_colormap = mpl.colors.ListedColormap(colors = nws_precip_colors)

precip_levels_in = [   0.01,   0.10,  0.25,   0.50, 
                       0.75,   1.00,  1.50,   2.00, 
                       2.50,   3.00,  4.00,   5.00,
                       6.00,   8.00, 10.00,  20.00] # in Inches!!!

precip_levels_mm = [  0.25,   2.50,   5.00,  10.00, 
                     20.00,  25.00,  40.00,  50.00, 
                     60.00,  75.00, 100.00, 125.00,
                    150.00, 200.00, 250.00, 500.00] # in mm

#
###################################################


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

# In[4]:


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

# In[5]:


####################################################
####################################################
####################################################
#
# Crack open the UCAR thredds NOMADS array.
#

nam_model = xr.open_dataset(nam_opendap_url)

nam_model = nam_model.metpy.parse_cf()



data_crs = nam_model.metpy_crs.metpy.cartopy_crs





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

# In[6]:


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

# Precip & 700-hPa VVel



vvel_700   = nam_model["Vertical_velocity_pressure_isobaric"][:,k_0700hpa_height,:,:]

precip = nam_model["Total_precipitation_surface_3_Hour_Accumulation"]


precip.values         = precip.values / 25.4
precip.attrs['units'] = 'in'



    
#
####################################################
####################################################
####################################################


# In[7]:


####################################################
####################################################
####################################################
#
# Plot Sample Map for Records
#

start_time = nam_model["reftime"].values
time_dim   = precip.dims[0]
times_utc  = precip.coords[time_dim].to_numpy()
fxx        = (times_utc-start_time)/ np.timedelta64(1, 'h')


for i in range(len(times_utc)) :

    tz='America/Denver'
    time_utc   = times_utc[i]
    valid_time = pd.to_datetime(start_time).tz_localize(tz="UTC").strftime("%Y-%m-%d %H00 %Z")
    local_time = pd.to_datetime(times_utc[i]).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H00 %Z")

    plot_label = "NAM 3-hrly Precipitation"
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


    
    rain_norm = mpl.colors.BoundaryNorm(boundaries = precip_levels_in, 
                                                ncolors    = 15)

 
    
    
    contourf_levels = precip_levels_in

    contourf_plot = precip[i,:,:].plot.contourf(cmap      = precip_colormap,
                                                extend   = 'max',
                                                norm      = rain_norm,
                                                levels    = contourf_levels,
                                                cbar_kwargs = {"label"       : "3-Hourly Precip (in)",
                                                               "orientation" : "horizontal",
                                                               "pad"         :0.01,
                                                               "shrink"      :0.75,
                                                               "aspect"      :30})    


    contour_plot2 = precip[i,:,:].plot.contour(colors     =        "cyan",
                                               linewidths =           1, 
                                               levels     = np.array([0.002]))


    
    ax.set_title(time_label,
                        fontsize=15, color="black")


    
    plt.tight_layout()

    plt.savefig(png_file_root+str(i).zfill(2)+".png")
    
    plt.close()



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
          png_file_root + "*.png"  + 
          " " + 
          gif_file_name)


#
##################################################


# In[ ]:




