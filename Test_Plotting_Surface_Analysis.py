#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Plotting Fronts
# 
# This uses MetPy to decode text surface analysis bulletins from the Weather Prediction Center.
# The features in this bulletin are then plotted on a map, making use of MetPy's various
# path effects for matplotlib than can be used to represent a line as a traditional front.
# 

# In[ ]:


import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.colors as mcolors

import timezonefinder    as tzf
import pytz              as pytz
import urllib    as urllib
import shutil
import metpy as metpy
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature

import os                as     os
import pygrib            as pygrib

from metpy.cbook import get_test_data
from metpy.io import parse_wpc_surface_bulletin
from metpy.plots import (add_metpy_logo, ColdFront, OccludedFront, StationaryFront,
                         StationPlot, WarmFront)

working_dir = "./temp_sfc_analysis/"

os.system("rm -frv "+working_dir+"/*")

tz='America/Denver'



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



# Get Wweather LUT
# 
# 

# In[ ]:


#display(time_frame)
print(pd.Timestamp.now().to_pydatetime())
print(pd.Timestamp.now().to_pydatetime()  )
print(pd.Timestamp.now().round('3h').to_pydatetime())


# lag_hours = 1
# lag_minutes = 60
# 
# 
# time_data = np.arange('2024-01-26T03', '2024-01-26T15',15, dtype='datetime64[m]')
# 
# time_data_3 = pd.to_datetime(  time_data - np.timedelta64(90,'m') - np.timedelta64(lag_minutes,'m' ) ).round('3h')
# 
# time_frame = pd.DataFrame(data={"Now":time_data})
# time_frame["Past_Nearest_3"] = time_data_3 
# time_frame["time_diff"]      = time_frame["Now"] - time_frame["Past_Nearest_3"]
# 
# time_frame

# In[ ]:





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

beta_on   = 0
lag_hours = 1
lag_minutes = 90

current_datetime = dt.datetime.utcnow()



map_datetime  =  pd.to_datetime( (current_datetime - dt.timedelta(minutes=90)) - dt.timedelta(minutes=lag_minutes)  ).round('3h')
ndfd_datetime =  map_datetime - dt.timedelta(hours=1)
    


#
# Burn Current Time to File in WRF Root Directory
#


product_string_YYYY_MM_DD_HH00UTC =  map_datetime.strftime("%Y-%m-%d %H00 UTC")
fronts_date_YYYYMMDD_HH00         =  map_datetime.strftime("%Y%m%d_%H00")
ndfd_date_YYYYMMDD_HH00           = ndfd_datetime.strftime("%Y%m%d_%H00")

print("Current Time ", current_datetime)
print("Product Time ", map_datetime)
print("  Label Time ", product_string_YYYY_MM_DD_HH00UTC)
print(" Fronts Time ", fronts_date_YYYYMMDD_HH00)
print("   NDFD Time ", ndfd_date_YYYYMMDD_HH00)


#
####################################################
####################################################
####################################################


# In[ ]:


fronts_date_YYYYMMDD_HH00


# Pull Fronts
# 
# https://thredds.ucar.edu/thredds/catalog/noaaport/text/fronts/catalog.html

# In[ ]:


fronts_url = "https://thredds.ucar.edu/thredds/fileServer/" +  \
             "noaaport/text/fronts/Fronts_highres_KWBC_"    +  \
             fronts_date_YYYYMMDD_HH00 + ".txt"

temp_front_file = "./temp_sfc_analysis/fronts.txt"


print("downloading "+ fronts_url)
print("         to "+ temp_front_file)


urllib.request.urlretrieve(fronts_url, temp_front_file)

df = parse_wpc_surface_bulletin(temp_front_file)



# Gridded Analyses
# 
# https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NDFD/NWS/CONUS/CONDUIT/catalog.html
# 
# ```
# 374:267673420:vt=2024013021:surface:70 hour fcst:WX Weather information [WxInfo]:
#     ndata=2953665:undef=1479384:mean=1.29788:min=0:max=40
#     grid_template=30:winds(N/S):
# 	Lambert Conformal: (2145 x 1377) input WE|EW:SN output WE:SN res 0
# 	Lat1 20.191999 Lon1 238.445999 LoV 265.000000
# 	LatD 25.000000 Latin1 25.000000 Latin2 25.000000
# 	LatSP -90.000000 LonSP 0.000000
# 	North Pole (2145 x 1377) Dx 2539.703000 m Dy 2539.703000 m 
# ```

# ndfd_url  = "https://thredds.ucar.edu/thredds/dodsC/" +  \
#             "grib/NCEP/NDFD/NWS/CONUS/CONDUIT/"       +  \
#             "NDFD_NWS_CONUS_conduit_2p5km_"           + \
#             ndfd_date_YYYYMMDD_HH00 + ".grib2"
# 
# ndfd_http = "https://thredds.ucar.edu/thredds/fileServer/" + \
#             "grib/NCEP/NDFD/NWS/CONUS/CONDUIT/"       +  \
#             "NDFD_NWS_CONUS_conduit_2p5km_"           + \
#             ndfd_date_YYYYMMDD_HH00 + ".grib2"
# 
# print("cracking NDFD grib file "+ndfd_url)
# 
# ds_ndfd   = xr.open_dataset(filename_or_obj=ndfd_url)
# 
# ds_ndfd   = ds_ndfd.metpy.parse_cf()
# ndfd_crs  = ds_ndfd.metpy_crs.metpy.cartopy_crs
# 
# wx_code = ds_ndfd["Weather_string_surface"][0,:,:]   #.sel(time3=map_datetime)
# wx_code.values[wx_code.values==0]=[np.nan]
# print("Weather Time Stamp:",wx_code.coords)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# fig = plt.figure(figsize=[11,8])
# ax  = plt.subplot(1, 1, 1,
#                   projection = ndfd_crs)
# wx_code.plot(cmap = "nipy_spectral")
# ax.add_feature(cfeature.STATES.with_scale('50m'),    
#                linewidth = 0.25, 
#                edgecolor = "grey")
# plt.show()
# 
# 

# In[ ]:





# Best Forecast Fields

# url_ndfd = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NDFD/NWS/CONUS/NOAAPORT/NDFD_NWS_CONUS_2p5km_20240121_0000.grib2"
# 
# ds_ndfd2   = xr.open_dataset(filename_or_obj=url_ndfd)
# ds_ndfd2   = ds_ndfd2.metpy.parse_cf()
# 
# proj_ndfd = ds_ndfd2.metpy_crs.metpy.cartopy_crs
# 
# 
# fig = plt.figure(figsize   = [11, 8], 
#                  facecolor = 'white')
# 
# ax = fig.add_subplot(1, 1, 1, 
#                      projection=proj_ndfd)
# 
# ds_ndfd["Categorical_Rain_surface"][2,:,:].plot.imshow(ax=ax)
# 
# ax.add_feature(cfeature.STATES.with_scale('50m'),    
#                linewidth = 0.25, 
#                edgecolor = "white")
# 
# 
# plt.show()

# MSLP from HRRR
# 
# https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/RAP/CONUS_40km/RR_CONUS_40km_20240126_1200.grib2
# 
# https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/RAP/CONUS_40km/RR_CONUS_40km_20240126_1200.grib2
# 
# MSLP_MAPS_System_Reduction_msl

# In[ ]:


hrrr_url = "https://thredds.ucar.edu/thredds/dodsC/" +  \
           "grib/NCEP/RAP/CONUS_20km/"               +  \
           "RR_CONUS_20km_"                          + \
           fronts_date_YYYYMMDD_HH00 + ".grib2"

mslp_name = "MSLP_MAPS_System_Reduction_msl"


print("cracking HRRR grib file "+hrrr_url)

ds_hrrr       = xr.open_dataset(filename_or_obj=hrrr_url)
hrrr_time_dim = ds_hrrr[mslp_name].dims[0]


ds_hrrr   = ds_hrrr.metpy.parse_cf()
hrrr_crs  = ds_hrrr.metpy_crs.metpy.cartopy_crs

mslp        = ds_hrrr["MSLP_MAPS_System_Reduction_msl"][0,:,:]   #.sel(hrrr_time_dim=map_datetime)
mslp.values = mslp.values / 100





rain = ds_hrrr[         "Categorical_Rain_surface"][0,:,:] 
snow = ds_hrrr[         "Categorical_Snow_surface"][0,:,:] 
icep = ds_hrrr[  "Categorical_Ice_Pellets_surface"][0,:,:] 
frzr = ds_hrrr["Categorical_Freezing_Rain_surface"][0,:,:] 

water_equiv =  ds_hrrr["Precipitation_rate_surface"][0,:,:]

#water_equiv.values[water_equiv.values == 0] = [np.nan]
water_equiv.values = water_equiv.values * 0.0393701 * 3600.0

print("inches per hour", np.nanmax(water_equiv.values))
print("inches per hour", np.nanmin(water_equiv.values))

max_rain =  0.50
min_rain =  0.03
water_equiv.values[water_equiv.values < min_rain] = [0] 
water_equiv.values[water_equiv.values > max_rain] = [max_rain] 
water_equiv.values = (water_equiv.values-0)/(max_rain-0)

water_equiv.values[water_equiv.values < 0.25] = [0.25] 


print("alphas",np.nanmax(water_equiv.values))
print("alphas",np.nanmin(water_equiv.values))


rain.values[rain.values == 0] = [np.nan] 
snow.values[snow.values == 0] = [np.nan]
icep.values[icep.values == 0] = [np.nan]
frzr.values[frzr.values == 0] = [np.nan]



# In[ ]:





# In[ ]:


def plot_clock_stationary(fig,time_utc):
    #####################################################
#

    axins = fig.add_axes(rect     =    [0,
                                        1-0.12, #0.015,
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
    

    
    
    plt.setp(axins.get_yticklabels(), visible=False)
    plt.setp(axins.get_xticklabels(), visible=False)
    axins.spines['polar'].set_visible(False)
    axins.set_ylim(0,1)
    axins.set_theta_zero_location('N')
    axins.set_theta_direction(-1)
    axins.set_facecolor(Clock_BgndC)
    axins.grid(False)
    
    axins.plot([angles_h,angles_h], [0,0.60], color=Clock_Color, linewidth=1.5, zorder=99999)
    axins.plot([angles_m,angles_m], [0,0.95], color=Clock_Color, linewidth=1.5, zorder=99999)
    axins.plot(circle_theta, circle_radius,  color=Mines_Blue, linewidth=1, zorder=99999)
#
##################################################


# In[ ]:


def plot_bulletin(ax, data):
    """Plot a dataframe of surface features on a map."""
    # Set some default visual styling
    size = 9
    fontsize = 10
    HLfontsize = 30
    spacing = 2

    
    complete_style = { 'HIGH': {'color': 'blue', 'fontsize': HLfontsize},
                        'LOW': {'color': 'red', 'fontsize': HLfontsize},
                       'WARM': {'linewidth': 1, 'path_effects': [WarmFront(size=size, spacing=spacing)]},
                       'COLD': {'linewidth': 1, 'path_effects': [ColdFront(size=size, spacing=spacing)]},
                      'OCFNT': {'linewidth': 1, 'path_effects': [OccludedFront(size=size, spacing=spacing)]},
                      'STNRY': {'linewidth': 1, 'path_effects': [StationaryFront(size=size, spacing=spacing)]},
                       'TROF': {'linewidth': 2, 'linestyle': 'dashed',
                                'edgecolor': 'darkred'}}

    complete_stylet = {'HIGH': {'color': 'blue', 'fontsize': fontsize},
                      'LOW': {'color': 'red', 'fontsize': fontsize},
                      'WARM': {'linewidth': 1, 'path_effects': [WarmFront(size=size, spacing=spacing)]},
                      'COLD': {'linewidth': 1, 'path_effects': [ColdFront(size=size, spacing=1)]},
                      'OCFNT': {'linewidth': 1, 'path_effects': [OccludedFront(size=size, spacing=1)]},
                      'STNRY': {'linewidth': 1, 'path_effects': [StationaryFront(size=size, spacing=1)]},
                      'TROF': {'linewidth': 2, 'linestyle': 'dashed',
                               'edgecolor': 'darkred'}}


    # Handle H/L points using MetPy's StationPlot class
    for field in ('HIGH', 'LOW'):
        rows = data[data.feature == field]
        x, y = zip(*((pt.x, pt.y) for pt in rows.geometry))
        sp = StationPlot(ax, x, y, transform=ccrs.PlateCarree(), clip_on=True)
        sp.plot_text('C', [field[0]] * len(x), **complete_style[field])
        sp.plot_parameter('S2', rows.strength, **complete_stylet[field])

    
    # Handle all the boundary types
    for field in ('WARM', 'COLD', 'STNRY', 'OCFNT', 'TROF'):
        rows = data[data.feature == field]
        ax.add_geometries(rows.geometry, crs=ccrs.PlateCarree(), **complete_style[field],
                          facecolor='none')


# In[ ]:





# In[ ]:













# 
# 

# In[ ]:





# 

# In[ ]:


# Set up a default figure and map


myproj = ccrs.AlbersEqualArea(central_longitude  = -96.0, 
                              central_latitude   =  37.5, 
                              false_easting      =   0.0, 
                              false_northing     =   0.0, 
                              standard_parallels = (29.5, 45.5))

bbox = [-125,  -67,
          23,   53] 

bbox = [-125,  -67,
          23,   52] 

Lx = 14.152777777777779  
Ly =  9.88888888888889


time_utc = df["valid"][0].to_pydatetime()
valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H%M %Z")
local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H%M %Z")





fig = plt.figure(figsize   = [Lx, Ly], 
                 facecolor = 'white')

ax = fig.add_subplot(1, 1, 1, 
                     projection=myproj)

plt.subplots_adjust(left   = 0.01, 
                    right  = 0.99, 
                    top    = 0.99, 
                    bottom = 0, 
                    wspace = 0)

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
               linewidth = 0.5,
               edgecolor = Mines_Blue)
ax.add_feature(cfeature.STATES.with_scale('50m'),    
               linewidth = 0.25, 
               edgecolor = Mines_Blue)
ax.add_feature(cfeature.LAKES.with_scale('50m'),   
               linewidth = 0.5,
               facecolor = "none", 
               edgecolor = Mines_Blue)

ax.set_frame_on(False)

ax.set_extent(bbox, crs=ccrs.PlateCarree())


###############################
#
# Fonts Fields
#

plot_bulletin(ax, df)

#
###############################

###############################
#
# NDFD Weather
#
#
#wx_code.plot.imshow(ax=ax, 
#                    alpha = 0.95,
#                    add_colorbar = False,
#                   transform = ndfd_crs,
#                   cmap = "nipy_spectral")
#
#
###############################

###############################
#
# HRRR Weather
#
myalpha = water_equiv.values
#myalpha = 1
rain.plot.imshow(ax = ax, alpha = myalpha, cmap =  "Greens", add_colorbar = False, transform = hrrr_crs)
snow.plot.imshow(ax = ax, alpha = myalpha, cmap =   "Blues", add_colorbar = False, transform = hrrr_crs)
icep.plot.imshow(ax = ax, alpha = myalpha, cmap = "Purples", add_colorbar = False, transform = hrrr_crs)
frzr.plot.imshow(ax = ax, alpha = myalpha, cmap =    "Reds", add_colorbar = False, transform = hrrr_crs)

#
###############################


###############################
#
# HRRR Weather
#

clevels = np.arange(900,1200,4)
mslpplot = mslp.plot.contour(ax     = ax,
                             colors  = "#001633",
                             levels = clevels,
                             transform = hrrr_crs)
ax.clabel(mslpplot, 
          levels = mslpplot.levels, 
          colors = "#001633",
          inline=True, 
          fontsize=10)
#
###############################



plt.suptitle("NWS-WPC Surface Analysis",
             fontsize = 30, 
             color    = Mines_Blue)

plot_clock_stationary(fig, time_utc)
ax.set_title(valid_time + "  (" + local_time+")",
             fontsize=20, 
             color=Mines_Blue)

plt.savefig("./temp_sfc_analysis/NWS_Sfc_Analysis.png",
                        facecolor   = 'white', 
                        transparent =   False)


plt.close()
os.system("mv -fv ./temp_sfc_analysis/NWS_Sfc_Analysis.png ./graphics_files/")





# In[ ]:





# In[ ]:


print("Current Time ", current_datetime)
print("Product Time ", map_datetime)
print("  Label Time ", product_string_YYYY_MM_DD_HH00UTC)
print(" ")
print("- Fronts Time ", fronts_date_YYYYMMDD_HH00)
print(" ")
print("--- MSLP Time ", mslp.coords)
print(" ")
print("--- PREC Time ", water_equiv.coords)


# In[ ]:





# In[ ]:





# In[ ]:




