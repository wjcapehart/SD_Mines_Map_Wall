#!/usr/bin/env python
# coding: utf-8

# ![Masthead AES](https://kyrill.ias.sdsmt.edu/wjc/eduresources/AES_Masthead.png)
# 
# # Plotting Fronts
# This uses MetPy to decode text surface analysis bulletins from the Weather Prediction Center and then provides precipitation, mean sea level pressure, and lightning from other products.
# 
# ![Surface Analysis Product](https://kyrill.ias.sdsmt.edu/Map_Wall/graphics_files/NWS_Sfc_Analysis.png)
# 
# The data sources are from the Unidata THREDDS Service, as follows:
# 
# * Fronts, Highs, Lows & Troughs: [NOAAPort KWBC Front Text Bulletin ASUS02 (Coded Surface Frontal Positions)](https://www.wpc.ncep.noaa.gov/html/read_coded_bull_hr.shtml). Typically Available ~1.5 hr from valid time.
# * Mean Sea Level Reduced Pressure: [20-km Rapid Refresh Forecast/Analysis](https://rapidrefresh.noaa.gov/#:~:text=The%20Rapid%20Refresh%20is%20the,system%20to%20initialize%20that%20model.). F00 is available ~1.5-2.0 hr from valid time; otherwise, uses previous F01 forecast
# * Rain/Snow/Ice/Freezing Rain: [20-km Rapid Refresh Forecast/Analysis](https://vlab.noaa.gov/web/mdl/nbm). F00 is available ~1 hr from valid time; otherwise, uses previous F01 forecast
# * Lightening: [NOAA-NSSL Multi-Radar/Multi-Sensor System](https://www.nssl.noaa.gov/projects/mrms/).  Available 10-15 minutes from valid time.  Here we display the last 30-minutes of lightning strikes

# In[ ]:





# ## Libraries

# In[ ]:


import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.colors as mcolors
import cfgrib            as cfgrib
import cf_xarray         as cfgrib
import netCDF4           as nc



import timezonefinder    as tzf
import pytz              as pytz
import urllib    as urllib
import shutil
import metpy as metpy
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature

import os                as     os
import pygrib            as pygrib

from statsmodels.distributions.empirical_distribution import ECDF

from datetime            import timezone

import metpy             as metpy

from metpy.cbook import get_test_data
from metpy.io import parse_wpc_surface_bulletin
from metpy.plots import (add_metpy_logo, ColdFront, OccludedFront, StationaryFront,
                         StationPlot, WarmFront)


import siphon.catalog     as siphcat  
import siphon.ncss        as siphncss

from   siphon.cdmr        import Dataset
from   siphon.catalog     import TDSCatalog


working_dir = "./temp_sfc_analysis/"

os.system("rm -frv "+working_dir+"/*")

tz='America/Denver'


#
####################################################



# ## Malicious Compliance with SD Mines Communication Policy

# In[ ]:


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


# ---
# ## Clock Graphic

# In[ ]:


def plot_clock_stationary(fig,local_time):
    #####################################################
#

    axins = fig.add_axes(rect     =    [0,
                                        1-0.12, #0.015,
                                        0.12*8/9,
                                        0.12],
                          projection  =  "polar")

    time_for_clock = pd.to_datetime(local_time).time()

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


# ---
# ## Unidata Program for Plotting and Processing the Frontal Analysis Bulletin

# In[ ]:


def plot_bulletin(ax, data):
    """Plot a dataframe of surface features on a map."""
    # Set some default visual styling
    size = 9
    fontsize = 10
    HLfontsize = 30
    spacing = 2


    complete_style = { 'HIGH': {'color': 'blue', 'fontsize': HLfontsize},
                        'LOW': {'color': 'darkred', 'fontsize': HLfontsize},
                       'WARM': {'linewidth': 1, 'path_effects': [WarmFront(size=size, spacing=spacing)]},
                       'COLD': {'linewidth': 1, 'path_effects': [ColdFront(size=size, spacing=spacing)]},
                      'OCFNT': {'linewidth': 1, 'path_effects': [OccludedFront(size=size, spacing=spacing)]},
                      'STNRY': {'linewidth': 1, 'path_effects': [StationaryFront(size=size, spacing=spacing)]},
                       'TROF': {'linewidth': 2, 'linestyle': 'dashed',
                                'edgecolor': 'brown'}}

    complete_stylet = {'HIGH': {'color': 'blue', 'fontsize': fontsize},
                      'LOW': {'color': 'darkred', 'fontsize': fontsize},
                      'WARM': {'linewidth': 1, 'path_effects': [WarmFront(size=size, spacing=spacing)]},
                      'COLD': {'linewidth': 1, 'path_effects': [ColdFront(size=size, spacing=1)]},
                      'OCFNT': {'linewidth': 1, 'path_effects': [OccludedFront(size=size, spacing=1)]},
                      'STNRY': {'linewidth': 1, 'path_effects': [StationaryFront(size=size, spacing=1)]},
                      'TROF': {'linewidth': 2, 'linestyle': 'dashed',
                               'edgecolor': 'brown'}}


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





# ---
# ## Read and pre-process Frontal Analysis Product

# In[ ]:


front_file_root   = "Fronts_highres_KWBC_"
fronts_collection = 'https://thredds.ucar.edu/thredds/catalog/noaaport/text/fronts/catalog.xml'

cat = TDSCatalog(fronts_collection)

result_loop = []
for item in cat.datasets:
    if front_file_root in item:
        result_loop.append(item)

most_recent_front_file = sorted(result_loop)[-1]
front_time_string      = most_recent_front_file[20:33]

url_fronts      = "https://thredds.ucar.edu/thredds/fileServer/noaaport/text/fronts/" + \
                   most_recent_front_file

temp_front_file = "./temp_sfc_analysis/fronts.txt"


print("downloading "+ url_fronts)
print("         to "+ temp_front_file)


urllib.request.urlretrieve(url_fronts, temp_front_file)

fronts_df = parse_wpc_surface_bulletin(temp_front_file)



time_fronts = fronts_df["valid"].iloc[0].replace(tzinfo=pytz.utc).to_pydatetime()


local_time  = time_fronts.astimezone(pytz.timezone(tz))


# ## Reading Rapid Refresh MSLP.

# In[ ]:


try:
   time_hrrr_mslp   = time_fronts
   hrrr_time_string = time_hrrr_mslp.strftime("%Y%m%d_%H%M")
   hrrr_url         = "https://thredds.ucar.edu/thredds/dodsC/" +  \
                      "grib/NCEP/RAP/CONUS_20km/"               +  \
                      "RR_CONUS_20km_"                          + \
                      hrrr_time_string + ".grib2"
   hrrr             = xr.open_dataset(filename_or_obj  = hrrr_url,
                                      decode_timedelta =     False)
   mslp_time        = 0
except:
   time_hrrr_mslp   = time_fronts - dt.timedelta(hours=1)
   hrrr_time_string = time_hrrr_mslp.strftime("%Y%m%d_%H%M")
   hrrr_url         = "https://thredds.ucar.edu/thredds/dodsC/" +  \
                      "grib/NCEP/RAP/CONUS_20km/"               +  \
                      "RR_CONUS_20km_"                          + \
                      hrrr_time_string + ".grib2"
   hrrr             = xr.open_dataset(filename_or_obj  = hrrr_url,
                                      decode_timedelta =     False)
   mslp_time        = 1


mslp_name = "MSLP_MAPS_System_Reduction_msl"


print("cracking HRRR grib file "+hrrr_url)
print("              time HRRR ",time_hrrr_mslp)
print("             timeoffset ",mslp_time)

hrrr      = xr.open_dataset(filename_or_obj=hrrr_url,
                            decode_timedelta = False)
hrrr      = hrrr.metpy.parse_cf()
hrrr_crs  = hrrr.metpy_crs.metpy.cartopy_crs

mslp =  hrrr[mslp_name][mslp_time,:,:]
mslp.values = mslp.values/100.
mslp.attrs["units"] = "hPa"


# --- 
# ## Read Alpha Field for National Blend

# In[ ]:


alpha_xf = xr.open_dataset(filename_or_obj = "./National_Blend_PoP_Alpha.nc",
                           engine          = "netcdf4",
                           decode_cf       = True)


alpha_crs = alpha_xf["lambert_conformal_conic"]
alpha2d = alpha_xf["National_Grid_Alpha"]

alpha2d.values = alpha2d.values * 0.8

#display(alpha2d)

#alpha2d.plot()


# ## Reading and Aggregating Probabilistic Preciputation from the National Blend
# 
# Method blends Rain, Freezing Rain, Snow and Ice Precipitation Categories by selecting the maximum likly probability of precipitation type for a given grid location.

# In[ ]:


time_nbm_prec   = time_fronts - dt.timedelta(hours=1)
nbm_time_string = time_nbm_prec.strftime("%Y%m%d_%H%M")

dap_nbm_prec    = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NBM/CONUS/" + \
                  "National_Blend_CONUS_" + nbm_time_string + ".grib2"

url_nbm_prec    = "https://thredds.ucar.edu/thredds/fileServer/grib/NCEP/NBM/CONUS/" + \
                  "National_Blend_CONUS_" + nbm_time_string + ".grib2"

temp_nbm_file   = "./temp_sfc_analysis/National_Blend_CONUS.grib2"


nbmf      = xr.open_dataset(filename_or_obj  = dap_nbm_prec,
                            decode_timedelta = False)

nbmf      = nbmf.metpy.parse_cf()
nbmf_crs  = nbmf.metpy_crs.metpy.cartopy_crs

prob_rain = nbmf["Precipitation_type_surface_probability_between_1p0_and_2"][0,:,:]
prob_fzrn = nbmf["Precipitation_type_surface_probability_between_3p0_and_4"][0,:,:]
prob_snow = nbmf["Precipitation_type_surface_probability_between_5p0_and_7"][0,:,:]
prob_icep = nbmf["Precipitation_type_surface_probability_between_8p0_and_9"][0,:,:]

ny = prob_rain.shape[0]
nx = prob_rain.shape[1] 




pop_3d = prob_rain.copy()
pcp_type = xr.DataArray(data = ["rain","snow","fzrn","ice"],
                        coords={"pcp_type":["rain","snow","fzrn","ice"]},
                        attrs= {"long name":"Precipitation Type"})

pop_3d = xr.DataArray(name="Probability of Categorical Precip",
                      dims={"pcp_type":pcp_type.size,
                            "y":prob_snow.y.size,
                            "x":prob_snow.x.size},
                      coords={"pcp_type":pcp_type,
                            "y":prob_snow.y,
                            "x":prob_snow.x,
                             "metpy_crs":prob_snow.metpy_crs},
                      attrs={"long_name":"Probability of Categorical Precip",
                             "units":"%"})


pop_3d[0,:,:] = prob_rain
pop_3d[1,:,:] = prob_snow
pop_3d[2,:,:] = prob_fzrn
pop_3d[3,:,:] = prob_icep

#####################################
#
# Create Final Precipitation Working Images
#

# Step 1: Find the maximum value along the pcp_type dimension for each (y, x) pair
max_pcp = pop_3d.max(dim='pcp_type')  # This gives a 2D array with max values along pcp_type for each (y, x)

# Step 2: Create a mask where the value in pop_3d equals the max value for each (y, x)
mask = pop_3d == max_pcp

# Step 3: Apply the mask to set non-max values to np.nan
pop_3d = pop_3d.where(mask)

#
#####################################


pop_3d = pop_3d.where(pop_3d >= 1., np.nan)



# ---

# In[ ]:


## Pull past 30-minute NLDN Fields from MRMS Products


# In[ ]:


nldn_collection = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/MRMS/CONUS/Lightning/MRMS_CONUS_Lightning_'+ front_time_string +'.grib2/catalog.xml'

cat             = TDSCatalog(nldn_collection)

nldn_xf         = cat.datasets[0].remote_access(service="CdmRemote", 
                                                use_xarray=True)
print("Accessing NLDN for " + front_time_string)
nldn_30m = nldn_xf["NLDN_CG_030min_AvgDensity_altitude_above_msl"][0,0,:,:]
nldn_lon = nldn_xf["lon"]
nldn_lat = nldn_xf["lat"]




indices = np.argwhere(nldn_30m.values>0)
n_nldn  = indices.shape[0]

nldn = pd.DataFrame(columns=["lon","lat","dens"])
if (n_nldn > 0):
    nldn["lon"]  = nldn_lon[indices[:,1]]
    nldn["lat"]  = nldn_lat[indices[:,0]]
    nldn["dens"] = nldn_30m.values[indices[:,0],indices[:,1]]
    print(nldn)
else:
    print("No Lightning")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###############################
#
# Map Product Projection Information
#


class LowerAlbersEqualArea(ccrs.AlbersEqualArea):
    @property
    def threshold(self):
        return 1e3

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

#
###############################


###############################
#
# Time for Labels and Clock
#

valid_times = time_fronts.strftime("%Y-%m-%d %H%M %Z")
local_times = local_time.strftime("%Y-%m-%d %H%M %Z")
hrrr_times  = time_hrrr_mslp.strftime("%Y-%m-%d %H%M %Z")
nbm_times   = time_nbm_prec.strftime("%Y-%m-%d %H%M %Z")

#
###############################

###############################
#
# Drop Initial Plotting and Mapping
#

fig = plt.figure(figsize   = [Lx, Ly], 
                 facecolor = 'white')

ax = fig.add_subplot(1, 1, 1, 
                     projection=myproj)

plt.subplots_adjust(left   = 0.01, 
                    right  = 0.99, 
                    top    = 0.99, 
                    bottom = 0, 
                    wspace = 0)


ax.set_frame_on(False)
ax.set_extent(bbox, crs=ccrs.PlateCarree())

#
###############################


###############################
#
# Blended Analysis Precip Product
#

pop_3d[0,:,:].plot.pcolormesh(ax=ax, 
                      transform=nbmf_crs, 
                      cmap='Greens', 
                      add_colorbar=False,
                      alpha = alpha2d,
                      vmin=0, vmax=100)

pop_3d[1,:,:].plot.pcolormesh(ax=ax, 
                      transform=nbmf_crs, 
                      cmap='Blues', 
                      add_colorbar=False,
                      alpha = alpha2d,
                      vmin=0, vmax=100)

pop_3d[2,:,:].plot.pcolormesh(ax=ax, 
                      transform=nbmf_crs, 
                      cmap='Reds', 
                      add_colorbar=False,
                      alpha = alpha2d,
                      vmin=0, vmax=100)
pop_3d[3,:,:].plot.pcolormesh(ax=ax, 
                      transform=nbmf_crs, 
                      cmap='Purples', 
                      add_colorbar=False,
                      alpha = alpha2d,
                      vmin=0, vmax=100)

#
###############################

###############################
#
# NLDN Lightning Data
#

if (n_nldn > 0):
    plt.scatter(nldn["lon"],
                nldn["lat"],
                transform  = ccrs.PlateCarree(),
                color      = "gold",
                s          = 0.75)
#
###############################

###############################
#
# Geographic Features
# 

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
               linewidth = 0.50, 
               edgecolor = Mines_Blue)

ax.add_feature(cfeature.STATES.with_scale('50m'),    
               linewidth = 0.25, 
               edgecolor = Mines_Blue)

ax.add_feature(cfeature.LAKES.with_scale('50m'),     
               linewidth = 0.50, 
               facecolor = "none", 
               edgecolor = Mines_Blue)
#
###############################



###############################
#
# HRRR MSLP
#

clevels = np.arange(900,1200,4)
mslpplot = mslp.plot.contour(ax     = ax,
                             colors  = "#001633",
                             levels = clevels,
                             transform = hrrr_crs)
ax.clabel(mslpplot, 
          levels = mslpplot.levels, 
          colors = Mines_Blue,
          inline=True, 
          fontsize=10)
#
###############################

###############################
#
# Fonts Fields
#

plot_bulletin(ax, fronts_df)

#
###############################

###############################
#
#  Labels and Annotations
#

plt.suptitle("NWS-WPC Surface Analysis",
             fontsize = 30, 
             color    = Mines_Blue)

plot_clock_stationary(fig, local_time)
ax.set_title(valid_times + "  (" + local_times+")",
             fontsize=20, 
             color=Mines_Blue)

hrrr_label = "Rapid Refresh MSLP Using "+ hrrr_times+" Fx"+str(mslp_time).zfill(2)
natbl_label = "National Blend PoPs Using "+ nbm_times+" Fx"+str(1).zfill(2)
nldn_label = "30-min NLDN Strikes Using "+ valid_times +" Fx"+str(0).zfill(2)

xbot =  2983074
ybot = -1604583

ax.text(xbot, ybot+100000,
        hrrr_label, 
        fontsize="x-small", 
        horizontalalignment = "right")

ax.text(xbot, ybot+50000,
        natbl_label, 
        fontsize="x-small", 
        horizontalalignment = "right")

ax.text(xbot, ybot,
        nldn_label, 
        fontsize="x-small", 
        horizontalalignment = "right")

#
###############################


#########################################
#
# Save to File
#

plt.savefig("./temp_sfc_analysis/NWS_Sfc_Analysis.png",
                        facecolor   = 'white', 
                        transparent =   False)

print([ax.get_xbound(), ax.get_ybound()])
print([ax.get_xlim(), ax.get_ylim()])


#plt.show()

plt.close()
os.system("mv -fv ./temp_sfc_analysis/NWS_Sfc_Analysis.png ./graphics_files/")

#
#########################################


# ## Closeout

# In[ ]:


#########################################
#
# And we're done
#

print("And We're Done!")

#
#########################################


# In[ ]:




