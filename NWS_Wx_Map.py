#!/usr/bin/env python
# coding: utf-8

# # Add Clock to NWS Map.
# 
# ## Libraries

# In[ ]:


####################################################
####################################################
####################################################
#
# Libraries
#

from   mpl_toolkits.axes_grid1.inset_locator import inset_axes

import urllib.request
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image
import numpy as np
from datetime import datetime, timedelta

import pytz as pytz
import os as os

import time

import timezonefinder as tzf

import pandas            as pd

# https://www.wpc.ncep.noaa.gov/basicwx/91fndfd.gif
# https://www.wpc.ncep.noaa.gov/basicwx/6_hour.txt

#
####################################################
####################################################
####################################################


# ## Mines Colors and Fonts

# In[ ]:


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


# ## Time Zone Handling & Current Time

# In[ ]:


####################################################
####################################################
####################################################
#
# Time Zone Handling & Current Time
#

tf     = tzf.TimezoneFinder()
tz     = tf.certain_timezone_at(lng = -103, 
                                lat =   44)

mtz = time.tzname[time.daylight]
print("Machine Time Zone:",mtz)
print("  Local Time Zone:",tz)
#
####################################################
####################################################
####################################################


# ## Get Current Time

# In[ ]:


####################################################
####################################################
####################################################
#
# Get Current Time
#

if(mtz != "UTC"):
    now_date = pd.to_datetime(datetime.now()).tz_localize(tz=tz).tz_convert(tz="UTC")
else:
    now_date = pd.to_datetime(datetime.now()).tz_localize(tz="UTC").tz_convert(tz="UTC")

#
####################################################
####################################################
####################################################


# ## Downloading Image and Timing Files from NCEP

# In[ ]:


####################################################
####################################################
####################################################
#
# Downloading Image and Timing Files from NCEP
#

os.system("rm -frv ./temp_sfc_analysis/*")

url_map_overlay  = "https://www.wpc.ncep.noaa.gov/basicwx/91fndfd.gif"
file_map_overlay = "./temp_sfc_analysis/91fndfd.gif"


url_map_data     = "https://www.wpc.ncep.noaa.gov/basicwx/6_hour.txt"
file_map_data    = "./temp_sfc_analysis/6_hour.txt"

print("downloading "+ url_map_overlay)
urllib.request.urlcleanup()
with urllib.request.urlopen(url_map_overlay) as response, open(file_map_overlay, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

print("downloading "+ url_map_data)
urllib.request.urlcleanup()
with urllib.request.urlopen(url_map_data) as response, open(file_map_data, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

#
####################################################
####################################################
####################################################


# ## Read Map Metadata File Internal Date Data

# In[ ]:


####################################################
####################################################
####################################################
#
# Read Map Metadata File Internal Date Data
#

print(tz)

with open(file_map_data) as f:
    lines         = f.readlines()

date_string = lines[4][7:28]

print(date_string)

product_time  = pd.to_datetime(datetime.strptime(date_string, '%H00Z %a %b %d %Y')).tz_localize(tz="UTC")

#
####################################################
####################################################
####################################################


# ## Get Server Metadata File Creation Time

# In[ ]:


####################################################
####################################################
####################################################
#
# Get Server Metadata File Creation Time
#

urllib.request.urlcleanup()
metadata_file_date = urllib.request.urlopen(url_map_data, timeout=30)
metadata_file_date = metadata_file_date.headers['last-modified']
metadata_file_date = pd.to_datetime(datetime.strptime(metadata_file_date, "%a, %d %b %Y %H:%M:%S %Z")).tz_localize(tz="GMT").tz_convert(tz="UTC")

#
####################################################
####################################################
####################################################


# ## Get Server Map Image File Creation Time

# In[ ]:


####################################################
####################################################
####################################################
#
# Get Server Map Image File Creation Time
#

urllib.request.urlcleanup()
map_file_date      = urllib.request.urlopen(url_map_overlay, timeout=30)
map_file_date      = map_file_date.headers['last-modified']
map_file_date      = pd.to_datetime(datetime.strptime(map_file_date, "%a, %d %b %Y %H:%M:%S %Z")).tz_localize(tz="GMT").tz_convert(tz="UTC")

#
####################################################
####################################################
####################################################


# ## Summary of Times

# In[ ]:


####################################################
####################################################
####################################################
#
# Summary of Times
#
print("       Current Time: ",           now_date)
print("Map Image File Time: ",      map_file_date)
print(" Metadata File Time: ", metadata_file_date)
print(" Product Label Time: ",       product_time)

#
####################################################
####################################################
####################################################


# In[ ]:


####################################################
####################################################
####################################################
#
# Append to Timings File
#

timings_file = "./graphics_files/NWS_Map_Timings_File.csv"

print("       Current Time: ",           now_date)
print("Map Image File Time: ",      map_file_date)
print(" Metadata File Time: ", metadata_file_date)
print(" Product Label Time: ",       product_time)

now_string      = now_date.strftime("%Y-%m-%d %H:%M:%S")
mapfile_string  = map_file_date.strftime("%Y-%m-%d %H:%M:%S")
metadata_string = metadata_file_date.strftime("%Y-%m-%d %H:%M:%S")
product_string  = product_time.strftime("%Y-%m-%d %H:%M:%S")

record_string = now_string      + ", "+ \
                mapfile_string  + ", "+ \
                metadata_string + ", "+ \
                product_string

print(record_string)

with open(timings_file, 'a') as f:
    print(record_string, file = f)

#
####################################################
####################################################
####################################################


# In[ ]:





# ## Graphical Map Clock Information

# In[ ]:


####################################################
####################################################
####################################################
#
# Graphical Map Clock Information
#

local_time      = pd.to_datetime(product_time).tz_convert(tz=tz)
local_time_zone = local_time.strftime("%Z")
dow             = local_time.strftime("%a")
time_for_clock  = local_time.time()

hour   = time_for_clock.hour
minute = time_for_clock.minute
second = time_for_clock.second

if ((hour >= 6) and (hour < 18)):
    Clock_Color = Mines_Blue
    Clock_BgndC = "white"           
else:
    Clock_Color = "white"
    Clock_BgndC = Mines_Blue  

print(product_time)
print(local_time)
print(local_time_zone)
print(dow)
print(time_for_clock, " ", Clock_Color,Clock_BgndC )

time_label  = "NWS Surface Forecast " + \
              product_time.strftime("%Y-%m-%d %H00 %Z") +  \
             " (" + local_time.strftime("%Y-%m-%d %H00 %Z")    + ")"

print(time_label)

#
####################################################
####################################################
####################################################


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Generate Map Image

# In[ ]:


####################################################
####################################################
####################################################
#
# Graphical Map Clock Information
#

#########################################
#
# Read Map Image
#

urllib.request.urlcleanup()
img = Image.open(fp      = file_map_overlay,
                 formats =["gif"])

img_data = np.asarray(img.convert("RGB"))

nx, ny = img.size
dpi    = 72.

Lx = nx/dpi
Ly = ny/dpi

print("nx = ", nx, " ny = ", ny, " dpi = ", dpi)
print("Lx = ", Lx, " Ly=", Ly)

#
#########################################


#########################################
#
# Generate Map Axes and Lay Down NCEP Map
#

fig, ax = plt.subplots(figsize=(Lx, Ly), dpi = dpi)

plt.axis('off')

plt.subplots_adjust(left   = 0, 
                    right  = 1, 
                    top    = 1, 
                    bottom = 0, 
                    wspace = 0,
                    hspace = 0)

plt.margins(x = 0, y = 0)

ax.imshow(img_data)

#
#########################################


#########################################
#
# Insert a The Local Time Clock
#

axins = fig.add_axes(rect     =    [0.00,
                                    1-0.16,
                                    0.12,
                                    0.12],
                      projection  =  "polar")

circle_theta  = np.deg2rad(np.arange(0,360,0.01))
circle_radius = circle_theta * 0 + 1

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
axins.margins(x = 0, y = 0)

axins.plot([angles_h,angles_h], [0,0.60], color=Clock_Color, linewidth=2)
axins.plot([angles_m,angles_m], [0,0.95], color=Clock_Color, linewidth=2)
axins.plot(circle_theta, circle_radius, color="darkgrey", linewidth=2)

#
#########################################


#########################################
#
# Add Time Label Information
#

ax.add_patch(Rectangle(xy=(0, 0), 
                       width = nx, 
                       height = 20,
                       edgecolor = 'white',
                       facecolor = 'white',
                       fill=True,
                       lw=1))



ax.annotate(time_label, 
             [0.5,1], 
             xycoords              = "axes fraction", 
             fontsize              =            28, 
             verticalalignment     =           "top",
             horizontalalignment   =        "center",
             backgroundcolor       =         "white",
             zorder                =           99999,
             bbox = dict(facecolor ='white',edgecolor ="white"))

#
#########################################


#########################################
#
# Save to File
#

plt.savefig("./temp_sfc_analysis/NWS_Sfc_Analysis.png",
                        facecolor   = 'white', 
                        transparent =   False)


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

