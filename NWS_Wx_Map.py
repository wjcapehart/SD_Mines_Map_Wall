#!/usr/bin/env python
# coding: utf-8

# # Add Clock to NWS Map.
# 
# ## Libraries

# In[1]:


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


import timezonefinder as tzf

import pandas            as pd



# https://www.wpc.ncep.noaa.gov/basicwx/ndfd_overlay_loop.txt
# https://www.wpc.ncep.noaa.gov/basicwx/91fndfd.gif


# ## Colors

# In[2]:


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


# ## Pull Data

# In[3]:


os.system("rm -frv ./temp_sfc_analysis/*")

url_map_overlay  = "https://www.wpc.ncep.noaa.gov/basicwx/91fndfd.gif"
file_map_overlay = "./temp_sfc_analysis/91fndfd.gif"


url_map_data     = "https://www.wpc.ncep.noaa.gov/basicwx/6_hour.txt"
file_map_data    = "./temp_sfc_analysis/6_hour.txt"

print("downloading "+ url_map_overlay)
with urllib.request.urlopen(url_map_overlay) as response, open(file_map_overlay, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

print("downloading "+ url_map_data)
with urllib.request.urlopen(url_map_data) as response, open(file_map_data, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)


# ## Read Map Image Data

# In[4]:


img = Image.open(fp      = file_map_overlay,
                 formats =["gif"])

img_data = np.asarray(img.convert("RGB"))


# In[5]:


nx, ny = img.size
dpi    = 72.

Lx = nx/dpi
Ly = ny/dpi

print("nx = ", nx, " ny = ", ny, " dpi = ", dpi)
print("Lx = ", Lx, " Ly=", Ly)


# ## Read Map Timing Data

# In[6]:


with open(file_map_data) as f:
    lines = f.readlines()

date_string = lines[4][7:28]

print(date_string)

tf     = tzf.TimezoneFinder()
tz     = tf.certain_timezone_at(lng = -103, 
                                lat =   44)

print(tz)
utc_time = pd.to_datetime(datetime.strptime(date_string, '%H00Z THU %b %d %Y'))

local_time      = pd.to_datetime(utc_time).tz_localize(tz="UTC").tz_convert(tz=tz)
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

print(utc_time)
print(local_time)
print(local_time_zone)
print(dow)
print(time_for_clock, " ", Clock_Color,Clock_BgndC )


time_label  = "NWS Surface Forecast " + \
              utc_time.tz_localize(tz="UTC").strftime("%Y-%m-%d %H00 %Z") +  \
             " (" + local_time.strftime("%Y-%m-%d %H00 %Z")    + ")"

print(time_label)


# ## Generate Image

# In[7]:


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


#########################################
#
# Insert a Clock
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


plt.savefig("./temp_sfc_analysis/NWS_Sfc_Analysis.png",
                        facecolor   = 'white', 
                        transparent =   False)
plt.close()
os.system("mv -fv ./temp_sfc_analysis/NWS_Sfc_Analysis.png ./graphics_files/")


# In[ ]:





# In[ ]:





# In[ ]:




