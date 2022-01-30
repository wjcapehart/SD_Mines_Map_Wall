#!/usr/bin/env python
# coding: utf-8

# # Thermal Infrared

# In[1]:


from datetime import datetime
from siphon.catalog import TDSCatalog
from datetime import datetime
import metpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import colortables
from metpy.plots import add_timestamp
import numpy             as np
import os                as os
import pandas             as pd


# In[2]:


# %load solutions/data_url.py
total_frames = 45*2
os.system("rm -v ./sat_ir_temp_files/*")


# Cell content replaced by load magic replacement.

# Create variables for URL generation
image_date = datetime.utcnow().date()
region = 'CONUS'
channel = 13

# We want to match something like:
# https://thredds-test.unidata.ucar.edu/thredds/catalog/satellite/goes16/GOES16/Mesoscale-1/Channel08/20181113/catalog.html

# Construct the data_url string
data_url = ('https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/'
            f'CloudAndMoistureImagery/{region}/Channel{channel:02d}/current/catalog.xml')

# Print out your URL and verify it works!
print(data_url)


# In[3]:


cat = TDSCatalog(data_url)


# In[4]:


cat.datasets[1:total_frames]


# In[5]:


len(cat.datasets[1:total_frames])


# In[6]:


for i in range(1,len(cat.datasets[1:total_frames])+1,2) : 

    dataset = cat.datasets[i]
    ds = dataset.remote_access(use_xarray=True)
    dat = ds.metpy.parse_cf('Sectorized_CMI')
    proj = dat.metpy.cartopy_crs
    x = dat['x']
    y = dat['y']


    tz='America/Denver'
    time_utc = datetime.strptime(ds.start_date_time, '%Y%j%H%M%S')
    valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H:%M:%S %Z")
    local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    file_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d_%H%M")


    print(valid_time,local_time)



    fig = plt.figure(figsize=(13, 8),
                           facecolor = 'white')

    plt.suptitle("GOES 16 Band 13 [10.3 µm Clean Longwarve IR Window]",
                    fontsize=20)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_title(valid_time + "  (" + local_time+")",
                    fontsize=15)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='black')

    im = ax.imshow(dat, extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')

    wv_cmap = colortables.get_colortable('WVCIMSS_r')
    im.set_cmap(wv_cmap)
    plt.tight_layout()
    plt.savefig("./sat_ir_temp_files/Sat_IR_Loop_Image_"+file_time+".png")
    plt.close()
    


# In[7]:


##################################################
#
# Convert PNGs into an Animated GIF
#

os.system("convert -delay 15 " + 
          "./sat_ir_temp_files/Sat_IR_Loop_Image_*.png"  + 
          " " + 
          "./graphics_files/RealTime_SAT_IR_Loop.gif")


#
##################################################


# In[ ]:




