#!/usr/bin/env python
# coding: utf-8

# # Water Vapour

# In[30]:


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


# In[31]:


# %load solutions/data_url.py
total_frames = 45*2

processing_directory = "./sat_wv_temp_files/"



# Cell content replaced by load magic replacement.

# Create variables for URL generation
image_date = datetime.utcnow().date()
region = 'CONUS'
channel = 8

# We want to match something like:
# https://thredds-test.unidata.ucar.edu/thredds/catalog/satellite/goes16/GOES16/Mesoscale-1/Channel08/20181113/catalog.html

# Construct the data_url string
data_url = ('https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/'
            f'CloudAndMoistureImagery/{region}/Channel{channel:02d}/current/catalog.xml')

# Print out your URL and verify it works!
print(data_url)


# In[40]:


cat = TDSCatalog(data_url)


# In[45]:


##################################################
#
# Create File Inventories
#

file_names_to_retain = list()
file_names_to_use    = list()


for i in range(1,len(cat.datasets[0:total_frames])+1,1) : 
    filename = png_processing_directory + cat.datasets[i].name.replace(".nc",".png")
    file_names_to_retain.append(filename)
    file_names_to_use.append(filename)

        
files_on_hand = [png_processing_directory + s for s in os.listdir(png_processing_directory)]

file_names_to_retain.sort()
file_names_to_use.sort()

#
##################################################    


# In[ ]:





# In[34]:


for i in range(1,len(cat.datasets[1:total_frames])+1,1) : 

    dataset = cat.datasets[i]
    
    print(dataset.name.replace(".nc",".png")
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

    plt.suptitle("GOES 16 Band 8 [6.2 µm Upper-level Water Vapor]",
                    fontsize=20, color="black")
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_title(valid_time + "  (" + local_time+")",
                    fontsize=15, color="black")
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='black')

    im = ax.imshow(dat, extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')

    wv_norm, wv_cmap = colortables.get_with_range('WVCIMSS_r', 195, 265)
    im.set_cmap(wv_cmap)
    im.set_norm(wv_norm)
    plt.tight_layout()
    plt.savefig("./sat_wv_temp_files/Sat_WV_Loop_Image_"+file_time+".png")
    plt.close()
    


# In[35]:


##################################################
#
# Convert PNGs into an Animated GIF
#

os.system("convert -delay 15 " + 
          "./sat_wv_temp_files/Sat_WV_Loop_Image_*.png"  + 
          " " + 
          "./graphics_files/RealTime_SAT_WV_Loop.gif")


#
##################################################


# In[ ]:




