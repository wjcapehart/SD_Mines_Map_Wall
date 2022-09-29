#!/usr/bin/env python
# coding: utf-8

# # Water Vapor

# In[ ]:


##################################################
#
# Libraries
#


from metpy.plots    import colortables
from metpy.plots    import add_timestamp
from datetime       import datetime
from siphon.catalog import TDSCatalog
from datetime       import datetime

import numpy             as np
import os                as os
import pandas            as pd
import pathlib           as pathlib

import metpy
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import matplotlib.pyplot as plt


#
##################################################


# In[ ]:


##################################################
#
# Control Setup
#

# %load solutions/data_url.py

MAINDIR = os.getcwd() + "/"
print(MAINDIR)




total_frames = 45*2

png_processing_directory = "./temp_files_sat_wv/"

gif_file_name = "./graphics_files/RealTime_SAT_WV_Loop.gif"

image_header_label = "GOES 16 Band 8 [6.2 µm Upper-level Water Vapor]"

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

#
##################################################


# In[ ]:


##################################################
#
# Pull Catalog
#

cat = TDSCatalog(data_url)

#
##################################################


# In[ ]:


##################################################
#
# Create File Inventories
#

file_names_to_retain = list()
file_names_to_use    = list()


for i in range(len(cat.datasets)-total_frames,len(cat.datasets),1) :
    filename = png_processing_directory + cat.datasets[i].name.replace(".nc",".png")
    file_names_to_retain.append(filename)
    file_names_to_use.append(filename)

        
files_on_hand = [png_processing_directory + s for s in os.listdir(png_processing_directory)]

file_names_to_retain.sort()
file_names_to_use.sort()


#
##################################################    


# In[ ]:


##################################################
#
# Clean PNG Directory
#

for filename in files_on_hand:
    if filename not in file_names_to_retain:
        print("Purging ", filename )
        os.remove( filename  )
    else:
        print("Keeping ", filename )
#
##################################################    


# In[ ]:


##################################################
#
# Create PNGs
#

for i in range(len(cat.datasets)-total_frames,len(cat.datasets),1) :

    dataset = cat.datasets[i]
    
    dataset_png_file_name = png_processing_directory + dataset.name.replace(".nc", ".png")
    
    if (not pathlib.Path(dataset_png_file_name).is_file() ):

        ds = dataset.remote_access(use_xarray=True)
        dat = ds.metpy.parse_cf('Sectorized_CMI')
        proj = dat.metpy.cartopy_crs
        x = dat['x']
        y = dat['y']


        tz='America/Denver'
        time_utc   = datetime.strptime(ds.start_date_time, '%Y%j%H%M%S')
        valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H:%M:%S %Z")
        local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        file_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d_%H%M")


        print(valid_time,local_time)



        fig = plt.figure(figsize=(12.25, 8), facecolor = 'white')

        plt.suptitle(image_header_label,
                     fontsize = 20, 
                     color    = "black")
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_title(valid_time + "  (" + local_time+")",
                        fontsize=15, color="black")
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
        ax.add_feature(cfeature.STATES.with_scale('50m'),    linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=2, edgecolor='black')

        im = ax.imshow(dat, extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')

        wv_norm, wv_cmap = colortables.get_with_range('WVCIMSS_r', 195, 265)
        im.set_cmap(wv_cmap)
        im.set_norm(wv_norm)
        
        #. plt.tight_layout()
        plt.subplots_adjust(left   = 0.01, 
                            right  = 0.99, 
                            top    = 0.91, 
                            bottom = 0, 
                            wspace = 0)
        
                #########################################
        #
        # Insert a Clock
        #
        
        axins = fig.add_axes(rect     =    [0.01,
                                            0.81,
                                            0.12*0.65306121,
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
        ax.set_frame_on(False)
 
        plt.savefig( dataset_png_file_name,
                        facecolor   = 'white', 
                        transparent =   False)
        plt.close()
    else:
        print("We already have this one!")
    

#
##################################################


# In[ ]:


##################################################
#
# Convert PNGs into an Animated GIF
#

big_string = " ".join(file_names_to_use)




print("creating " + MAINDIR + "./processing_sat_WV_gif.sh")
with open(MAINDIR + "./processing_sat_WV_gif.sh", 'w') as f:
    print("#!/bin/bash", file =  f)
    print(". /home/wjc/.bashrc", file = f)
    print("cd " + MAINDIR, file =  f) 
    print("convert -delay 10 " + 
          big_string + 
          " " + 
          gif_file_name, file =  f) 
    print("echo MAIN:SAT_WV::: We^re Outahere Like Vladimir", file =  f) 

os.system("chmod a+x " + MAINDIR + "./processing_sat_WV_gif.sh")
os.system(MAINDIR + "./processing_sat_WV_gif.sh > ./processing_sat_WV_gif.LOG 2>&1 ")
os.system("date")
print()

print("completed "+ gif_file_name)







#
##################################################


# In[ ]:




