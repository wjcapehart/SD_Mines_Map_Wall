#!/usr/bin/env python
# coding: utf-8

# # NWS Warning Polygons

# In[1]:


#Import the necessary packages
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import noaa_sdk as noaa_sdk
import pandas as pd
import shapely as shapely
from   metpy.plots       import  USCOUNTIES



import timezonefinder    as tzf
import pytz              as pytz


import geopandas as gp


# In[ ]:





# In[2]:


proj_data_text = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'

gif_file_name = "./graphics_files/NWS_Warnings.png"
xls_file_name = "./graphics_files/NWS_Warnings.xlsx"



myproj = ccrs.AlbersEqualArea(central_longitude=-96, 
                              central_latitude=37.5, 
                              false_easting=0.0, 
                              false_northing=0.0, 
                              standard_parallels=(29.5, 45.5))



tz='America/Denver'

time_utc = dt.datetime.utcnow()
valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H%M %Z")
local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H%M %Z")

print(valid_time)
print(local_time)


# In[3]:


# Open the NWS API in python to get the active alerts
n = noaa_sdk.noaa.NOAA()
alerts = n.active_alerts()


# In[4]:


#injest //shapefiles/CONUS_UGC_Zones/


UGC_Zones_Shapefile    = gp.read_file('./shapefiles/CONUS_UGC_Zones/CONUS_UGC_Zones.shp')
UGC_Counties_Shapefile = gp.read_file('./shapefiles/CONUS_UGC_Counties/CONUS_UGC_Counties.shp')
 
UGC_Shapefile = pd.concat([UGC_Zones_Shapefile, 
                           UGC_Counties_Shapefile]).drop(["UGC_Temp",
                                                          "Shape_Leng",
                                                          "Shape_Le_1",
                                                          "Shape_Area",
                                                          "STATE",
                                                          "CWA",
                                                          "COUNTYNAME",
                                                          "FIPS",
                                                          "TIME_ZONE",
                                                          "FE_AREA"],
                                                          axis = 'columns').reset_index(drop=True)  
    
UGC_Zone_County_List = UGC_Shapefile['UGC'].to_list()



# In[5]:


# priority warnings table
warning_priority_table = pd.read_csv("./warning_table_sorted.csv")

warning_priority_table = warning_priority_table.rename(columns={"hdln": "event"})



# In[6]:


current_warnings = pd.DataFrame(columns = ['event',
                                           'UGC',
                                           'coverage',
                                           'message'])



i = 0



for alert in alerts['features']:
    event = alert['properties']['event']
    if (event != "Test Message"): 
        
        try:

            ugc_codes = alert['properties']['geocode']['UGC']
            message   = alert['properties']['@id']

            for ugc_code in ugc_codes:
                if ugc_code in UGC_Zone_County_List: 
                    if (ugc_code[2] == 'C'):
                        coverage = "County"
                    else:
                        coverage = "Zones"

                    deleteme = pd.DataFrame([[event,
                              ugc_code,
                              coverage,
                              message]], 
                              columns = ['event',
                                         'UGC',
                                         'coverage',
                                         'message'])
                    current_warnings = pd.concat([current_warnings, 
                                                  deleteme]) 
                    i = i + 1
        except KeyError:
            print("poopie:")
            print(event, ugc_code)
    
current_warnings = current_warnings.merge(warning_priority_table, how='left', on='event')
current_warnings = UGC_Shapefile.merge(current_warnings, how='right', on='UGC')
current_warnings = current_warnings.sort_values("Rank", ascending=False)


if (current_warnings['color'].isnull().values.any()):
    print("Missing Colors Found")
    locs_missing = current_warnings.index[current_warnings['color'].isnull()].tolist()
    print(current_warnings.loc[locs_missing])

  
    for loc_missing in locs_missing:
        current_warnings.loc[loc_missing,"color"] = 'red'
    
print("replaced")


warning_color_table = current_warnings[["event","color"]].drop_duplicates()


current_warnings.drop(["geometry"], axis="columns").to_excel(xls_file_name)

print("done: ",i,"rows; ",len(warning_color_table),"event types")
print()


# In[7]:


# check for missing colors.

if (warning_color_table['color'].isnull().values.any()):
    print("Missing Colors Found")
    loc_missing = warning_color_table.index[warning_color_table['color'].isnull()].tolist()
    print(loc_missing)
    print(warning_color_table.loc[loc_missing])
    warning_color_table.loc[loc_missing,"color"] = '#ffffff'
    
print("replaced")
print(warning_color_table)


# In[8]:


legend_color_table = warning_color_table.values.tolist()
print(legend_color_table)
legend_color_table = []

for row in warning_color_table.iterrows():
    mypatch = [mpatches.Patch(color=row[1][1], label=row[1][0])]
    legend_color_table = legend_color_table + mypatch
    
    
  


# In[9]:


print(warning_color_table)


# In[10]:


bbox=[-120,-73,22.5,50]

xxx = [-120.0, -73.0]
yyy = [  22.5,  50.0]


fig = plt.figure(figsize   = (11, 6), 
                 facecolor = 'white')

ax = fig.add_subplot(1, 1, 1, 
                     projection=myproj)

plt.suptitle("NWS Watches and Warnings",
             fontsize = 20, 
             color    = "black")
ax.set_extent(bbox)
ax.set_title(valid_time + "  (" + local_time+")",
             fontsize=15, 
             color="black")

current_warnings.plot(ax = ax,aspect='equal',
                      facecolor=current_warnings["color"],
                      transform=ccrs.PlateCarree(),
                      edgecolor='None', linewidth=0)

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
               linewidth = 0.5)
ax.add_feature(cfeature.STATES.with_scale('50m'),    
               linewidth = 0.25, 
               edgecolor = 'black')
ax.add_feature(cfeature.LAKES.with_scale('50m'),   
               linewidth = 0.5,
               facecolor = "none", 
               edgecolor = 'black')
ax.add_feature(feature    = USCOUNTIES, 
                   linewidths = 0.1,
                   edgecolor  = 'black',
                   facecolor  = 'none')

ax.set_frame_on(False)

plt.subplots_adjust(left   = 0.01, 
                    right  = 0.75, 
                    top    = 0.91, 
                    bottom = 0.01)

labelspacing = 0.1
fig.legend(handles  = legend_color_table, 
           loc      = 'right',
           frameon  = False,
           labelspacing = labelspacing)




#########################################
#
# Insert a Clock
#

axins = fig.add_axes(rect     =    [0, # 0.015,
                                    1-0.12, #0.015,
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

axins.plot([angles_h,angles_h], [0,0.6], color="black", linewidth=1.5)
axins.plot([angles_m,angles_m], [0,0.95], color="black", linewidth=1.5)
axins.plot(circle_theta, circle_radius, color="darkgrey", linewidth=1)


#
#########################################
        
        
plt.savefig(gif_file_name,
                        facecolor   = 'white', 
                        transparent =   False)


plt.close()


print("done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




