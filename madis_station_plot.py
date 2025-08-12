#!/usr/bin/env python
# coding: utf-8

# Madis Public text service
# 
# ```
# https://madis-data.ncep.noaa.gov/madisPublic1/cgi-bin/madisXmlPublicDir?rdr=&time=0&minbck=-59&minfwd=0&recwin=2&dfltrsel=3&stanam=KRAP&stasel=1&pvdrsel=1&varsel=1&qctype=0&qcsel=0&xml=2&csvmiss=0&pvd=ASOS-HFM&nvars=TD&nvars=ALTSE&nvars=P&nvars=T&nvars=U&nvars=V&nvars=VIS&nvars=PRESWEA&nvars=SKYCOV
# 
#  STAID     ,OBDATE    ,OBTIME,PVDR     ,SUBPVDR    ,TD           ,ALTSE        ,P            ,T            ,U            ,V            ,DIS          ,PRESWEA                 ,SKYCOV_1,SKYCOV_2,SKYCOV_3,SKYCOV_4,SKYCOV_5,SKYCOV_6 ,
#  KRAP      ,08/12/2025,15:20,ASOS-HFM  ,           ,   285.149994,102065.750000, 90951.468750,   297.149994,     1.231653,    -3.383937, 16093.440430,                         ,CLR     ,        ,        ,        ,        ,         ,
#  
# ```
# Netcdf files for madis
# 
# https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/?C=M;O=D
# 
# 

# In[1]:


import metpy as metpy
import numpy as np
import xarray as xr 
import pandas as pd
import requests as requests
import os as os
import requests
from bs4 import BeautifulSoup
import urllib.request
import warnings
import matplotlib.pyplot as plt

from   datetime          import datetime, timedelta, timezone
import timezonefinder     as tzf
from   metpy.plots        import colortables, USCOUNTIES, StationPlot, sky_cover,current_weather


# In[ ]:





# In[ ]:





# In[2]:


madis_metar_url = "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/"
page = requests.get(madis_metar_url).text

soup = BeautifulSoup(page, 'html.parser')

links = [madis_metar_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('gz')]

links.sort()
links

new = True
for link in links[-2:]:
    filename = link.split('/')[-1]
    urllib.request.urlretrieve(link,filename)
    os.system("gunzip -v "+filename)
    file_root = filename.strip(".gz")
    os.rename(file_root,file_root + ".nc")
    if (new):
        warnings.filterwarnings("ignore")
        df1      = xr.open_dataset(filename_or_obj = file_root + ".nc")
        warnings.filterwarnings("default")
        os.remove(file_root + ".nc")
        new = False
    else:
        warnings.filterwarnings("ignore")
        df2      = xr.open_dataset(filename_or_obj = file_root + ".nc")
        warnings.filterwarnings("default")
        df_metar = xr.concat([df1,df2], dim="recNum")
        os.remove(file_root + ".nc")
        del df1
        del df2

ds_metar = df_metar.sortby("observationTime")
ds_metar["recNum"] = np.arange(ds_metar["recNum"].size)


target_dimension = "recNum"
filtered_variables = {}

for var_name, var_data in ds_metar.data_vars.items():
    if       (target_dimension in var_data.dims)  and \
       ( not ("maxStaticIds"   in var_data.dims)) and \
       ( not ("QCcheckNum"     in var_data.dims)) and \
       ( not ("ICcheckNum"     in var_data.dims)) and \
       ( not ("nInventoryBins" in var_data.dims)):
        if (not ("DD"  in var_name)) and \
           (not ("QCA" in var_name)) and \
           (not ("QCR" in var_name)) and \
           (not ("ICA" in var_name)) and \
           (not ("Status" in var_name)) and \
           (not ("ICR" in var_name)):
            filtered_variables[var_name] = var_data

ds_metar = xr.Dataset(filtered_variables)
del filtered_variables
ds_metar = ds_metar.drop_vars(names = ["nStaticIds",
                                       "invTime",
                                       "prevRecord",
                                       "globalInventory",
                                       "firstOverflow",
                                       "isOverflow",
                                       "secondsStage1_2",
                                       "secondsStage3",
                                       "test1",
                                       "filterSetNum",
                                       "reportTime",
                                       "receivedTime",
                                       "modifyTime",
                                       "providerId",
                                       "handbook5Id",
                                       "stationName",
                                       "homeWFO","dataProvider",
                                       "numericWMOid"])
ds_metar["stationId"]      = ds_metar["stationId"].astype(str)
ds_metar["stationType"]    = ds_metar["stationType"].astype(str)
ds_metar["rawMessage"]     = ds_metar["rawMessage"].astype(str)
ds_metar["presWx"]         = ds_metar["presWx"].astype(str)
ds_metar["skyCvr"]         = ds_metar["skyCvr"].astype(str)
ds_metar["autoRemark"]     = ds_metar["autoRemark"].astype(str)
ds_metar["operatorRemark"] = ds_metar["operatorRemark"].astype(str)

ds_metar_nocloud = ds_metar.drop_vars(names=["skyCvr","skyCovLayerBase"])
ds_metar_cloud   = ds_metar[["skyCvr","skyCovLayerBase"]]

df_metar_nocloud = ds_metar_nocloud.to_dataframe()
df_metar_cloud   = ds_metar_cloud.to_dataframe()

df_metar_cloud         = df_metar_cloud.unstack()
df_metar_cloud.columns = [f'{i}_{j+1}' for i, j in df_metar_cloud.columns]

df_metar = df_metar_nocloud.join(other = df_metar_cloud, on="recNum")




# In[3]:


df_latest = df_metar[df_metar["stationId"]=="KRAP"].set_index("observationTime").sort_index(ascending=False).iloc[0]

tf     = tzf.TimezoneFinder()
tz     = tf.certain_timezone_at(lng = df_latest['longitude'], 
                                lat = df_latest['latitude'])

df_latest["TimeUTC"] = pd.to_datetime(df_latest.name).tz_localize("UTC")
df_latest["TimeLOC"] = df_latest["TimeUTC"].tz_convert(tz)
df_latest["temperatureF"] = 1.8*(df_latest["temperature"] - 273.15) + 32
df_latest["dewpointF"] = 1.8*(df_latest["dewpoint"] - 273.15) + 32
df_latest["MSLP"] = metpy.calc.altimeter_to_sea_level_pressure(df_latest["altimeter"]   * metpy.units.units("Pa"), 
                                                               df_latest["elevation"]   * metpy.units.units("m"), 
                                                               df_latest["temperature"] * metpy.units.units("K")).magnitude / 100.



skyc = df_latest["skyCvr_1"] + " " + df_latest["skyCvr_2"] + " " + df_latest["skyCvr_3"]



# Set the cloud cover variable (measured in oktas)
if 'OVC' in skyc or 'VV' in skyc:
    df_latest["cloud_eights"] = 8
elif 'BKN' in skyc:
    df_latest["cloud_eights"] = 6
elif 'SCT' in skyc:
    df_latest["cloud_eights"] = 4
elif 'FEW' in skyc:
    df_latest["cloud_eights"] = 2
elif ('SKC' in skyc or 'NCD' in skyc or 'NSC' in skyc
      or 'CLR' in skyc or 'CAVOK' in skyc):
    df_latest["cloud_eights"] = 0
else:
    df_latest["cloud_eights"] = 10    

df_latest["cloud_tenths"] = df_latest["cloud_eights"]/8.    


current_wx = []
current_wx_symbol = []
if df_latest["presWx"].strip() not in ('', '//', 'NSW'):
    current_wx = df_latest["presWx"].strip().split()

    # Handle having e.g. '+' and 'TSRA' parsed into separate items
    if current_wx[0] in ('-', '+') and current_wx[1]:
        current_wx[0] += current_wx[1]
        current_wx.pop(1)

    current_wx_symbol = metpy.plots.wx_code_to_numeric(current_wx).tolist()
while len(current_wx) < 3:
    current_wx.append(np.nan)
while len(current_wx_symbol) < 3:
    current_wx_symbol.append(0)

print(current_wx)
print(current_wx_symbol)

df_latest["current_wx1"] = current_wx[0]
df_latest["current_wx2"] = current_wx[1]
df_latest["current_wx3"] = current_wx[2]

df_latest["current_wx1_symbol"] = current_wx_symbol[0]
df_latest["current_wx2_symbol"] = current_wx_symbol[1]
df_latest["current_wx3_symbol"] = current_wx_symbol[2]


df_latest["u_kts"] = -1.94384 * df_latest["windSpeed"] * np.sin(np.radians(df_latest["windDir"]))
df_latest["v_kts"] = -1.94384 * df_latest["windSpeed"] * np.cos(np.radians(df_latest["windDir"]))

df_latest["speed_kts"] = 1.94384 * df_latest["windSpeed"] 

df_latest["gust_kts"] = 1.94384 * df_latest["windGust"] 


# In[4]:


df_latest


# In[ ]:





# In[7]:


Mines_Blue = "#002554"

bottom_label = df_latest["stationId"] + " " + df_latest["TimeLOC"].tz_convert(tz=tz).strftime("%H:%M %Z")
temp_label = f"{df_latest['temperatureF']:.0f}°F"
dewp_label = f"{df_latest['dewpointF']:.0f}°F"
pres_label = f"  {df_latest['MSLP']:.0f}ᴹᴮ"


fig, ax = plt.subplots(figsize=[5,5])
stationplot = StationPlot(ax, 
                          0, 
                          0, 
                          #transform = ccrs.PlateCarree(),
                          fontsize  = 50)

stationplot.plot_text('NW', 
                           np.array( [temp_label] ), 
                           color=Mines_Blue,
                           fontsize=45)
stationplot.plot_text('SW', 
                           np.array([dewp_label]), 
                           color=Mines_Blue,
                           fontsize=45)

#stationplot.plot_text('NE', 
#                           np.array([pres_label]), 
#                           color=Mines_Blue,
#                           fontsize=25,ha="center")


stationplot.plot_symbol('C', 
                        np.array([df_latest['cloud_eights']]), 
                        sky_cover,color=Mines_Blue)

stationplot.plot_symbol('W', 
                        np.array([df_latest['current_wx1_symbol']]), 
                        current_weather,color=Mines_Blue)

stationplot.plot_barb(np.array([df_latest['u_kts']]), 
                      np.array([df_latest['v_kts']]),
                     color=Mines_Blue, linewidth=2)

stationplot.plot_text((0, -2), 
                      np.array([bottom_label]), 
                      color=Mines_Blue,
                      fontsize=25, ha="center")
plt.axis('off')
plt.savefig("./graphics_files/local_station.svg", 
            bbox_inches = "tight", 
            pad_inches  = 0.0)
#plt.show()
plt.close()

print("Completed")


# In[ ]:





# In[ ]:




