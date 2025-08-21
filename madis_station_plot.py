#!/usr/bin/env python
# coding: utf-8

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
from   metpy.plots        import colortables, USCOUNTIES, StationPlot, pressure_tendency,sky_cover,current_weather
pd.options.mode.copy_on_write = True


# In[2]:


stationname     = "KRAP"
madis_metar_url = "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/"
page = requests.get(madis_metar_url).text

soup = BeautifulSoup(page, 'html.parser')

links = [madis_metar_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('gz')]

links.sort()



# In[ ]:






# In[3]:


new = True
for link in links[-4:]:
    filename = link.split('/')[-1]
    urllib.request.urlretrieve(link,filename)
    os.system("gunzip -v "+filename)
    file_root = filename.strip(".gz")
    os.rename(file_root,file_root + ".nc")
    if (new):
        warnings.filterwarnings("ignore")
        df_metar      = xr.open_dataset(filename_or_obj = file_root + ".nc")
        warnings.filterwarnings("default")
        os.remove(file_root + ".nc")
        new = False
    else:
        warnings.filterwarnings("ignore")
        df2      = xr.open_dataset(filename_or_obj = file_root + ".nc")
        warnings.filterwarnings("default")
        df_metar = xr.concat([df_metar,df2], dim="recNum")
        os.remove(file_root + ".nc")


ds_metar           = df_metar.sortby("observationTime")
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

df_metar = df_metar[df_metar["stationId"]==stationname]. \
                set_index("observationTime")

df_metar["temperatureF"] = 1.8*(df_metar["temperature"] - 273.15) + 32
df_metar["dewpointF"]    = 1.8*(df_metar["dewpoint"]    - 273.15) + 32


df_metar.sort_index(inplace=True)
df_metar["secs_from_3hrs"] = np.abs((pd.to_datetime(df_metar.index[-1])-pd.to_datetime(df_metar.index)).astype(int)/1e9-3*3600)
df_metar["secs_from_1p5hrs"] = ((pd.to_datetime(df_metar.index[-1])-pd.to_datetime(df_metar.index)).astype(int)/1e9-3*1800)

df_metar["cloud_tenths"] = np.nan
df_metar["cloud_eights"] = np.nan
df_metar["TimeUTC"] = np.nan
df_metar["TimeLOC"] = np.nan

tf     = tzf.TimezoneFinder()
tz     = tf.certain_timezone_at(lng = df_metar.iloc[0]['longitude'], 
                                lat = df_metar.iloc[0]['latitude'])

df_metar["TimeUTC"] = pd.to_datetime(df_metar.index).tz_localize("UTC")

df_metar["TimeLOC"] = pd.to_datetime(df_metar.index).tz_localize("UTC").tz_convert(tz)


for index, row in df_metar.iterrows():
    # Set the cloud cover variable (measured in oktas)
    skyc = row["skyCvr_1"] + " " + row["skyCvr_2"] + " " + row["skyCvr_3"]



    if  ('OVC' in skyc or 
         'VV'  in skyc):
        df_metar.loc[index, "cloud_eights"] = 8
    elif 'BKN' in skyc:
        df_metar.loc[index, "cloud_eights"] = 6
    elif 'SCT' in skyc:
        df_metar.loc[index, "cloud_eights"] = 4
    elif 'FEW' in skyc:
        df_metar.loc[index, "cloud_eights"] = 2
    elif ('SKC'   in skyc or 
          'NCD'   in skyc or 
          'NSC'   in skyc or 
          'CLR'   in skyc or 
          'CAVOK' in skyc):
        df_metar.loc[index, "cloud_eights"] = 0
    else:
        df_metar.loc[index, "cloud_eights"] = 10    

    df_metar.loc[index, "cloud_tenths"] = df_metar.loc[index, "cloud_eights"]/8.  


    df_metar.loc[index, "MSLP"] = metpy.calc.altimeter_to_sea_level_pressure(row["altimeter"]   * metpy.units.units("Pa"), 
                                                                             row["elevation"]   * metpy.units.units("m"), 
                                                                             row["temperature"] * metpy.units.units("K")).magnitude / 100.

    #print(f"Index: {index}, skyc: {skyc}, Sky10: {row['cloud_tenths']}")


first_index = np.where(df_metar["secs_from_3hrs"] == df_metar["secs_from_3hrs"].min())[0][0]
df_metar = df_metar[first_index:] 
df_half1 = df_metar[df_metar["secs_from_1p5hrs"]>=0]
df_half2 = df_metar[df_metar["secs_from_1p5hrs"]<=0]


df_3hours = df_metar.iloc[ 0]
df_latest = df_metar.iloc[-1]

ptrend =  (df_latest["MSLP"] - df_3hours["MSLP"])


# In[4]:


df_metar


# In[5]:


line_half1 = np.polyfit(df_half1["secs_from_3hrs"]/3600 ,df_half1["MSLP"], 1)
line_half2 = np.polyfit(df_half2["secs_from_3hrs"]/3600 ,df_half2["MSLP"], 1)
line_all   = np.polyfit(df_metar["secs_from_3hrs"]/3600 ,df_metar["MSLP"], 1)
para_all   = np.polyfit(df_metar["secs_from_3hrs"]/3600 ,df_metar["MSLP"], 2)

trend_1rst = round(line_half1[0]*1.5,1)
trend_2rst = round(line_half2[0]*1.5,1)

print("Total Change : ",round(ptrend/3,2),      "mb/hr;", round(        ptrend,1), "mb")
print(" Mean Change : ",round(line_all[0],2),   "mb/hr;", round( line_all[0]*3,1), "mb")
print("  First Half : ",round(line_half1[0],2), "mb/hr;", round(trend_1rst,1), "mb")
print(" Second Half : ",round(line_half2[0],2), "mb/hr;", round(trend_2rst,1), "mb")


if   (ptrend >  0.05):
    if (trend_1rst >  0) & (trend_2rst >  0):
        trend_code = 2
    elif (trend_1rst >  0) & (trend_2rst ==  0):
        trend_code = 1
    elif (trend_1rst >  0) & (trend_2rst <  0):
        trend_code = 0
    elif (trend_1rst <  0) & (trend_2rst >  0):
        trend_code = 3
elif (ptrend < -0.05):
    if (trend_1rst <  0) & (trend_2rst <  0):
        trend_code = 7
    elif (trend_1rst >  0) & (trend_2rst ==  0):
        trend_code = 6
    elif (trend_1rst >  0) & (trend_2rst <  0):
        trend_code = 8
    elif (trend_1rst <  0) & (trend_2rst >  0):
        trend_code = 9
else:
    trend_code = 4

print("trend code:",trend_code)



# 
# plt.scatter(df_half1["secs_from_3hrs"]/3600, df_half1["MSLP"],
#             marker =         'o', 
#             color  =       "red", 
#             alpha  =         0.5,
#             label  = "1rst Half")
# plt.scatter(df_half2["secs_from_3hrs"]/3600, df_half2["MSLP"],
#             marker =         'o', 
#             color  =      "blue", 
#             alpha  =         0.5,
#             label  = "2nd Half")
# 
# plt.plot((df_metar["secs_from_3hrs"]/3600),
#          (df_metar["secs_from_3hrs"]/3600) **2 * para_all[0] + 
#          (df_metar["secs_from_3hrs"]/3600)     * para_all[1] +  para_all[2], 
#          color = "grey",
#          label = "2nd-order trend")
# plt.axline(slope =       line_all[0], 
#            xy1   =   [0,line_all[1]], 
#            color =          "purple",
#            alpha =              0.90)
# plt.axline(slope =     line_half1[0], 
#            xy1   = [0,line_half1[1]], 
#            color =             "red",
#            alpha =              0.75)
# plt.axline(slope =     line_half2[0], 
#            xy1   = [0,line_half2[1]], 
#            color =            "blue",
#            alpha =              0.75)
# 
# plt.legend()
# plt.xlabel("hours")
# plt.ylabel("Mean Sea Level Pressure (hPa)")
# 
# plt.show()

# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


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


# In[ ]:





# In[7]:


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


# In[ ]:





# In[ ]:





# In[8]:


Mines_Blue = "#002554"

bottom_label  = df_latest["stationId"] + " " + df_latest["TimeLOC"].tz_convert(tz=tz).strftime("%H:%M %Z")
temp_label    = f"{df_latest['temperatureF']:.0f}°F  "
dewp_label    = f"{df_latest['dewpointF']:.0f}°F  "
pres_label    = f" {df_latest['MSLP']:.0f}ᵐᵇ"
pres_tend_lab = f"   {ptrend:+4.1f}ᵐᵇ"

if ((df_latest['visibility']/1609) > .9):
    vis_label    = f"{(df_latest['visibility']/1609):.0f}ᵐⁱ"
else:
    vis_label    = f"{round(df_latest['visibility']/1609,1):.1f}ᵐⁱ"

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

stationplot.plot_text('NE', 
                           np.array([pres_label]), 
                           color=Mines_Blue,
                           fontsize=25)

stationplot.plot_text('E', 
                           np.array([pres_tend_lab]), 
                           color=Mines_Blue,
                           fontsize=20)

stationplot.plot_symbol('E2', 
                           np.array([trend_code]).astype("int"), 
                           pressure_tendency,
                           color=Mines_Blue,
                           fontsize=25)


stationplot.plot_symbol('C', 
                        np.array([df_latest['cloud_eights']]).astype("int"), 
                        sky_cover,color=Mines_Blue)

stationplot.plot_symbol('W', 
                        np.array([df_latest['current_wx1_symbol']]), 
                        current_weather,color=Mines_Blue)
stationplot.plot_text('W2', 
                           np.array([vis_label]), 
                           color=Mines_Blue,
                           fontsize=20)


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





# In[ ]:





# In[ ]:




