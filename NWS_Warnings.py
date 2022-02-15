#!/usr/bin/env python
# coding: utf-8

# NWS Warning Board

# In[ ]:


from awips.dataaccess import DataAccessLayer
from awips.tables import vtec
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature,NaturalEarthFeature
from shapely.geometry import MultiPolygon,Polygon
import pyproj as pyproj
import pandas as pd
import matplotlib.patches as mpatches
from   metpy.plots       import  USCOUNTIES


# In[ ]:


proj_data_text = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'

gif_file_name = "./graphics_files/NWS_Warnings.png"


myproj = ccrs.AlbersEqualArea(central_longitude=-96, 
                              central_latitude=37.5, 
                              false_easting=0.0, 
                              false_northing=0.0, 
                              standard_parallels=(29.5, 45.5))
def warning_color(phensig):
    return vtec[phensig]['color']



# In[ ]:


tz='America/Denver'

time_utc = datetime.utcnow()
valid_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").strftime("%Y-%m-%d %H%M %Z")
local_time = pd.to_datetime(time_utc).tz_localize(tz="UTC").tz_convert(tz=tz).strftime("%Y-%m-%d %H%M %Z")

print(valid_time)
print(local_time)


# In[ ]:


DataAccessLayer.changeEDEXHost("edex-cloud.unidata.ucar.edu")
request = DataAccessLayer.newDataRequest()
request.setDatatype("warning")
request.setParameters('phensig')
times   = DataAccessLayer.getAvailableTimes(request)



# In[ ]:



# Get records for last 50 available times
response = DataAccessLayer.getGeometryData(request, times[-50:-1])
print("Using " + str(len(response)) + " records")


# In[ ]:


# Each record will have a numpy array the length of the number of "parameters"
# Default is 1 (request.setParameters('phensig'))
parameters = {}
for x in request.getParameters():
    parameters[x] = np.array([])


# In[ ]:



current_warnings = pd.DataFrame(columns = ['hdln',
                                           'phensigString',
                                           'site',
                                           'ref',
                                           'color'])

siteids=np.array([])
periods=np.array([])
reftimes=np.array([])


for ob in response:

    site = ob.getLocationName()
    per  = ob.getDataTime().getValidPeriod()
    ref  = ob.getDataTime().getRefTime()

    # do not plot if phensig is blank (SPS)
    if (1>0) : #ob.getString('phensig'):

        phensigString = ob.getString('phensig')

        siteids = np.append(siteids,site)
        periods = np.append(periods,per)
        reftimes = np.append(reftimes,ref)

        for parm in parameters:
            parameters[parm] = np.append(parameters[parm],ob.getString(parm))

  

        if (0 > 1) : 
            print(vtec[phensigString]['hdln']
              + " (" + phensigString + ") issued at " + str(ref)
              + " ("+str(poly.geom_type) + geom_count + ")")

        color = warning_color(phensigString)

        
        deleteme = pd.DataFrame([[vtec[phensigString]['hdln'],
                                  phensigString,
                                  site,
                                  ref,
                                  color]], 
                                  columns = ['hdln',
                                           'phensigString',
                                           'site',
                                           'ref',
                                           'color'])
        
        current_warnings = pd.concat([current_warnings, deleteme]) 
        
current_warnings = current_warnings.reset_index().drop(columns=["index"])


current_warnings.to_excel("./Current_Warnings.xlsx")

warning_color_table = current_warnings.drop(columns=["phensigString","site","ref"]).drop_duplicates()


legend_color_table = warning_color_table.values.tolist()
legend_color_table



legend_color_table = []

for row in warning_color_table.iterrows():
    mypatch = [mpatches.Patch(color=row[1][1], label=row[1][0])]
    legend_color_table = legend_color_table + mypatch
    
  


# In[ ]:


bbox=[-120,-73,22.5,50]

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





siteids=np.array([])
periods=np.array([])
reftimes=np.array([])

first = True

for ob in response:

    poly = ob.getGeometry()
    site = ob.getLocationName()
    per  = ob.getDataTime().getValidPeriod()
    ref  = ob.getDataTime().getRefTime()

    # do not plot if phensig is blank (SPS)
    if (1 > 0) : #ob.getString('phensig'):

        phensigString = ob.getString('phensig')

        siteids = np.append(siteids,site)
        periods = np.append(periods,per)
        reftimes = np.append(reftimes,ref)

        for parm in parameters:
            parameters[parm] = np.append(parameters[parm],ob.getString(parm))

        if poly.geom_type == 'MultiPolygon':
            geometries = np.array([])
            geometries = np.append(geometries,MultiPolygon(poly))
            geom_count = ", " + str(len(geometries)) +" geometries"
        else:
            geometries = np.array([])
            geometries = np.append(geometries,Polygon(poly))
            geom_count=""

        for geom in geometries:
            bounds = Polygon(geom)
            intersection = bounds.intersection
            geoms = (intersection(geom)
                 for geom in geometries
                 if bounds.intersects(geom))

        if (0 > 1) : 
            print(vtec[phensigString]['hdln']
              + " (" + phensigString + ") issued at " + str(ref)
              + " ("+str(poly.geom_type) + geom_count + ")")

        color = warning_color(phensigString)
        shape_feature = ShapelyFeature(geoms,ccrs.PlateCarree(),
                        facecolor=color, edgecolor=color)
        ax.add_feature(shape_feature)



ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
               linewidth = 0.5)
ax.add_feature(cfeature.STATES.with_scale('50m'),    
               linewidth = 0.25, 
               edgecolor = 'black')
ax.add_feature(cfeature.BORDERS.with_scale('50m'),   
               linewidth = 0.5, 
               edgecolor = 'black')
ax.add_feature(cfeature.LAKES.with_scale('50m'),   
               linewidth = 0.5,
               facecolor = "none", 
               edgecolor = 'black')
ax.add_feature(feature    = USCOUNTIES, 
                   linewidths = 0.1,
                   edgecolor  = 'black',
                   facecolor  = 'none')


plt.subplots_adjust(left=.01, 
                        right=.78, 
                        top=0.91, 
                        bottom=.01)

fig.legend(handles=legend_color_table, loc='right',frameon=False)


#plt.tight_layout()

plt.savefig(gif_file_name)

plt.close()


print("done")


# In[ ]:





# In[ ]:





# In[ ]:




