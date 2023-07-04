#!/bin/bash
. /home/wjc/.bashrc
cd /Users/wjc/GitHub/SD_Mines_Map_Wall/
convert -delay 25 ./temp_files_radar/Radar_Loop_Image_*.png ./graphics_files/RealTime_Radar_Loop.gif
echo MAIN:RADAR::: We^re Outahere Like Vladimir
