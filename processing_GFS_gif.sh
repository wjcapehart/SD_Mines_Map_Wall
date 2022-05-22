#!/bin/bash
. /home/wjc/.bashrc
cd /Users/wjc/GitHub/SD_Mines_Map_Wall/
convert -delay 10 ./temp_files_gfs/GFS_4_Panel_*.png ./graphics_files/GFS_4_Panel.gif
echo MAIN:GFS:: We're Outahere Like Vladimir
