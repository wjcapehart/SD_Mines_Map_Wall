#!/bin/bash
echo on
echo
echo "Starting Radar Display Scriot on Cyclone"
echo "Date: `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
rm -frv  /projects/SD_Mines_Map_Wall/radar_temp_files/*.png
rm -frv  /projects/SD_Mines_Map_Wall/radar_temp_files/*.txt
echo
echo  "Firing Things Up!"
echo
. source /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/Radar_and_Metars.py > /projects/SD_Mines_Map_Wall/__log_radar_${DATESTRING}.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
