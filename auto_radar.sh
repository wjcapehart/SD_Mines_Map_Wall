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
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_radar/*.png
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_radar/*.txt
echo
echo  "Firing Things Up!"
echo
.  /home/wjc/.bashrc ; $CONDA_PYTHON_EXE /projects/SD_Mines_Map_Wall/Radar_and_Metars.py > /projects/SD_Mines_Map_Wall/__log_radar.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
