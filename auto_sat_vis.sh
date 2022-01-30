#!/bin/bash
echo on
echo
echo "Starting Visible Images Display Scriot on Cyclone"
echo "Date: `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
rm -frv  /projects/SD_Mines_Map_Wall/sat_vis_files/*.png
echo
echo  "Firing Things Up!"
echo
. source /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/SAT_VIS_Images.py > /projects/SD_Mines_Map_Wall/__log_sat_vis_${DATESTRING}.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
