#!/bin/bash
echo on
echo
echo "Starting Water Vapor Images Display Scriot on Cyclone"
echo "Date: `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
echo
echo  "Firing Things Up!"
echo
. source /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/SAT_VIS-IR_Images_Meso.py > /projects/SD_Mines_Map_Wall/__log_sat_Meso.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off