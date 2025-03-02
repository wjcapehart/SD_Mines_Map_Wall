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
.  /home/wjc/.bashrc ; $CONDA_PYTHON_EXE /projects/SD_Mines_Map_Wall/SAT_WV_Images.py > /projects/SD_Mines_Map_Wall/__log_sat_wv.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
