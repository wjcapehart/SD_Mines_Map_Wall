#!/bin/bash
echo on
echo
echo "Starting NAM Surface on Cyclone"
echo "Date: `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_namsfc/*.png
echo
echo  "Firing Things Up!"
echo
. source /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/NAM_SFC.py > /projects/SD_Mines_Map_Wall/__log_namsfc.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
