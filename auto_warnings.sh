#!/bin/bash
echo on
echo
echo "Starting NWS Warnings on Cyclone"
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
. source /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/NWS_Warnings.py > /projects/SD_Mines_Map_Wall/__log_warnings.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
