#!/bin/bash
echo on
echo
echo "Starting NWS Surface Analysis on Kyrill"
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
.  /home/wjc/.bashrc ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/NWS_Wx_Map.py > /projects/SD_Mines_Map_Wall/__log_nws_map.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
