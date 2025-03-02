#!/bin/bash
echo on
echo "enable bashrc"
source ~/.bashrc
echo
echo "CONDA_PYTHON_EXE = "
echo $CONDA_PYTHON_EXE
echo
echo "CONDA_EN_CODE"
echo $CONDA_PROMPT_MODIFIER 
echo
echo "Starting GFS 3-Panel on Cyclone"
echo "Date: `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_gfs/*.png
echo
echo  "Firing Things Up!"
#
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_gfs/*.png
#
.  /home/wjc/.bashrc ; $CONDA_PYTHON_EXE /projects/SD_Mines_Map_Wall/GFS_4_Panel_20km.py > /projects/SD_Mines_Map_Wall/__log_gfs4.log 2>&1


echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
