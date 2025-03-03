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
echo "Starting NAM 3-Panel on Cyclone"
echo "Date:  `date`"
echo
echo
DATESTRING=`date +"%Y-%m-%d_%H%M"`
echo
echo
echo "Entering Working Directory"
echo
cd /projects/SD_Mines_Map_Wall
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_nam/*.png
echo
echo  "Firing Things Up!"
#
rm -frv  /projects/SD_Mines_Map_Wall/temp_files_nam/*.png
#
. source /home/wjc/.bashrc ; $CONDA_PYTHON_EXE /projects/SD_Mines_Map_Wall/NAM_4_Panel.py > /projects/SD_Mines_Map_Wall/__log_nam4.log 2>&1


echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
