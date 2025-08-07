#!/bin/bash
echo on
echo "enable bashrc"
source ~/.bash_profile 
echo
echo "CONDA_PYTHON_EXE = "
echo $CONDA_PYTHON_EXE
echo
echo "CONDA_EN_CODE"
echo $CONDA_PROMPT_MODIFIER 
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
rm -frv /projects/SD_Mines_Map_Wall/temp_sfc_analysis/*
echo
echo  "Firing Things Up!"
echo
.  /home/wjc/.bash_profile  ; ~/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/NWS_Wx_Map.py > /projects/SD_Mines_Map_Wall/__log_nws_map.log 2>&1
#.  /home/wjc/.bash_profile  ; /home/wjc/miniconda3/bin/python /projects/SD_Mines_Map_Wall/Surface_Analysis_Rapid_Refresh.py > /projects/SD_Mines_Map_Wall/__log_nws_map.log 2>&1
echo
echo "Ending Script"
echo
echo "Completed: `date`"
echo
echo off
