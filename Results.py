########## Libraries ##########
import sys
import os
import numpy as np
import optuna
import json
import csv

########## Global ##########
"""
    max_calls_obj_func (global variable)
        Minimum of calls for end parametrization.
    obj_func_calls (global variable)
        Counter of every time that
        the objective function is called.
"""
max_calls_obj_func = 300000
obj_func_calls = 0

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))
from ReadTSP import ReadTsp_Coordenates # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore

########## Secundary Functions ##########

########## Procedure ##########

# Obtain files.
Path_T = "InstanciasTSP/Experimental"
Content = os.listdir(Path_T)
files = []
for file in Content:
    if(os.path.isfile(os.path.join(Path_T,file))):
        files.append(Path_T+"/"+file)

    
Instances = []
for tsp_file in files:
    ReadTsp_Coordenates(tsp_file)
    Instances.append(ReadTsp_Coordenates)

num_instances = len(files)