########## Libraries ##########
import sys
import os
import numpy as np
import optuna
import json
import csv

########## Inputs ##########
"""
    max_calls_obj_func
        Minimum of calls for end parametrization.
    
    obj_func_calls
        Counter of every time that
        the objective function is called.
    
    study_file
        File with bests params
"""
max_calls_obj_func = 1000000
obj_func_calls = 0
study_file = 'optuna_study.txt'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))
from ReadTSP import ReadTsp_Coordenates # type: ignore
from TabuSearch import first_solution  # type: ignore
from TabuSearch import get_neighbors  # type: ignore
from TabuSearch import best_neighbor_C  # type: ignore
from TabuSearch import ObjFun_C  # type: ignore
from TabuSearch import TabuSearch_C  # type: ignore

########## Secundary Functions ##########
def load_best_params(filename):
    """
        load_best_params (functions)
            Input: File path.
            Description: Load parameters from a JSON format
            text file.
    """
    # Open text file, read it as JSON file
    # and allocates the info in a dicctionary.
    with open(filename, 'r') as f:
        best_params = json.load(f)
    return best_params

########## Procedure ##########

# Obtain files.
Path_T = "InstanciasTSP/Experimental"
Content = os.listdir(Path_T)
files = []
for file in Content:
    if(os.path.isfile(os.path.join(Path_T,file))):
        files.append(Path_T+"/"+file)

# Generate instances.
Instances = []
for tsp_file in files:
    Coordenates = ReadTsp_Coordenates(tsp_file)
    Instances.append(Coordenates)

# files check.
if os.path.exists(study_file):
    # The files are there.
    print("Loading existing study...")
    best_params = load_best_params(study_file)
    print(best_params)
else:
    print("The file {study_file} not found".format(study_file))


Best_SOL = []
for i in range(len(files)):
    random = np.random.permutation(np.arange(1, len(Instances[i])))
    Best_SOL.append(random)

while(True):
    for i in range(len(Instances)):
        result, calls = TabuSearch_C(Instances[i], len(Instances[i]), 
                                   best_params['MaxIterations'], 
                                   best_params['TabuSize'], 
                                   best_params['numDesireSolution'], 
                                   best_params['ErrorTolerance'])
        
        current_best = ObjFun_C(Best_SOL[i], Instances[i])
        opponent = ObjFun_C(result, Instances[i])
        

        if(current_best > opponent):
            Best_SOL[i] = opponent

        obj_func_calls = calls
        print(obj_func_calls)
        if(obj_func_calls > max_calls_obj_func):
            flag = 1
            break

    if(flag == 1):
        break

print(Best_SOL)