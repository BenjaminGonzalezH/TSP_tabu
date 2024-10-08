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

if(os.path.exists("output.txt")):
    solutions = []
    with open("output.txt", "r") as file:
        for line in file:
            # Convert each line (string) to a list of integers
            solution = list(map(int, line.strip().split(',')))
            solutions.append(solution)

    for i in range(len(solutions)):
        result = ObjFun_C(solutions[i],Instances[i])
        print(result)

else:
    Best_SOL = []
    for i in range(len(Instances)):
        result, _ = TabuSearch_C(Instances[i], len(Instances[i]), 
                                    best_params['MaxIterations'], 
                                    best_params['TabuSize'], 
                                    best_params['numDesireSolution'], 
                                    best_params['ErrorTolerance'])
    
        Best_SOL.append(result)

    with open("output.txt", "w") as file:
        for solution in Best_SOL:
            # Convertir el sub-arreglo a una cadena y escribirlo en el archivo
            file.write(" ".join(map(str, solution)) + "\n")