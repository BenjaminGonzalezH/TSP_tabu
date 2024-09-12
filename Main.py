########## Libraries ##########
import sys
import os
import numpy as np
import optuna

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))
from ReadTSP import ReadTsp # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import first_solution  # type: ignore
from TabuSearch import get_neighbors # type: ignore
from TabuSearch import best_neighbor  # type: ignore
from TabuSearch import TabuSearch  # type: ignore


########## Procedure ##########

# Obtain trainning TSP's instances.
Path_T = "InstanciasTSP/Parametrizacion"
Content = os.listdir(Path_T)
files = []
for file in Content:
    if(os.path.isfile(os.path.join(Path_T,file))):
        files.append(Path_T+"/"+file)

# Defining Parametrization function.
def Parametrization(trial):
    # Define intervals.
    MaxIterations = trial.suggest_int('MaxIterations', 50, 500)
    TabuSize = trial.suggest_int('TabuSize', 5, 50)
    numDesireSolution = trial.suggest_int('numDesireSolution', 10, 100)
    ErrorTolerance = trial.suggest_loguniform('ErrorTolerance', 1e-5, 1e-1)

    total_obj_value = 0
    num_instances = len(files)

    for file in files:
        # Read the TSP instance
        DistanceMatrix = ReadTsp(file)

        # Run Tabu Search with the parameters from Optuna
        best_solution = TabuSearch(DistanceMatrix, len(DistanceMatrix), MaxIterations, 
                                   TabuSize, numDesireSolution, ErrorTolerance)
        
        # Evaluate the quality of the solution
        objective_value = ObjFun(best_solution, DistanceMatrix)

        # Sum the objective values
        total_obj_value += objective_value

    # Return the average objective value across all instances
    return total_obj_value / num_instances


def main():
    # Create a study object
    study = optuna.create_study(direction='minimize')
    
    # Optimize
    study.optimize(Parametrization, n_trials=50)

    # Print the best parameters and the best score
    print('Best parameters:', study.best_params)
    print('Best score:', study.best_value)

    # Optionally, you can re-evaluate with the best parameters on all instances
    best_params = study.best_params
    print("Re-evaluating with best parameters...")
    total_obj_value = 0

    for tsp_file in files:
        DistanceMatrix = ReadTsp(tsp_file)
        best_solution = TabuSearch(DistanceMatrix, len(DistanceMatrix), 
                                   best_params['MaxIterations'], 
                                   best_params['TabuSize'], 
                                   best_params['numDesireSolution'], 
                                   best_params['ErrorTolerance'])
        objective_value = ObjFun(best_solution, DistanceMatrix)
        total_obj_value += objective_value

    avg_obj_value = total_obj_value / len(files)
    print('Average objective value with best parameters:', avg_obj_value)

if __name__ == '__main__':
    main()