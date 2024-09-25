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
from ReadTSP import ReadTsp # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore

########## Secundary Functions ##########

# Save the study object to a file in txt format (JSON and CSV)
def save_study_txt(study, best_params_file, trials_file):
    """
        save_study_txt (function)
            Input: Study and files that are gonna be allocated
            in the path of this file.
    """
    # Save the best parameters in a readable text file (JSON format)
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the entire study's trials as CSV for detailed analysis
    with open(trials_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial Number", "Value", "MaxIterations", "TabuSize", "numDesireSolution", "ErrorTolerance"])
        for trial in study.trials:
            writer.writerow([trial.number, trial.value, 
                             trial.params['MaxIterations'], 
                             trial.params['TabuSize'], 
                             trial.params['numDesireSolution'], 
                             trial.params['ErrorTolerance']])

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
    
def Parametrization(trial, Instances):
    """
        Parametrization (Function)
            Input: trial (parameters for evaluation) and
            list of matrix (one each instance).
            Output: Best trial of parameters.
            Description: Do the parametrization prodecure with
            every trial.
    """
    # Define intervals.
    MaxIterations = trial.suggest_int('MaxIterations', 50, 500)
    TabuSize = trial.suggest_int('TabuSize', 5, 50)
    numDesireSolution = trial.suggest_int('numDesireSolution', 10, 100)
    ErrorTolerance = trial.suggest_loguniform('ErrorTolerance', 1e-5, 1e-1)

    # Initializate count of obj function calls.
    total_obj_value = 0
    num_instances = len(files)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        best_solution, calls = TabuSearch(Instances[i], len(Instances[i]), MaxIterations, 
                                   TabuSize, numDesireSolution, ErrorTolerance)
        
        # Call global variable and update calls count.
        global obj_func_calls
        obj_func_calls = calls
        print(calls)

        # Evaluate the quality of the solution
        objective_value = ObjFun(best_solution, Instances[i])

        # Sum the objective values
        total_obj_value += objective_value

    # Return the average objective value across all instances
    return total_obj_value / num_instances

def Parametrization_capsule(Instances):
    """
        Parametrization_capsule (Function)
            Encapsulate other inputs from principal
            parametrization function.
    """
    return lambda trial: Parametrization(trial, Instances)

def stop_optimization_callback(study, trial):
    """
        stop_optimization_callback (function)
            Input: Study (class that allocates the procedure
            of parametrization) and trial (lastest).
            Description: function that manage the end of the
            parametrization procedure.
    """
    # Use globals.
    global max_calls_obj_func
    global obj_func_calls

    # Stop critetion.
    if obj_func_calls >= max_calls_obj_func:
        print(f"Límite de llamadas a la función objetivo alcanzado: {max_calls_obj_func}. Deteniendo la optimización.")
        raise optuna.exceptions.OptunaError("Límite de llamadas a la función objetivo alcanzado.")

########## Procedure ##########

# Obtain trainning TSP's instances.
Path_T = "InstanciasTSP/Parametrizacion"
Content = os.listdir(Path_T)
files = []
for file in Content:
    if(os.path.isfile(os.path.join(Path_T,file))):
        files.append(Path_T+"/"+file)


def main():
    # Load instances.
    Instances = []
    for tsp_file in files:
        DistanceMatrix = ReadTsp(tsp_file)
        Instances.append(DistanceMatrix)
    num_instances = len(files)

    # Create or load a study
    study_file = 'optuna_study.txt'
    trials_file = 'optuna_trials.csv'
    
    # files check.
    if os.path.exists(study_file) and os.path.exists(trials_file):
        # The files are there.
        print("Loading existing study...")
        best_params = load_best_params(study_file)
        print(best_params)
    else:
        # New study.
        print("Creating a new study...")
        study = optuna.create_study(direction='minimize')

        # Optimize
        try:
            study.optimize(Parametrization_capsule(Instances), n_trials=10, callbacks=[stop_optimization_callback])
        except Exception as e:
            print(f"Error during optimization: {e}")

        # Save the study after optimization
        save_study_txt(study, study_file, trials_file)

        # After the optimization, assign the best_params
        best_params = study.best_params
        print(best_params)

    # Print the best parameters and the best score
    print('Best parameters:', best_params)

    # Optionally, you can re-evaluate with the best parameters on all instances
    print("Re-evaluating with best parameters...")
    total_obj_value = 0

    for i in range(num_instances):
        best_solution, _ = TabuSearch(Instances[i], len(Instances[i]), 
                                   best_params['MaxIterations'], 
                                   best_params['TabuSize'], 
                                   best_params['numDesireSolution'], 
                                   best_params['ErrorTolerance'])
        objective_value = ObjFun(best_solution, Instances[i])
        total_obj_value += objective_value

    avg_obj_value = total_obj_value / len(files)
    print('Average objective value with best parameters:', avg_obj_value)

if __name__ == '__main__':
    main()