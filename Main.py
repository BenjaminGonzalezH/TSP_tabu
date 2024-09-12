########## Libraries ##########
import sys
import os
import numpy as np

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

""" # Obtain trainning TSP's instances.
Path_T = "InstanciasTSP/Parametrizacion"
Content = os.listdir(Path_T)
files = []
for file in Content:
    if(os.path.isfile(os.path.join(Path_T,file))):
        files.append(Path_T+"/"+file)

# Reading TSP instances.
for filename in files:
    ReadTsp(filename)
"""

DistanceMatrix = np.random.rand(10, 10)
DistanceMatrix = (DistanceMatrix + DistanceMatrix.T) / 2
np.fill_diagonal(DistanceMatrix, 0)

Vector = first_solution(10)
Best = TabuSearch(DistanceMatrix,len(Vector),MaxIterations=100, TabuSize=10, numDesireSolution=50,
               ErrorTolerance=0.001)
print(ObjFun(Best,DistanceMatrix))