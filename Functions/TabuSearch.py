########## Libraries ##########
import numpy as np

########## Functions ##########

def ObjFun(Solution, DistanceMatrix):
    """
    ObjFun (function)
        Input: Permutation vector and
        distance matrix.
        Output: Value in objective function.
        Description: Calculates the objetctive
        function using permutation vector and
        distance matrix.
    """
    # Initialization of length of vector and
    # sum.
    tamVector = len(Solution)
    sum = 0

    # Sum connetions between nodes.
    for i in range(-1,tamVector-1):
        sum = sum + DistanceMatrix[Solution[i]-1][Solution[i+1]-1]
    
    return sum


def first_solution(AmountNodes):
    """
    first_solution (function)
        Input: Number of nodes.
        Output: Permutation vector.
        Description: Generates first solution.
    """
    Vector = np.random.permutation(np.arange(1, AmountNodes))
    return Vector

def get_neighbors(Solution, DesireNum):
    """
    get_neighbors (function)
        Input: Reference Solution and DesireNum.
        Output: List of neighbor solutions.
        Description: Generates a 
        neighborhood of solutions.
    """
    # Generates solutions.
    neighbors = []
    for _ in range(DesireNum):
        # Copy input's solution.
        neighbor = np.copy(Solution)

        # Swaping.
        i, j = np.random.choice(len(Solution), 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Add neighbor.
        neighbors.append(neighbor.tolist())

    return neighbors

def best_neighbor(Neighborhood, DistanceMatrix, TabuList):
    """
    get_neighbors (function)
        Input: List of neighbor solutions.
        Output: neighbor  with best improvement 
        in objective function.
        Description: Evaluates Neighborhood
        of permutation solutions.
    """
    # Best solution and value in obj. function.
    Best = Neighborhood[0]
    Best_f = ObjFun(Neighborhood[0], DistanceMatrix)

    # Searching.
    for i in range(1,len(Neighborhood)-1):
        # Take candidate.
        Candidate = Neighborhood[i]
        Candidate_f = ObjFun(Neighborhood[i], DistanceMatrix)

        # Conditions for change.
        if (Best_f > Candidate_f):
            if (Candidate not in TabuList):
                Best = Candidate
    
    return Best


def TabuSearch(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10, numDesireSolution=50,
               ErrorTolerance=0.001):
    """
    get_neighbors (function)
        Input: Distance Matrix (TSP instance), Total number of
        nodes, Max iterations for algorithm, size of
        tabu list, number of solutions to generate and
        error tolerance.
        Output: Unique solutions (Best found).
        Description: Implementation of Tabu Search
        metaheuristic.
        Reference: https://www.geeksforgeeks.org/what-is-tabu-search/
    """ 
    # Setting initial variables.   
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    tabu_list = []

    # Until Max Iterations (Stop Criteria)
    for _ in range(MaxIterations):
        # Creating Neighborhood
        Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)
        # Search the best neighbor.
        BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Criteria for diversification.
        BestSolution_f = ObjFun(BestSolution,DistanceMatrix)
        CurrentSolution_f = ObjFun(BestNeighbor,DistanceMatrix)
        # There is no improvement.
        if (best_neighbor is None):
            CurrentSolution = first_solution(AmountNodes)
            Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix,tabu_list)
        # Improvement (%) is not enough.
        elif (abs(BestSolution_f-CurrentSolution_f) < ErrorTolerance):
            CurrentSolution = first_solution(AmountNodes)
            Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix,tabu_list)

        # If there is no improvement (Stop).
        if (BestNeighbor is None):
            break

        # Adding on tabu list and change current solution.
        CurrentSolution = BestNeighbor
        tabu_list.append(CurrentSolution)
        if (len(tabu_list) > TabuSize):
            tabu_list.pop(0)

        # Change best solution(?).
        if (ObjFun(BestNeighbor,DistanceMatrix) <
            ObjFun(BestSolution,DistanceMatrix)):
            BestSolution = BestNeighbor


    return BestSolution

