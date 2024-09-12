########## Libraries ##########
import numpy as np

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

def best_neighbor(Neighborhood, DistanceMatrix):
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
            Best = Candidate
    
    return Best


def TabuSearch():
    print("Hi")