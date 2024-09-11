########## Libraries ##########
import numpy as np

########## Functions ##########

def EuclidianDistance(node_A , node_B):
    """
    EuclidianDistance (function) 
        Input: Two points (coordenates x, y) that
        represents two nodes (locations) in TSP.
        Output: Euclidian distance.
        Description: Calculates euclidian distance
        between two coordenates.
    """
    return np.sqrt((node_B[0] - node_A[1])**2 
                       + (node_B[1]-node_A[1])**2)

def ReadTsp(filename):
    """
    ReadTsp (function)
        Input: File name.
        Output: Distance Matrix.
        Description: Read a TSP instance (.tsp file)
        and creates discance matrix.
    """
    # Input.
    infile = open(filename,'r')

    # Read instance first line.
    Name = infile.readline().strip().split()[2]

    # Skip comments until DIMENSION.
    for line in infile:
        # Reading line.
        line = infile.readline().strip().split()

        # Verification of DIMENSION.
        if(line[0] == 'DIMENSION'):
            Dimension = line[2]

        # Verification of EOF or Node coordenates secction.
        elif(line[0] == 'EOF' or line[0] == 'NODE_COORD_SECTION'):
            break

    # Tomar coordenada.
    nodelist = []
    int_dim = int(Dimension)
    for i in range(0, int_dim):
        x,y = infile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])
    
    print(nodelist[-1])