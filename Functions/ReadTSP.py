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
                       + (node_B[0]-node_A[1])**2)

def EuclideanDistanceMatrix(NodeList, Tam_Nodes):
    """
    EuclideanDistanceMatrix (function) 
        Input: List with coordenates of
        each point (location) and tam of the
        list.
        Output: Euclidean Distance matrix for all pairs.
        Description: Calculates Euclidean distance
        Matrix for each pairs of locations.
    """
    Matrix = np.zeros((Tam_Nodes,Tam_Nodes))

    for i in range(0, Tam_Nodes):
        for j in range(i+1, Tam_Nodes):
            Matrix[i][j] = (EuclidianDistance(NodeList[i], NodeList[j]))
            Matrix[j][i] = Matrix[i][j]

    return Matrix


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

    # Take coordenates.
    nodelist = []
    int_dim = int(Dimension)
    for i in range(0, int_dim):
        x,y = infile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])
    
    # Close input file.
    infile.close()
    
    # Create distance matrix.
    DistanceMatrix = EuclideanDistanceMatrix(nodelist,int_dim)
    print(Name)
    return DistanceMatrix

    
    