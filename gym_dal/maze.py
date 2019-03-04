import itertools
import numpy as np
import numpy.random as npr

def generate_maze(maze_size, decimation=0.):
    """
    Generates a maze using Kruskal's algorithm
    https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    m = (maze_size - 1)//2
    n = (maze_size - 1)//2
    maze = np.ones((maze_size, maze_size))
    for i, j in list(itertools.product(range(m), range(n))):
        maze[2*i+1, 2*j+1] = 0
    m = m - 1
    L = np.arange(n+1)
    R = np.arange(n)
    L[n] = n-1

    while m > 0:
        # plt.imshow(maze)
        # plt.pause(0.1)

        for i in range(n):
            j = L[i+1]
            if (i != j and npr.randint(3) != 0):
                R[j] = R[i]
                L[R[j]] = j
                R[i] = i + 1
                L[R[i]] = i
                maze[2*(n-m)-1, 2*i+2] = 0
                #plt.imshow(maze)
            if (i != L[i] and npr.randint(3) != 0):
                L[R[i]] = L[i]
                R[L[i]] = R[i]
                L[i] = i
                R[i] = i
            else:
                maze[2*(n-m), 2*i+1] = 0
                
        m -= 1

    for i in range(n):
        j = L[i+1]
        if (i != j and (i == L[i] or npr.randint(3) != 0)):
            R[j] = R[i]
            L[R[j]] = j
            R[i] = i+1
            L[R[i]] = i
            maze[2*(n-m)-1, 2*i+2] = 0
            
        L[R[i]] = L[i]
        R[L[i]] = R[i]
        L[i] = i
        R[i] = i

    return maze

def generate_map(map_size, num_cells_togo, save_boundary=True, min_blocks = 10):
    
    maze=generate_maze(map_size)

    if save_boundary:
        maze = maze[1:-1, 1:-1]
        map_size -= 2

    index_ones = np.arange(map_size*map_size)[maze.flatten()==1]

    reserve = min(index_ones.size, min_blocks)    
    num_cells_togo = min(num_cells_togo, index_ones.size-reserve)

    if num_cells_togo > 0:
        blocks_remove=npr.choice(index_ones, num_cells_togo, replace = False)
        maze[blocks_remove//map_size, blocks_remove%map_size] = 0

    if save_boundary:
        map_size+=2
        maze2 = np.ones((map_size,map_size))
        maze2[1:-1,1:-1] = maze
        return maze2
    else:
        return maze

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("num_grid", type=int)
    parser.add_argument("num_rm", type=int)

    args = parser.parse_args()
    
    m = generate_map(args.num_grid, args.num_rm, save_boundary = False)

    import matplotlib.pyplot as plt
    plt.imshow(m)
    plt.xticks(range(args.num_grid))
    plt.yticks(range(args.num_grid))
    plt.grid()
    plt.show()

