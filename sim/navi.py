import numpy as np
from recordtype import recordtype
from utils import *
import pdb

Cell = recordtype("Cell", "x y")


def generate_four_maps(map_high, grid_rows, grid_cols):
    I = grid_rows
    J = map_high.shape[0]
    
    east_map = np.zeros((grid_rows, grid_cols),np.float)
    south_map = np.zeros((grid_rows, grid_cols),np.float)
    west_map = np.zeros((grid_rows, grid_cols),np.float)
    north_map = np.zeros((grid_rows, grid_cols),np.float)
    
    # East map. left to right
    for i in range(grid_rows):
        for j in range(grid_cols):
            if j == 0:
                east_map[i,j] = 1
            else:
                center_x, center_y = grid_cell_to_map_cell(i,j,I,J)
                prev_center_x, prev_center_y = grid_cell_to_map_cell(i, j-1,I,J)
                if prev_center_x != center_x:
                    print("wtf bro!")
                for btc in range(prev_center_y, center_y):
                    if map_high[center_x, btc] == 1:
                        east_map[i, j] = 1
                        break

    # West map. right to left
    for i in range(grid_rows):
        for j in range(grid_cols):
            if j == grid_cols - 1:
                west_map[i,j] = 1
            else:
                center_x, center_y = grid_cell_to_map_cell(i,j,I,J)
                prev_center_x, prev_center_y = grid_cell_to_map_cell(i, j+1,I,J)
                if prev_center_x != center_x:
                    print("wtf bro!")
                for btc in range(prev_center_y, center_y, -1):
                    if map_high[center_x, btc] == 1:
                        west_map[i, j] = 1
                        break

    # South map. top to bottom
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i == 0:
                south_map[i,j] = 1
            else:
                center_x, center_y = grid_cell_to_map_cell(i,j,I,J)
                prev_center_x, prev_center_y = grid_cell_to_map_cell(i-1, j,I,J)
                if prev_center_y != center_y:
                    print("wtf bro!")
                for btc in range(prev_center_x, center_x):
                    if map_high[btc, center_y] == 1:
                        south_map[i, j] = 1
                        break

    # North map. top to botatom
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i == grid_rows - 1:
                north_map[i,j] = 1
            else:
                center_x, center_y = grid_cell_to_map_cell(i,j,I,J)
                prev_center_x, prev_center_y = grid_cell_to_map_cell(i+1, j,I,J)
                if prev_center_y != center_y:
                    print("wtf bro!")
                for btc in range(prev_center_x, center_x, -1):
                    if map_high[btc, center_y] == 1:
                        north_map[i, j] = 1
                        break

    return north_map, east_map, south_map, west_map



def compute_shortest(north_map, east_map, south_map, west_map, current, target, gridsize):
    #gridsize=11
    shortest_map = np.ones((gridsize,gridsize))*100
    shortest_map[target.x, target.y] = 0

    for trump in range(gridsize*2): #largest manhattan distance
        # print("************************************")
        # print(trump)
        # print("************************************")
        # print(shortest_map)
        # print(current.x, current.y)
        for i in range(gridsize):
            for j in range(gridsize):
                if shortest_map[current.x, current.y] != 100:
                    break
                if east_map[i,j] == 0:
                    shortest_map[i,j] = min(shortest_map[i,j], shortest_map[i,j-1]+1)
                if west_map[i,j] == 0:
                    shortest_map[i,j] = min(shortest_map[i,j], shortest_map[i,j+1]+1)
                if south_map[i,j] == 0:
                    shortest_map[i,j] = min(shortest_map[i,j], shortest_map[i-1,j]+1)
                if north_map[i,j] == 0:
                    shortest_map[i,j] = min(shortest_map[i,j], shortest_map[i+1,j]+1)
            if shortest_map[current.x, current.y] != 100:
                break
        if shortest_map[current.x, current.y] != 100:
            return shortest_map
    return shortest_map


def give_me_path(shortest_map, source, target, gridsize):
    grid_list = []
    current_coords = Cell(source.x, source.y)
    grid_list.append(current_coords)

    while current_coords.x != target.x or current_coords.y != target.y:
        if shortest_map[current_coords.x, current_coords.y] == 100.0:
            break
        minn = shortest_map[current_coords.x, current_coords.y]
        if current_coords.x > 0:
            if shortest_map[current_coords.x - 1, current_coords.y] <= minn:
                minn = shortest_map[current_coords.x - 1, current_coords.y]
                note = Cell(current_coords.x - 1, current_coords.y)
        if current_coords.x < gridsize-1:
            if shortest_map[current_coords.x + 1, current_coords.y] <= minn:
                minn = shortest_map[current_coords.x + 1, current_coords.y]
                note = Cell(current_coords.x + 1, current_coords.y)
        if current_coords.y > 0:
            if shortest_map[current_coords.x, current_coords.y - 1] <= minn:
                minn = shortest_map[current_coords.x, current_coords.y -1]
                note = Cell(current_coords.x, current_coords.y -1)
        if current_coords.y < gridsize-1:
            if shortest_map[current_coords.x, current_coords.y + 1] <= minn:
                minn = shortest_map[current_coords.x, current_coords.y + 1]
                note = Cell(current_coords.x, current_coords.y + 1)
        current_coords = note
        grid_list.append(current_coords)
    grid_list.append(target)
    # print ('grid_list', grid_list)
    return grid_list


def give_me_actions(grid_list, curr_dir):
    actions = []
    # curr_dir
    # prev_pos = grid_list[0]
    # print('len(grid_list)',len(grid_list))
    for i in range(len(grid_list)-1):
        prev_pos = grid_list[i]
        next_pos = grid_list[i+1]
        if next_pos.x > prev_pos.x:
            next_dir = 2
        elif next_pos.x < prev_pos.x:
            next_dir = 0
        elif next_pos.y > prev_pos.y:
            next_dir = 3
        elif next_pos.y < prev_pos.y:
            next_dir = 1
        else:
            next_dir = -1
        
        # print('next dir, curr dir', next_dir, curr_dir)
        if next_dir == -1:
            actions.append(3)
        elif abs(next_dir - curr_dir) == 0: 
            actions.append(2) # go straight
        elif abs(next_dir - curr_dir) == 2:
            actions.append(0) # turn left
            actions.append(0) # turn left
            actions.append(2) # go straight
        elif next_dir-curr_dir == 1 or next_dir-curr_dir == -3:
            #turn left and go straight
            actions.append(0)
            actions.append(2)
        elif next_dir-curr_dir == -1 or next_dir-curr_dir == 3:
            # turn right and go straight
            actions.append(1)
            actions.append(2)
        else:
            print("you are drunk bro")
            
        curr_dir = next_dir
        prev_pos = next_pos
    return actions




if __name__ == '__main__':

    # for i in range(11):
    #   for j in range(11):
    #       I,J = grid_cell_to_map_cell(i,j,11,224)
    #       print(i,j, I, J)

    full_map = np.load('./maps/montreal_map_inv.npy')
    #full_map = (255 - full_map)/255.0
    # print(full_map)
    north_map, east_map, south_map, west_map = generate_four_maps(full_map, 11, 11)
    # print(north_map, east_map, south_map, west_map)
    current = Cell(2,2)
    target = Cell(3,3)
    print ('current',current)
    print('target',target)
    shortest_map = compute_shortest(north_map, east_map, south_map, west_map, current, target)
    print('shortest', shortest_map)
    grid_list = give_me_path(shortest_map, current, target)
    print('grid list',grid_list)
    actions = give_me_actions(grid_list, 0)
    print('actions',actions)
