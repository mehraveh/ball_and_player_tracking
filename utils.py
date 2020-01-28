
import numpy as np
from scipy.spatial import distance

def detect_owner(blue_player_coordinates, red_player_coordinates, ball_coordinate, prev_distance, cor, prev_owner):
	min_dis = 100000
	min_coordinate = (0,0)
	owner = 1
	for coordinate in blue_player_coordinates:
		dist = distance.euclidean(coordinate, ball_coordinate)
		if dist < min_dis:
			min_dis = dist
			min_coordinate = coordinate
			owner = 1

	for coordinate in red_player_coordinates:
		dist = distance.euclidean(coordinate, ball_coordinate)
		if dist < min_dis:
			min_dis = dist
			min_coordinate = coordinate
			owner = 2
	if min_dis < 45:
		return min_dis, min_coordinate, owner, True
	else:
		return prev_distance, cor, prev_owner, False



