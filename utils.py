import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

OCCUPANCY_FROM_STATE_FACTOR = 2
OCCUPANCY_FROM_STATE = np.array([[-OCCUPANCY_FROM_STATE_FACTOR, 0, 71.5], [0, -OCCUPANCY_FROM_STATE_FACTOR, 48], [0, 0, 1]])
STATE_FROM_OCCUPANCY = np.linalg.inv(OCCUPANCY_FROM_STATE)

def transform_position(matrix, position):
    return (matrix @ np.array([[position[0]], [position[1]], [1]]))[0:2, 0]

def transform_pose(matrix, pose):
    matrix_pose = np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]], [np.sin(pose[2]), np.cos(pose[2]), pose[1]], [0, 0, 1]])
    transformed_mat = matrix @ matrix_pose
    expanded_rot = np.array([[transformed_mat[0,0], transformed_mat[0,1], 0], [transformed_mat[1,0], transformed_mat[1,1], 0], [0, 0, 1]])
    r = R.from_matrix(expanded_rot)
    return np.array([transformed_mat[0,2], transformed_mat[1,2], r.as_euler('zyx')[0]])

def get_state_position_surface(position, color=(255, 255, 255, 255)):
    occupancy_position = transform_position(OCCUPANCY_FROM_STATE, position)
    ii = int(occupancy_position[0] * 800 / 96)
    jj = int(occupancy_position[1] * 1000 / 96)
    if ii < 0 or ii >= 800 or jj < 0 or jj >= 1000:
        print('out of bounds!')
        return None
    surf = pygame.Surface((1000, 800), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, color, pygame.Rect(jj - 3, ii - 3, 6, 6), 4)
    # surf = pygame.transform.flip(surf, False, True)
    return surf

def get_state_pose_surface(pose, color=(255, 255, 255, 125)):
    occupancy_pose = transform_pose(OCCUPANCY_FROM_STATE, pose)
    ii = int(occupancy_pose[0] * 800 / 96)
    jj = int(occupancy_pose[1] * 1000 / 96)
    if ii < 0 or ii >= 800 or jj < 0 or jj >= 1000:
        print('out of bounds!')
        return None
    surf = pygame.Surface((1000, 800), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.line(surf, color, (jj, ii), (jj + 10*np.sin(occupancy_pose[2]), ii + 10*np.cos(occupancy_pose[2])), 4)
    # surf = pygame.transform.flip(surf, False, True)
    return surf