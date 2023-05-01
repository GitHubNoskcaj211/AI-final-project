import numpy as np
from utils import transform_pose, transform_position, OCCUPANCY_FROM_STATE_FACTOR, OCCUPANCY_FROM_STATE, STATE_FROM_OCCUPANCY

MINIMUM_ROAD_COLOR = np.array([99, 99, 99])
MAXIMUM_ROAD_COLOR = np.array([110, 110, 110])
START_CAR_INDEX_II = 66
END_CAR_INDEX_II = 77
START_CAR_INDEX_JJ = 45
END_CAR_INDEX_JJ = 51

def compute_road_grid(screen):
    valid_road_pixels = np.full((screen.shape[0], screen.shape[1]), False, dtype=np.bool_)
    valid_road_pixels[np.all(screen >= MINIMUM_ROAD_COLOR, axis=2) & np.all(screen <= MAXIMUM_ROAD_COLOR, axis=2)] = True
    return valid_road_pixels

def compute_speed(hull):
    return (hull.linearVelocity[0] ** 2 + hull.linearVelocity[1] ** 2) ** 0.5

def compute_angular_velocity(hull):
    return hull.angularVelocity

def compute_wheel_angle(car):
    return car.wheels[0].joint.angle

def compute_wheel_angular_velocities(car):
    return [w.omega for w in car.wheels]

def compute_road_middle(road):
    occupancy_car_position = transform_position(OCCUPANCY_FROM_STATE, np.array([0, 0]))
    road_segment = road[int(occupancy_car_position[0]),:]
    road_segment_position = np.where(road_segment == True)
    if road_segment_position[0].shape[0] == 0:
        return (71, 47)
    return np.array([occupancy_car_position[0], np.mean(road_segment_position)])

def compute_road_mean(road):
    road_positions = np.where(road == True)
    if road_positions[0].shape[0] == 0:
        return (50, 47)
    mean_road = np.mean(road_positions, axis=1)
    return mean_road

def compute_target_pixel_far(road):
    occupancy_car_position = transform_position(OCCUPANCY_FROM_STATE, np.array([0, 0]))
    road_positions = np.transpose(np.where(road == True))
    differences = road_positions - occupancy_car_position
    distances = differences[:,0] ** 2 + differences[:,1] ** 2
    if distances.shape[0] == 0:
        return (50, 2) # Goal to the left when can't find road
    farthest_road_indices = np.argsort(distances)[-min(50, distances.shape[0]):]
    if distances[farthest_road_indices[0]] < 70:
        return (50, 2) # Goal to the left when can't find road
    farthest_road_position = np.mean(road_positions[farthest_road_indices,:], axis=0)
    return farthest_road_position

class Perception:
    def __init__(self):
        self.road = None
        self.car_angular_velocity = None
        self.car_speed = None

    # This will give perception of the current system.
    def compute_perception(self, state):
        screen, car = state
        self.road = compute_road_grid(screen)
        self.car_angular_velocity = compute_angular_velocity(car.hull)
        self.car_wheel_angle = compute_wheel_angle(car)
        self.car_speed = compute_speed(car.hull)
        self.target_pixel_far = compute_target_pixel_far(self.road)
        self.road_mean = compute_road_mean(self.road)
        self.road_middle = compute_road_middle(self.road)
        self.wheel_angular_velocities = compute_wheel_angular_velocities(car)