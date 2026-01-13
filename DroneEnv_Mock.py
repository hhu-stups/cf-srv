import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import numpy as np
import sys
import os
import socket
import json

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


DEPTH = 5
WIDTH = 5
NR_OBSTACLES = 4
MAX_BATTERY = 100
BATTERY_COST_STEP = 2
MAX_EPISODE_STEPS = 1000
SAVE_PATH = "models/drone"

class DroneEnv(gym.Env):

    def __init__(self, requires_external_values):
        super(DroneEnv, self).__init__()
        self.requires_external_values = requires_external_values
        self.reset()
        self.action_space = spaces.Discrete(5)  # 0: Forward, 1: Backward, 2: Left, 3: Right, 4: Land


    def reset(self, seed=None, options=None):
        super().reset()
        self.base_x = WIDTH // 2 - 1
        self.base_y = DEPTH // 2 - 1
        self.x = self.base_x
        self.y = self.base_y
        self.battery = MAX_BATTERY
        self.flying = True
        self.mission_done = False
        self.done = False
        self.last_positions = [(self.x, self.y), (self.x, self.y), (self.x, self.y), (self.x, self.y), (self.x, self.y), (self.x, self.y), (self.x, self.y)]

        all_fields = [(x, y) for x in range(WIDTH) for y in range(DEPTH)]
        all_possible_fields = set(all_fields) - set([(self.x, self.y)])

        # wahrscheinlich ganzes Feld notwendig
        # kürzester Pfad zur Basisstation
        self.observation_space = spaces.Box(
            low=np.array([
                0, # mission done
                0, # flying status
                0, # current x coordinate
                0, # current y coordinate
                0, # current battery
                0, # distance to base
                #-WIDTH, # base x coordinate
                #-DEPTH, # base y coordinate
                0, # shortest_distance

                -1, # delta up
                -1, # delta right
                -1, # delta down
                -1, # delta left,

                -1, # delta goal up
                -1, # delta goal right
                -1, # delta goal down
                -1, # delta goal left
                #0, # shortest x distance to unexplored
                #0, # shortest y distance to unexplored

                0, 0, 0, 0, # free statuses of surrounding fields: Up, Right, Down, Left
                0, 0, 0, 0, # obstacle statuses of surrounding fields: Up, Right, Down, Left
                0, 0, 0, 0 # visited statuses of surrounding fields: Up, Right, Down, Left
                ], dtype=np.float32),
            high=np.array([
                1, # mission done
                1, # flying status
                WIDTH, # current x coordinate
                DEPTH, # current y coordinate
                MAX_BATTERY, # current battery
                DEPTH*WIDTH, # distance to base
                #WIDTH, # base x coordinate
                #DEPTH, # base y coordinate,
                DEPTH * WIDTH, # shortest distance to unexplored

                1, # delta up
                1, # delta right
                1, # delta down
                1, # delta left

                1, # delta goal up
                1, # delta goal right
                1, # delta goal down
                1, # delta goal left

                #WIDTH, # shortest x distance to unexplored
                #DEPTH, # shortest y distance to unexplored
                1, 1, 1, 1, # free statuses of surrounding fields: Up, Right, Down, Left
                1, 1, 1, 1, # obstacle statuses of surrounding fields: Up, Right, Down, Left
                1, 1, 1, 1 # visited statuses of surrounding fields: Up, Right, Down, Left
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.grid = np.ones((WIDTH, DEPTH), dtype=int)
        if not self.requires_external_values:
            self.obstacles = set(random.sample(list(all_possible_fields), NR_OBSTACLES))
            all_possible_fields = set(all_possible_fields) - self.obstacles
            for o in self.obstacles:
                self.grid[o[0], o[1]] = 2 #2 for obstacles

        self.visited = set()
        self.visited.add((self.x, self.y))
        self.explored = set()
        self.explore()

        return self._get_obs(), {}

    def explore(self):
        self.explored.add((self.x, self.y))
        for i in [-1,0,1]:
            new_x = self.x + i
            for j in [-1,0,1]:
                new_y = self.y + j
                if (i == 0 or j == 0) and new_x >= 0 and new_x < WIDTH and new_y >= 0 and new_y < DEPTH:
                    self.explored.add((new_x, new_y))

    def _get_obs(self):
        surrounding_fields = self._get_surrounding_fields()
        surrounding_fields_visited = self._get_surrounding_fields_visited()
        base_x, base_y = self.base_x, self.base_y

        distance_x_to_base = abs(self.x - base_x)
        distance_y_to_base = abs(self.y - base_y)
        distance_to_base = DEPTH * WIDTH

        previous_x = self.x
        previous_y = self.y

        if self.x >= 0 and self.x < WIDTH and self.y >= 0 and self.y < DEPTH:
            field_as_array = self.get_field_as_array()
            walkable_field = [[0 if entry == 2 or entry == 3 else 1 for entry in row] for row in field_as_array]
            grid = Grid(matrix = walkable_field)
            start_pos = grid.node(self.x, self.y)
            base_pos   = grid.node(base_x, base_y)
            finder = AStarFinder(diagonal_movement = DiagonalMovement.never)
            path, runs = finder.find_path(start_pos, base_pos, grid)
            distance_to_base = len(path) - 1 if len(path) > 0 else 0
            if len(path) > 1:
                previous_x = path[1].x
                previous_y = path[1].y

        delta_x_goal, delta_y_goal, estimated_distance = self.estimate_shortest_distance()

        delta_x_base_back = (self.x - previous_x)
        delta_y_base_back = (self.y - previous_y)


        return np.array([
            self.mission_done,
            self.flying,
            self.x, self.y,
            0 if (distance_x_to_base + distance_y_to_base < 1.0) else self.battery/(distance_x_to_base + distance_y_to_base),
            distance_to_base,
            #distance_x_to_base, distance_y_to_base, # Relative distance to base
            estimated_distance,

            1 if delta_x_base_back == 0 and delta_y_base_back == -1 else 0,
            1 if delta_x_base_back == 1 and delta_y_base_back == 0 else 0,
            1 if delta_x_base_back == 0 and delta_y_base_back == 1 else 0,
            1 if delta_x_base_back == -1 and delta_y_base_back == 0 else 0,

            1 if delta_x_goal == 0 and delta_y_goal == -1 else 0,
            1 if delta_x_goal == 1 and delta_y_goal == 0 else 0,
            1 if delta_x_goal == 0 and delta_y_goal == 1 else 0,
            1 if delta_x_goal == -1 and delta_y_goal == 0 else 0,

            1 if surrounding_fields[0] == 1 else 0, 1 if surrounding_fields[1] == 1 else 0, 1 if surrounding_fields[2] == 1 else 0, 1 if surrounding_fields[3] == 1 else 0,
            1 if surrounding_fields[0] == 2 else 0, 1 if surrounding_fields[1] == 2 else 0, 1 if surrounding_fields[2] == 2 else 0, 1 if surrounding_fields[3] == 2 else 0,
            surrounding_fields_visited[0], surrounding_fields_visited[1], surrounding_fields_visited[2], surrounding_fields_visited[3]]
        )

    def _safe_get_field(self, i, j):
        if i >= 0 and i < WIDTH and j >= 0 and j < DEPTH:
            return self.grid[i,j]
        return 2 #2 for obstacles

    def _get_surrounding_fields(self):
        backward = self._safe_get_field(self.x, self.y-1)
        left = self._safe_get_field(self.x+1, self.y)
        forward = self._safe_get_field(self.x, self.y+1)
        right = self._safe_get_field(self.x-1, self.y)
        return forward,right,backward,left

    def _get_surrounding_fields_visited(self):
        backward = 1 if (self.x, self.y-1) in self.visited else 0
        left = 1 if (self.x+1, self.y) in self.visited else 0
        forward = 1 if (self.x, self.y+1) in self.visited else 0
        right = 1 if (self.x-1, self.y) in self.visited else 0
        return forward,right,backward,left

    def move_backward(self):
        self.y = self.y - 1

    def move_forward(self):
        self.y = self.y + 1

    def move_right(self):
        self.x = self.x - 1

    def move_left(self):
        self.x = self.x + 1

    def land(self):
        self.flying = False

    def outside_field(self):
        return self.x < 0 or self.x > WIDTH - 1 or self.y < 0 or self.y > DEPTH - 1

    def mission_accomplished(self):
        distance_x_to_base = abs(self.x - self.base_x)
        distance_y_to_base = abs(self.y - self.base_y)
        distance_to_base = distance_x_to_base + distance_y_to_base
        return (self.mission_done == True) and (distance_to_base < 1.0) and (self.flying == False)

    def mission_failed_safe_at_base(self):
        distance_x_to_base = abs(self.x - self.base_x)
        distance_y_to_base = abs(self.y - self.base_y)
        distance_to_base = distance_x_to_base + distance_y_to_base
        return (self.mission_done == False) and (distance_to_base < 1.0) and (self.flying == False)

    def mission_failed_safe_not_at_base(self):
        distance_x_to_base = abs(self.x - self.base_x)
        distance_y_to_base = abs(self.y - self.base_y)
        distance_to_base = distance_x_to_base + distance_y_to_base
        return (self.mission_done == False) and (distance_to_base > 0.0) and (self.flying == False)

    def is_done(self):
        if self.outside_field():
            return True
        if self.battery == 0:
            return True
        if self.requires_external_values:
            if self.grid[self.x, self.y] == 2:
                return True
        else:
            if (self.x, self.y) in self.obstacles:
                return True
        if self.mission_accomplished():
            return True
        if self.mission_failed_safe_at_base():
            return True
        if self.mission_failed_safe_not_at_base():
            return True
        if self.flying == False:
            return True
        return False

    def reward(self, new_field_visited, number_newly_explored_fields):
        FAIL_SAFE_AT_BASE_REWARD_FACTOR = 1.5
        FAIL_SAFE_OUTSIDE_BASE_PENALTY_FACTOR = -1.5
        MISSION_ACCOMPLISHED_REWARD_FACTOR = 50.0
        MISSION_FAILED_PENALTY_FACTOR = -50.0
        WEIGHT_EXPLORATION_BONUS = 30.0
        DISTANCE_TO_BASE_RELATION_TO_BATTERY_FACTOR = 15.0
        DISTANCE_TO_UNEXPLORED_RELATION_TO_BATTERY_FACTOR = 15.0
        reward = 0.0
        ending = False
        previous_battery = self.battery
        if self.mission_accomplished():
            reward = MISSION_ACCOMPLISHED_REWARD_FACTOR * (DEPTH * WIDTH) # Mission accomplished reward: MISSION_ACCOMPLISHED_REWARD_FACTOR times size of field
            ending = True
        elif self.mission_failed_safe_at_base():
            reward = FAIL_SAFE_AT_BASE_REWARD_FACTOR * (WIDTH * DEPTH) # Fail Safe Reward: linearly scaled to number of explored field
            ending = True
        elif self.mission_failed_safe_not_at_base():
            reward = FAIL_SAFE_OUTSIDE_BASE_PENALTY_FACTOR * (WIDTH * DEPTH) # Fail Safe Reward: linearly scaled to number of explored field
            ending = True
        elif self.outside_field():
            reward = MISSION_FAILED_PENALTY_FACTOR * (DEPTH * WIDTH) # Mission fails Penalty: MISSION_FAILED_PENALTY_FACTOR times size of field
            ending = True
        elif (self.requires_external_values and self.grid[self.x, self.y] == 2) or (not self.requires_external_values and (self.x, self.y) in self.obstacles):
            reward = MISSION_FAILED_PENALTY_FACTOR * (DEPTH * WIDTH) # Mission fails Penalty: MISSION_FAILED_PENALTY_FACTOR times size of field
            ending = True
        elif self.battery == 0:
            reward = MISSION_FAILED_PENALTY_FACTOR * (DEPTH * WIDTH) # Mission fails Penalty: MISSION_FAILED_PENALTY_FACTOR times size of field
            ending = True
        if not ending:


            total_fields = WIDTH * DEPTH
            explored_ratio = len(self.visited) / total_fields
            base_x, base_y = self.base_x, self.base_y

            field_as_array = self.get_field_as_array()
            walkable_field = [[0 if entry == 2 else 1 for entry in row] for row in field_as_array]
            grid = Grid(matrix = walkable_field)
            start_pos = grid.node(self.x, self.y)
            search_pos   = grid.node(base_x, base_y)
            finder = AStarFinder(diagonal_movement = DiagonalMovement.never)
            path, runs = finder.find_path(start_pos, search_pos, grid)
            distance_to_base = len(path)

            if new_field_visited:
                exploration_bonus = WEIGHT_EXPLORATION_BONUS * explored_ratio * number_newly_explored_fields # Reward for exploring new field: WEIGHT_EXPLORATION_BONUS linearly scaled to ratio of explored fields for each newly explored field
                reward = reward + exploration_bonus

            reward = reward - DISTANCE_TO_BASE_RELATION_TO_BATTERY_FACTOR * (distance_to_base/self.battery) # Penalty: distance to base in relation to battery level weighted with DISTANCE_TO_BASE_RELATION_TO_BATTERY_FACTOR

            delta_x_goal, delta_y_goal, shortest_distance = self.estimate_shortest_distance(finder, grid)
            reward = reward - DISTANCE_TO_UNEXPLORED_RELATION_TO_BATTERY_FACTOR * (shortest_distance/self.battery) # Penalty: distance to unexplored field in relation to battery level weighted with DISTANCE_TO_UNEXPLORED_RELATION_TO_BATTERY_FACTOR
        return reward


    # 0: Forward, 1: Backward, 2: Left, 3: Right
    def step(self, action):
        old_x = self.x
        old_y = self.y
        if action == 0:
            self.move_forward()
        elif action == 1:
            self.move_backward()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()
        elif action == 4:
            self.land()
        self.last_positions.pop(0)
        self.last_positions.append((self.x, self.y))
        if not self.requires_external_values:
            self.battery = self.battery - BATTERY_COST_STEP
        new_field_visited = False
        number_newly_explored_fields = 0
        old_explored = len(self.explored)
        if not self.outside_field():
            if not (self.x, self.y) in self.visited:
                new_field_visited = True
            self.visited.add((self.x, self.y))
            self.explore()
        new_explored = len(self.explored)
        number_newly_explored_fields = new_explored - old_explored

        self.done = self.is_done()

        reward = self.reward(new_field_visited, number_newly_explored_fields)
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        field = np.zeros((WIDTH, DEPTH), dtype=int)
        for i in range(WIDTH):
            for j in range(DEPTH):
                if (i,j) == (self.x, self.y):
                    field[i,j] = 4
                elif (i,j) in self.explored:
                    field[i,j] = self.grid[i,j]
                else:
                    field[i,j] = 0
        return field.T

    def estimate_shortest_distance(self, finder=None, grid=None):
        distance = WIDTH + DEPTH
        result_x = WIDTH
        result_y = DEPTH

        delta_x_goal = 0
        delta_y_goal = 0

        if finder == None and grid == None:
            field_as_array = self.get_field_as_array()
            walkable_field = [[0 if entry == 2 or entry == 3 else 1 for entry in row] for row in field_as_array]
            grid = Grid(matrix = walkable_field)
            finder = AStarFinder(diagonal_movement = DiagonalMovement.never)

        for i in range(0,WIDTH):
            for j in range(0,DEPTH):
                if not (i,j) in self.explored:
                    #estimated_distance_x = abs(self.x - i)
                    #estimated_distance_y = abs(self.y - j)
                    #estimated_distance = estimated_distance_x + estimated_distance_y
                    if self.x >= 0 and self.x < WIDTH and self.y >= 0 and self.y < DEPTH:
                        start_pos = grid.node(self.x, self.y)
                        search_pos   = grid.node(i, j)


                        path, runs = finder.find_path(start_pos, search_pos, grid)
                        estimated_distance = len(path) - 1 if len(path) > 0 else 0
                        if len(path) > 1:
                            delta_x_goal = path[1].x - self.x
                            delta_y_goal = path[1].y - self.y
                        if distance > estimated_distance:
                            distance = estimated_distance
                            #result_x = estimated_distance_x
                            #result_y = estimated_distance_y
        return delta_x_goal, delta_y_goal, distance

    def get_field_as_array(self):
        field = np.zeros_like(self.grid, dtype=int)
        for (x, y) in self.explored:
            field[x, y] = self.grid[x, y]
        return field

    def get_field(self):
        field = []
        for i in range(1,WIDTH+1):
            for j in range(1,DEPTH+1):
                if (i-1,j-1) in self.explored:
                    field.append("({} |-> {} |-> {})".format(i, j, str(self.grid[i-1, j-1])))
                else:
                    field.append("({} |-> {} |-> 0)".format(i,j))
        result = ", ".join(field)
        return "{{{0}}}".format(result)

    def synchronize_external_values(self, left, right, forward, backward, battery_level):
        if self.x - 1 >= 0:
            self.grid[self.x-1, self.y] = right
        if self.x + 1 <= WIDTH - 1:
            self.grid[self.x+1, self.y] = left
        if self.y - 1 >= 0:
            self.grid[self.x, self.y-1] = backward
        if self.y + 1 <= DEPTH - 1:
            self.grid[self.x, self.y+1] = forward
        self.battery = battery_level

def get_paths():
    if len(sys.argv) > 2:
        model_id = sys.argv[2]
    else:
        model_id = 'new'

    save_path = os.path.join(SAVE_PATH, model_id)
    model_path = os.path.join(save_path, "trained_model")

    return save_path, model_path

action_names = {
    0: "MAIN_FORWARD",
    1: "MAIN_BACKWARD",
    2: "MAIN_LEFT",
    3: "MAIN_RIGHT",
    4: "MAIN_LAND"
}

action_names_inv = {
    "MOVE_FORWARD": 0,
    "MAIN_BACKWARD": 1,
    "MAIN_LEFT": 2,
    "MAIN_RIGHT": 3,
    "MAIN_LAND": 4
}

def read_line(socket):
    result = []
    while True:
        data = socket.recv(1).decode('utf-8')
        if data == '\n':
            break
        result.append(data)
    return ''.join(result)

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'train':
        env = DroneEnv(False)
        save_path, model_path = get_paths()
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256, 256, 256, 256, 256]),
                    learning_rate=5e-4,
                    buffer_size=50000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.9,  # Discount factor
                    exploration_fraction=0.4,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=save_path)
        print("Start training...")

        # Save a checkpoint every 100 000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=save_path,
            name_prefix="rl_model"
        )

        model.learn(int(300_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(SAVE_PATH)
    elif sys.argv[1] == 'test':
        env = DroneEnv(False)
        save_path, model_path = get_paths()
        model = DQN.load("models/drone.zip")

        for i in range(1):
            obs, _ = env.reset()
            print("INIT")
            print(env.render())
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                print("------------------")
                print(action_names.get(int(action)))
                print(env.render())
                print(env.get_field())
                print(reward)
                print(env.battery)
                if done:
                    break
    else:
        try:
            env = DroneEnv(True)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                port = int(sys.argv[1])
                client_socket.connect(("127.0.0.1", port))
                model = DQN.load("models/drone.zip")
                while True:
                    obs, _ = env.reset()
                    reward = 0.0
                    info = None
                    prev_obs = None
                    done = False
                    finished = False
                    delta = 1000

                    request = json.loads(read_line(client_socket))
                    response = json.dumps({
                        'op': '$setup_constants',
                        'delta': 0,
                        'predicate': "1=1",
                        'done': 'false'
                    }) + "\n"

                    client_socket.sendall(response.encode('utf-8'))

                    request = json.loads(read_line(client_socket))
                    response = json.dumps({
                        'op': '$initialise_machine',
                        'delta': 0,
                        'predicate': "1=1",
                        'done': 'false'
                    }) + "\n"
                    client_socket.sendall(response.encode('utf-8'))

                    request = json.loads(read_line(client_socket))
                    response = json.dumps({
                        'op': 'MAIN_SYNCHRONIZE_BATTERY',
                        'delta': delta,
                        'predicate': "1=1",
                        'done': 'false'
                    }) + "\n"
                    client_socket.sendall(response.encode('utf-8'))

                    request = json.loads(read_line(client_socket))
                    response = json.dumps({
                        'op': 'MAIN_TAKEOFF',
                        'delta': 0,
                        'predicate': "1=1",
                        'done': 'false'
                    }) + "\n"
                    client_socket.sendall(response.encode('utf-8'))

                    while not done and not finished:
                        request = json.loads(read_line(client_socket))
                        finished = (int(request['finished']) == 1)
                        enabled_operations = request['enabledOperations']
                        operations_list = enabled_operations.split(",")

                        if 'MAIN_SYNCHRONIZE_BATTERY' in operations_list:
                            response = json.dumps({
                                'op': 'MAIN_SYNCHRONIZE_BATTERY',
                                'delta': delta,
                                'predicate': "1=1",
                                'done': 'false',
                                'externalFormulas': {
                                    'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                    'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                    'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                    'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                    'battery': "battery"
                                }
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                        if 'MAIN_UPDATE_POSITION' in operations_list:
                            response = json.dumps({
                                'op': 'MAIN_UPDATE_POSITION',
                                'delta': 0,
                                'predicate': "1=1",
                                'done': 'false',
                                'externalFormulas': {
                                    'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                    'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                    'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                    'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                    'battery': "battery"
                                }
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                        if 'MAIN_SYNCHRONIZE_BATTERY' in operations_list:
                            response = json.dumps({
                                'op': 'MAIN_SYNCHRONIZE_BATTERY',
                                'delta': 0,
                                'predicate': "1=1",
                                'done': 'false',
                                'externalFormulas': {
                                    'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                    'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                    'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                    'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                    'battery': "battery"
                                }
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                        if 'MAIN_OBSERVE' in operations_list:
                            response = json.dumps({
                                'op': "MAIN_OBSERVE",
                                'delta': 0,
                                'predicate': "1=1",
                                'done': "false",
                                'externalFormulas': {
                                    'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                    'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                    'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                    'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                    'battery': "battery"
                                }
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                        if 'MAIN_SYNCHRONIZE_BATTERY' in operations_list:
                            response = json.dumps({
                                'op': 'MAIN_SYNCHRONIZE_BATTERY',
                                'delta': 0,
                                'predicate': "1=1",
                                'done': 'false',
                                'externalFormulas': {
                                    'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                    'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                    'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                    'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                    'battery': "battery"
                                }
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                        external_values = request['externalValues']
                        left = int(external_values['left'])
                        right = int(external_values['right'])
                        forward = int(external_values['forward'])
                        backward = int(external_values['backward'])
                        battery_level = int(external_values['battery'])

                        env.synchronize_external_values(left, right, forward, backward, battery_level)

                        prev_obs = obs
                        obs_tensor, _ = model.policy.obs_to_tensor(obs)
                        predictions = model.policy.q_net(obs_tensor)
                        action_order = (-predictions).argsort(dim=1)

                        new_action = 0

                        for action in action_order[0]:
                            if action_names.get(int(action)) in operations_list:
                                new_action = action
                                break

                        obs, reward, done, truncated, info = env.step(int(new_action))
                        actionName = action_names.get(int(new_action))


                        response = json.dumps({
                            'op': actionName,
                            'delta': delta,
                            'predicate': "reward = {0}".format(reward),
                            'done': "true" if done else "false",
                            'externalFormulas': {
                                'left': "IF prj1(current_position) + 1 : 1..WIDTH THEN field(prj1(current_position)+1, prj2(current_position)) ELSE 2 END",
                                'right': "IF prj1(current_position) - 1 : 1..WIDTH THEN field(prj1(current_position)-1, prj2(current_position)) ELSE 2 END",
                                'forward': "IF prj2(current_position) + 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) + 1) ELSE 2 END",
                                'backward': "IF prj2(current_position) - 1 : 1..DEPTH THEN field(prj1(current_position), prj2(current_position) - 1) ELSE 2 END",
                                'battery': "battery"
                            }
                        }) + "\n"
                        client_socket.sendall(response.encode('utf-8'))

                        if done:
                            break

        except Exception as e:
            print(f"Error: {e}")

        env.close()
if __name__ == "__main__":
    main()