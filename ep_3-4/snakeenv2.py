import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
	apple_position = [random.randrange(1,50),random.randrange(1,50)]
	score += 1
	return apple_position, score

def collision_with_boundaries(snake_head):
	if snake_head[0]>=50 or snake_head[0]<0 or snake_head[1]>=50 or snake_head[1]<0 :
		return 1
	else:
		return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0
	
class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
											shape=(5 + 25 + SNAKE_LEN_GOAL,), dtype=np.float32)

    def step(self, action):

        self.prev_actions.append(action)

        self.garden = np.zeros((50, 50))

        #-------------------------MOVEMENT/ACTION------------------------------------
        action -= 1
        # (-1, 0)-Left, (1, 0)-Right, (0, -1)-Up, (0, 1)-Down
        if action != 0:
            if self.snake_dir[0] == 0:
                self.snake_dir = (-self.snake_dir[1]*action, 0)
            else:
                self.snake_dir = (0, self.snake_dir[0]*action)

        self.snake_head[0] += self.snake_dir[0]
        self.snake_head[1] += self.snake_dir[1]
        #_____________________________________________________________________


        #-----------------------COLLISIONS AND INCREMENT----------------------
        apple_reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 1000

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            self.garden = np.zeros((50,50))
            self.done = True
        #_____________________________________________________________________

        #---------------------------UPDATE GARDEN-----------------------------
        if not self.done:
            self.garden[self.apple_position[0]][self.apple_position[1]] = 1
            for (x, y) in self.snake_position:
                self.garden[x-1][y-1] = -1
        #_____________________________________________________________________

        #----------------------------OBSERVATIONS-----------------------------
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_d_x = head_x - self.apple_position[0]
        apple_d_y = head_y - self.apple_position[1]

        prev_apple_d_x = self.snake_position[1][0] - self.apple_position[0]
        prev_apple_d_y = self.snake_position[1][1] - self.apple_position[1]

        apple_dir = abs(prev_apple_d_x) > abs(apple_d_x) or abs(prev_apple_d_y) > abs(apple_d_y)

        snake_length = len(self.snake_position)

        close_obs = np.zeros((5, 5))
        for i_r, r in enumerate(range(self.snake_head[0]-2, self.snake_head[0]+3)):
            for t_r, t in enumerate(range(self.snake_head[1]-2, self.snake_head[1]+3)):
                if t > 49 or r > 49: continue
                close_obs[t_r][i_r] = self.garden[r][t]

        self.observation = [head_x, head_y, apple_d_x, apple_d_y, snake_length] + list(close_obs.flatten()) + list(self.prev_actions)
        self.observation = np.array(self.observation)
        #___________________________________________________________________


        #--------------------------REWARDS----------------------------------
        if apple_dir:
            dir_reward = 1
        else:
            dir_reward = -2
        if self.done:
            self.reward = -100
        else:
            self.reward = apple_reward + dir_reward
        #___________________________________________________________________


        info = {}
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.img = np.zeros((500,500,3),dtype='uint8')

        # Initial Snake and Apple position
        self.snake_position = [[25,25],[24,25],[23,25]]
        self.apple_position = [random.randrange(0,50),random.randrange(0,50)]
        self.score = 0
        self.reward= 0
        self.snake_head = [25,25]
        self.snake_dir = (0, 1)

        #head_x, head_y, apple_d_x, apple_d_y, snake_length, previous moves
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_d_x = head_x - self.apple_position[0]
        apple_d_y = head_y - self.apple_position[1]

        snake_length = len(self.snake_position)

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        close_obs = np.zeros((5, 5))

        self.observation = [head_x, head_y, apple_d_x, apple_d_y, snake_length] + list(close_obs.flatten()) + list(self.prev_actions)
        self.observation = np.array(self.observation)


        return self.observation  # reward, done, info can't be included

    def render(self):
        cv2.imshow('a', self.img)
        cv2.waitKey(100)
    
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0]*10,self.apple_position[1]*10),(self.apple_position[0]*10+10, self.apple_position[1]*10+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0]*10,position[1]*10),(position[0]*10+10,position[1]*10+10),(0,255,0),3)