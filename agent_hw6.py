"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tile3 import IHT, tiles


class Agent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.iht = IHT(2048)
        self.offset_t = 8


    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.x = np.zeros(2048)
        self.x_last = np.zeros(2048)
        self.w = np.random.uniform(0,-0.001,2048)
        self.last_state = 0
        self.offset_t = 8
        self.alpha = 0.1 / self.offset_t
        self.agent_lambda = 0.9
        self.last_features = []
        self.last_action = 0
        self.last_dot = 0
        self.tiles_last = []



    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.last_state = state
        self.last_action, self.last_dot, self.tiles_last = self.choose_action(state[0],state[1])
        # self.x_last = np.copy(x)
        self.z = np.zeros(2048)
        return self.last_action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        action, dot, cur_tiles = self.choose_action(state[0],state[1])
        # self.x = np.copy(x)
        err = reward
        for t in self.tiles_last:
            err = err - self.w[t]
            self.z[t] = 1
        for t in cur_tiles:
            err = err + self.w[t]
        self.w = self.w + (self.alpha * err * self.z)
        self.z = self.agent_lambda * self.z
        # self.x_last = np.copy(self.x)
        # self.x.fill(0)
        self.tiles_last = cur_tiles
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        err = reward
        for t in self.tiles_last:
            err = err - self.w[t]
            self.z[t] = 1
        # err = reward - np.dot(self.w,self.x_last)
        self.w = self.w + self.alpha * err * self.z
        # self.x_last.fill(0)
        return

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message:
            pos_list = np.zeros((50,50))
            vel_list = np.zeros((50,50))
            tile = []
            steps = 50
            feat = np.zeros(2048)
            val = np.zeros((50,50))
            for i in range(steps):
                for j in range(steps):
                    pos = -1.2 + (i * 1.7/steps)
                    vel = -0.07 + (j * 0.14/steps)
                    pos_list[i][j] = pos
                    vel_list[i][j] = vel
                    # pos_list.append(pos)
                    # vel_list.append(vel)
                    action, act_val, tiles = self.choose_action(pos,vel)
                    val[i][j] = act_val
        return val, pos_list, vel_list


    def mytiles(self,x,x_dot,action):
        scaling_factor_x = 8/(1.7)
        scaling_factor_xdot = 8/(0.14)
        x = x + 1.2
        x_dot = x_dot + 0.07
        x = x * scaling_factor_x
        x_dot = x_dot * scaling_factor_xdot
        return tiles(self.iht,self.offset_t,[x,x_dot],[action])

    def choose_action(self,x,x_dot):
        tiles=[]
        tiles.append(self.mytiles(x,x_dot,0))
        tiles.append(self.mytiles(x,x_dot,1))
        tiles.append(self.mytiles(x,x_dot,2))
        # x1 = np.zeros(2048)
        # x2 = np.zeros(2048)
        # x3 = np.zeros(2048)
        action = 0
        act_val = 0
        act1 = 0
        act2 = 0
        act3 = 0
        for t in tiles[0]:
            act1 += self.w[t]
            # x1[t] = 1
        for t in tiles[1]:
            act2 += self.w[t]
            # x2[t] = 1
        for t in tiles[2]:
            act3 += self.w[t]
            # x3[t] = 1
        # act1 = np.dot(self.w,x1)
        # act2 = np.dot(self.w,x2)
        # act3 = np.dot(self.w,x3)
        actVal=np.array([act1,act2,act3])
        # val = max(act1,act2,act3)
        # eqAction = np.where(actVal>=val)
        # print(eqAction)
        # choAct = eqAction[0][np.random(eqAction[0].size)]
        choAct = np.argmax(actVal)
        tile = tiles[choAct]
        # features = np.array(2048)
        # if val == act1 and val == act2:
        #     flip = np.random.randint(2)
        #     if flip:
        #         action = 0
        #         act_val = act1
        #         features = x1
        #         tile = tiles1
        #     else:
        #         action = 1
        #         act_val = act2
        #         features = x2
        #         tile = tiles2
        # elif val == act1 and val == act3:
        #     flip = np.random.randint(2)
        #     if flip:
        #         action = 0
        #         act_val = act1
        #         features = x1
        #         tile = tiles1
        #     else:
        #         action = 2
        #         act_val = act3
        #         features = x3
        #         tile = tiles3
        # elif val == act2 and val == act3:
        #     flip = np.random.randint(2)
        #     if flip:
        #         action = 1
        #         act_val = act2
        #         features = x2
        #         tile = tiles2
        #     else:
        #         action = 2
        #         act_val = act3
        #         features = x3
        #         tile = tiles3
        # elif val == act1 and val == act2 and val == act3:
        #     flip = np.random.randint(3)
        #     if flip == 0:
        #         action = 0
        #         act_val = act1
        #         features = x1
        #         tile = tiles1
        #     elif flip == 1:
        #         action = 1
        #         act_val = act2
        #         features = x2
        #         tile = tiles2
        #     else:
        #         action = 2
        #         act_val = act3
        #         features = x3
        #         tile = tiles3
        # elif val == act1:
        #     action = 0
        #     act_val = act1
        #     features = x1
        #     tile = tiles1
        # elif val == act2:
        #     action = 1
        #     act_val = act2
        #     features = x2
        #     tile = tiles2
        # elif val == act3:
            # action = 2
            # act_val = act3
            # features = x3
            # tile = tiles3
        # return action, act_val, tile
        return choAct, actVal[choAct], tiles[choAct]