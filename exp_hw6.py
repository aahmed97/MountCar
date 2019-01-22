#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent
from rl_glue import RLGlue
from env_hw6 import Environment
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import plot


def question_1():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        st = time.time()
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            # print(steps[r, e])
        finish = time.time() - st
        print(str(finish)+" seconds elapsed")
    np.save('steps', steps)

def question_3():
    num_episodes = 1000
    num_runs = 1
    max_eps_steps = 100000

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    for r in range(num_runs):
        start = time.time()
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
    end = time.time()
    print(str(end-start)+" seconds elapsed")
    action_vals, pos, vel = rlglue.rl_agent_message("return info")
    action_vals = np.multiply(action_vals, -1)
    fig = plt.figure()
    ax = fig.gca(projection= '3d')
    ax.plot_surface(pos,vel,action_vals)
    plt.show()



if __name__ == "__main__":
    print("Starting question 1...")
    question_1()
    print("Done question 1")
    plot.plotting_func()
    print("Starting question 3...")
    question_3()
    print("Done question 3")

