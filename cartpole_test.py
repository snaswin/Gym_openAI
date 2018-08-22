####################################################################
##	Test script that explains the functionality of the environment
####################################################################

import numpy as np
import gym

env_name = "CartPole-v0"
#env_name = "MountainCar-v0"

env = gym.make(env_name)
env.reset()
random = False

for i in range(1000):
	env.render()
	#sample = env.action_space.sample()
	sample = i%2
	print("At ",i, ", ", sample)
	
	observation, reward, done, info = env.step(sample)	#random action
	#print("State pairs- ", observation, reward, info)
	
	state_size = env.observation_space.shape[0]	#state = Box(4,)
	action_size = env.action_space.n	#0 or 1 = 2
	print(state_size, action_size)
	
	if done:
		print("Episode is done at step ", i)
		
env.reset()
	
print("Random simulation Complete. ")

