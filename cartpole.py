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
	
	if random == True:
		sample = env.action_space.sample()
	else:
		sample = i%2
		
	print("At ",i, ", ", sample)
	observation, reward, done, info = env.step(sample)	#random action
	print("State pairs- ", observation, reward, info)
	if done:
		print("Episode is done at step ", i)
		
env.reset()
	
print("Random simulation Complete. ")

