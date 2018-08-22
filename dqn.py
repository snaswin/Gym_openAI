####################################################################
##	Test script that explains the functionality of the environment
####################################################################


'''
import numpy as np
import gym

env_name = "CartPole-v0"
#env_name = "MountainCar-v0"

env = gym.make(env_name)
env.reset()
random = False

state_size = env.observation_space.shape[0]	#state = Box(4,)
action_size = env.action_space.n	#0 or 1 = 2
#print(state_size, action_size)
	

for i in range(1000):
	env.render()
	#sample = env.action_space.sample()
	sample = i%2
	print("At ",i, ", ", sample)
	
	observation, reward, done, info = env.step(sample)	#random action
	#print("State pairs- ", observation, reward, info)

	if done:
		print("Episode is done at step ", i)
		
env.reset()
	
print("Random simulation Complete. ")

######################################################################

'''

import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

episodes = 1000

class dqnAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen = 2000)
		self.gamma = 0.95	#discount rate
		self.epsilon = 1.0	#exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.01
		self.model = self.build_model()
		
	def build_model(self):
		#NN for deep Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim= self.state_size, activation= 'relu') )
		model.add(Dense(24, activation= 'relu') )
		model.add(Dense(self.action_size, activation= 'linear') )
		model.compile(loss='mse', optimizer= Adam(lr= self.learning_rate))
		return model
	
	def remember(self, state, action, reward, next_state, done):
		#REGISTRY: (state, action, reward, next_state, done)
		self.memory.append( (state, action, reward, next_state, done) )
	
	def act(self, state):
		#random actions
		if np.random.rand() <= self.epsilon:							#random float
			return random.randrange(self.action_size)					#random int from a range
		#model predicted actions
		act_values = self.model.predict(state)
		return np.argmax(act_values[0] )
	
	#A method that trains NN with experiences in memory
	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		
		for state, action, reward, next_state, done in minibatch:
			target = reward												#if done, make our target reward
			if not done:												#Predict the future discounted reward
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0] ) )
			
			#Make agent to appx map the current state to future discounted reward
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		
		if self.epsilon > self.epsilon_min:
			self.epsilon = self.epsilon * self.epsilon_decay
		
		
########################################################################
## 	TRAINING

if __name__ == "__main__":
	
	#initialize gym env & agent
	#env_name = "CartPole-v0"
	env_name = "MountainCar-v0"
	env = gym.make(env_name)
	
	#extract x & y sizes from environment
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	#create Agent
	agent = dqnAgent(state_size, action_size)	#env => state_size & action_size
	
	done = False
	batch_size= 4
	
	for episode in range(episodes):
		state = env.reset()
		state = np.reshape(state, [1,2])
		
		#time_t ---> each frame of the game
		for time_t in range(1000):
			#render
			env.render()
			
			#Decide action
			action = agent.act(state)
			
			#Advance the game to next frame based on the action
			#reward is 1 for every frame the pole survived
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1,2])
			
			#Remember the previous state, action, reward & done
			agent.remember(state, action, reward, next_state, done)
			
			#make next_state the new current state
			state = next_state
			
			#done becomes True when the game ends
			# eg. the agent drops the pole/ failes the game
			if done:
				print("At ", episode,"/",episodes, ", the score is ", time_t)
				break
			
			#train agent with experience of the episode
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
			
				
			
			
			
			
			
			
			
			
			
		
		
		
		
		
		
			
				
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	




