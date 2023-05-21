# Stochastic Control with Deep Reinforcement Learning

This repository contains the code used for the research project "Stochastic Control with Deep Reinforcement Learning." 
The project explores the use of Deep Reinforcement Learning for solving stochastic control problems, which arise 
in many fields including finance, engineering, and robotics.

## Reinforcement Learning

Reinforcement Learning (RL) is a subset of machine learning that involves training an agent to make decisions in an 
environment to maximize a cumulative reward. Similar to how humans learn, RL involves learning through trial and error. The 
agent interacts with the environment and receives feedback in the form of rewards or penalties for its actions. The goal 
of the agent is to learn the optimal policy that maximizes the cumulative reward over time. 

## Deep Reinforcement Learning

Deep Reinforcement Learning (DRL) is a subset of Reinforcement Learning that uses deep neural networks 
to learn the optimal control policies of an agent. The agent learns by interacting with the environment, 
and the neural network is trained to predict the best action to take in a given state. The agent receives 
a reward signal from the environment for every action it takes, and the neural network is trained to maximize 
the cumulative reward over time.

#### Urban Transportation Planning

The first problem we explore is urban transportation planning, which involves optimizing the path of a bus through a busy city with 
uncirtain demand and traffic conditions. The goal of the agent is to serve as many passengers as possible while minimizing the average 
waiting time. Here we use a Deep Q Network approach to train the agent to learn the optimal policy mapping city demand and traffic observations
to possible locations the bus may drive to.

#### Asset Portfolio Construction

The second problem we explore is asset portfolio construction, which is the process of selecting a optimal weights for a portfolio 
of assets that maximizes returns while minimizing risk. Here, we used a Deep Deterministic Polict Gradient (DDPG) model to learn 
the optimal policy to select asset weights based on observed statistical measures of each asset.
