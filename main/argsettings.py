# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:26:33 2022

@author: RuoYang Chen
"""
import numpy as np
class ARGS:
    def __init__(self):       
        self.max_train_steps=int(2e5) #Maximum number of training steps
        self.evaluate_freq=4e3 #Evaluate the policy every 'evaluate_freq' steps
        self.save_freq=20 #Save frequency
        self.batch_size=2048 #Batch size
        self.mini_batch_size=64 #Minibatch size
        self.hidden_width=64 #The number of neurons in hidden layers of the neural network
        self.lr_a=3e-4 #Learning rate of actor
        self.lr_c=3e-4 #Learning rate of critic
        self.gamma=0.99 #Discount factor
        self.lamda=0.95 #GAE parameter
        self.epsilon=0.2 #PPO clip parameter
        self.K_epochs=10 #PPO parameter
        self.use_adv_norm=True #Trick 1:advantage normalization
        self.use_state_norm=False #Trick 2:state normalization
        self.use_reward_norm=False #Trick 3:reward normalization
        self.use_reward_scaling=True  #Trick 4:reward scaling
        self.entropy_coef=0.01 #Trick 5: policy entropy
        self.use_lr_decay=True #Trick 6:learning rate Decay
        self.use_grad_clip=True #Trick 7: Gradient clip
        self.use_orthogonal_init=True #Trick 8: orthogonal initialization
        self.set_adam_eps=True #Trick 9: set Adam epsilon=1e-5
        self.use_tanh=True #Trick 10: tanh activation function
        


class ARGS2:
    def __init__(self):
        self.a_size=[1000,1000] #area size (m^2)
        self.tslot = 0.5 #time slot (s)
        self.PT=np.array([0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
        self.PJ=np.array([0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
        self.W=10000000 #channel bandwidth (Hz)
        self.R_min = 2 
        self.v_UAV = 10 
        self.noise = 0.00000000316 
        self.R_limit = 6
        self.a1 = 1
        self.zeta1 = 1
        self.zeta2 = 10
        self.c_J = 10
        self.c_conf = 0.5
        self.c_eveon = 0.1
        self.c_QoS = 0.5
        self.UAV_disl = 5 
        self.LUnum = 20
        self.Jamnum = 2
        self.BSnum = 5
        self.Evenum = 5
