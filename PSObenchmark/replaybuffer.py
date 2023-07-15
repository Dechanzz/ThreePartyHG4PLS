import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args, name, device):
        #self.s = np.zeros((args.batch_size, args.state_dim))
        self.s = torch.zeros((args.batch_size, args.state_dim)).to(device)
        self.a = torch.zeros((args.batch_size, 1)).to(device)
        self.a_logprob = torch.zeros((args.batch_size, 1)).to(device)
        self.r = torch.zeros((args.batch_size, 1)).to(device)
        self.s_ = torch.zeros((args.batch_size, args.state_dim)).to(device)
        self.done = torch.zeros((args.batch_size, 1)).to(device)
        self.count = 0
        self.name= name

    def store(self, s, a, a_logprob, r, s_, done):
        #print(self.name," store log:",self.count)
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self, device):#ppotype=0: discrete; ppotype=1: continuous
        s = torch.tensor(self.s, dtype=torch.float).to(device)
        #if ppotype==0:
        a = torch.tensor(self.a, dtype=torch.long).to(device)  # In discrete action space, 'a' needs to be torch.long
        #else:
        #a = torch.tensor(self.a, dtype=torch.float) # In continuous action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        r = torch.tensor(self.r, dtype=torch.float).to(device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(device)
        done = torch.tensor(self.done, dtype=torch.float).to(device)

        return s, a, a_logprob, r, s_, done
    
    
    
