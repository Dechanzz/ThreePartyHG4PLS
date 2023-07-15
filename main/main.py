import copy
import numpy as np
import torch 
import math
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from ppo_continuous import PPO_continuous 
from argsettings import ARGS
from argsettings import ARGS2
import matplotlib
import matplotlib.pyplot as plt
#parameters
LUs = np.array([])
Jams = np.array([])
BSs = np.array([])
Eves = np.array([])
 
arg2 = ARGS2()

a_size=arg2.a_size #area size (m^2)
tslot = arg2.tslot #time slot (s)
PT=arg2.PT
PJ=arg2.PJ
W=arg2.W
R_min = arg2.R_min
v_UAV = arg2.v_UAV
noise = arg2.noise 
R_limit = arg2.R_limit
a1 = arg2.a1
zeta1 = arg2.zeta1
zeta2 = arg2.zeta2
c_J = arg2.c_J
c_conf = arg2.c_conf
c_eveon = arg2.c_eveon
c_QoS = arg2.c_QoS
UAV_disl = arg2.UAV_disl #safety distance between UAVs

LUnum = arg2.LUnum
Jamnum = arg2.Jamnum
BSnum = arg2.BSnum
Evenum = arg2.Evenum



class BS:
    def __init__(self,locxy):
        self.loc = locxy       

class Eve:
    def __init__(self,locxy):
        self.loc = locxy
        self.open = 1

class LU:
    def __init__(self,locxy):
        self.loc = locxy
        self.P = 0.0
        self.bs = -1 #-1 indicates none base station chosen 
        self.bsrange = np.array([]) #BSs that can be chosen

class Jammer:
    def __init__(self,locxy):
        self.loc = locxy
        self.P = 0.0

LUs=np.array([])
Jams=np.array([])
BSs=np.array([])
Eves=np.array([])

def initloc2D(num, area_size): #create num locs in map of area_size
    x = area_size[0]
    y = area_size[1]
    tr = np.array([])
    for i in range(0,num):
        tx = np.random.randint(0,x)
        ty = np.random.randint(0,y)
        tr = np.append(tr, np.array([tx,ty]))
    return tr   

def cacu_cgain(devx, devy): #return channel gain of devx and devy
    locx = devx.loc
    locy = devy.loc
    '''if locx.shape[0] > locy.shape[0]:
        temp0 = locx
        locx = locy
        locy = temp0
    while locx.shape[0] != locy.shape[0]:
        locx = np.append(locx, 0.0)'''
    return (np.linalg.norm(locx-locy))**(-2)

def dev_index(ind, LUs, Jams, BSs, Eves, slct=-1): #serch correspond devx and devy of index ind, if slct = 0: search sender, slct = 1: search receiver
    if len(ind)==2:
        if ind[0]>LUnum-1:
            devx = Jams[ind[0]-LUnum]
        else:
            devx = LUs[ind[0]]
        if ind[1]>BSnum-1:
            devy = Eves[ind[1]-BSnum]
        else:
            devy = BSs[ind[1]]   
        return devx, devy
    elif len(ind)==1:
        if slct == 0:
            if ind[0]>LUnum-1:
                devx = Jams[ind[0]-LUnum]
            else:
                devx = LUs[ind[0]]
        elif slct == 1:
            if ind[0]>BSnum-1:
                devx = Eves[ind[0]-BSnum]
            else:
                devx = BSs[ind[0]] 
        return devx
         
def cacu_rate(wid, sig, nois):
    return wid*math.log2(1+sig/nois)
   
def init_net_topo(LUs, Jams, BSs, Eves): #initialize the network topology graph of channel gain and power allocation and rates
    tnet_h = np.zeros([LUnum+Jamnum,BSnum+Evenum]) #h_ij
    tnet_P = np.zeros([LUnum+Jamnum,BSnum+Evenum]) #P^T and P^J
    tnet_R = np.zeros([LUnum,BSnum+Evenum]) #R^T and R^E
    tnet_R0 = np.zeros([LUnum,BSnum+Evenum]) #R without jamming
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_h[i][j]=cacu_cgain(tdx, tdy) #channel gain include all channel gains
            if j < BSnum:
                if j == tdx.bs:
                    tnet_P[i][j]=tdx.P
            elif j>=BSnum:
                tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    for i in range(LUnum,LUnum+Jamnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j],  LUs, Jams, BSs, Eves)
            tnet_h[i][j]=cacu_cgain(tdx, tdy) #channel gain include all channel gains
            tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            if tnet_P[i][j] != 0.0:
                pJg = 0.0
                for k in range(LUnum,LUnum+Jamnum):
                    pJg += tnet_P[k][j]*tnet_h[k][j]
                tnet_R[i][j] = math.log2(1+tnet_P[i][j]*tnet_h[i][j]/(noise+pJg))  
                tnet_R0[i][j] = math.log2(1+tnet_P[i][j]*tnet_h[i][j]/(noise))  
    return tnet_h, tnet_P, tnet_R, tnet_R0

def update_rate(topoh, topoP):
    tnet_R = np.zeros([LUnum,BSnum+Evenum]) #R^T and R^E
    tnet_R0 = np.zeros([LUnum,BSnum+Evenum]) #R without jamming
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            if topoP[i][j] != 0.0:
                pJg = 0.0
                for k in range(LUnum,LUnum+Jamnum):
                    pJg += topoP[k][j]*topoh[k][j]
                tnet_R[i][j] = math.log2(1+topoP[i][j]*topoh[i][j]/(noise+pJg))  
                tnet_R0[i][j] = math.log2(1+topoP[i][j]*topoh[i][j]/(noise))  
    return tnet_R, tnet_R0

def maxtrans_rates(): #
    tnet_R = np.zeros([LUnum,BSnum]) #R^T and R^E
    for i in range(0,LUnum):
        for j in range(0,BSnum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_R[i][j] = math.log2(1+PT[-1]*cacu_cgain(tdx, tdy)/noise)                  
    return tnet_R

def gene_action(topo_limit, LUs):
    for i in range(0,LUnum):
        for j in range(0,BSnum):
            LUs[i].bsrange=np.append(LUs[i].bsrange,j)
            #if topo_limit[i][j]>=R_limit:
                #LUs[i].bsrange=np.append(LUs[i].bsrange,j)

# initialize LUs, Jams, BSa, Eves
useold = True

if useold: 
    np.load.__defaults__=(None, True, True, 'ASCII')
    LUs = np.load('orgLUs.npy')
    Jams = np.load('orgJams.npy')
    BSs = np.load('orgBSs.npy')
    Eves = np.load('orgEves.npy')
    np.load.__defaults__=(None, False, True, 'ASCII')
else:
    for i in range(0,LUnum):
        LUs=np.append(LUs,LU(initloc2D(1,[1000,1000])))
    for i in range(0,Jamnum):
        Jams=np.append(Jams,Jammer(initloc2D(1,[1000,1000])))
    for i in range(0,BSnum):
        BSs=np.append(BSs,BS(initloc2D(1,[1000,1000])))
    for i in range(0,Evenum):
        Eves=np.append(Eves,Eve(initloc2D(1,[1000,1000])))

for i in LUs:
    print("LU", i.bsrange)




def testnum(ary,lim):
    tres = 0
    for i in ary:
        if i>=lim:
            tres+=1
    return tres

if not useold: 
    topo_limit=maxtrans_rates()
    for i in range(0,LUnum):
        while testnum(topo_limit[i],R_limit)<2 :
            LUs[i].loc=initloc2D(1,[1000,1000])
            topo_limit=maxtrans_rates()
    gene_action(topo_limit, LUs)       
    for i in LUs:   
        i.P=PT[np.random.randint(1,len(PT))]
        i.bs=int (i.bsrange[np.random.randint(0,len(i.bsrange))])
    for i in Jams:
        i.P=PJ[np.random.randint(1,len(PJ))]
else:
    # topo_limit=maxtrans_rates()
    # gene_action(topo_limit, LUs)
    for i in LUs:   
        i.P=PT[np.random.randint(1,len(PT))]
        i.bs=int (i.bsrange[np.random.randint(0,len(i.bsrange))])
    for i in Jams:
        i.P=PJ[np.random.randint(1,len(PJ))]

for i in LUs:
    print("LU", i.bsrange)

    
topo_h, topo_P, topo_R, topo_R0 = init_net_topo(LUs, Jams, BSs, Eves)
# topo_limit, topo_h, topo_P, topo_R, topo_R0

org_LUs = copy.deepcopy(LUs)
org_Jams = copy.deepcopy(Jams)
org_BSs = copy.deepcopy(BSs)
org_Eves = copy.deepcopy(Eves)


if not useold: 
    np.save('orgLUs.npy',org_LUs)
    np.save('orgJams.npy',org_Jams)
    np.save('orgBSs.npy',org_BSs)
    np.save('orgEves.npy',org_Eves)
    
# environment
def gene_state(topo_h, topo_P, col_parti, device):
    rtrn_state = np.append((topo_h*topo_P).flatten(),np.array(col_parti))
    rtrn_state = torch.tensor(rtrn_state).to(device)
    return rtrn_state

def gene_state_J(topo_h, topo_P, col_parti, mu1, mu2, device):
    rtrn_state = np.append((topo_h*topo_P).flatten(),np.array(col_parti))
    rtrn_state = np.append(rtrn_state, [mu1,mu2])
    rtrn_state = torch.tensor(rtrn_state).to(device)
    return rtrn_state    
    
def action_trans_Eve(action_code, LUs, Jams, BSs, Eves):#transform the Eve actions code to real actions
    #action code is the action chosen by actor network
    #Evesdroppers have 4*4 action, the code from 0~15
    a_str = bin(int(action_code))[2:].zfill(Evenum)
    for i in range(0,Evenum):
        if a_str[i]=='0':
            Eves[i].open = 0
        elif a_str[i]=='1':
            Eves[i].open = 1  
    tnet_h = np.zeros([LUnum+Jamnum,BSnum+Evenum]) #h_ij
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_h[i][j]=cacu_cgain(tdx, tdy) #channel gain include all channel gains  
    for i in range(LUnum,LUnum+Jamnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_h[i][j]=cacu_cgain(tdx, tdy) #channel gain include all channel gains   
    return tnet_h #updated topo_h after Eves' actions

def action_trans_LU(action_codes, LUs, Jams, BSs, Eves):#transform the LU action codes to real actions
    #action codes are the action arry of length len(LUs) chosen by actor networks
    #single LU has actions of len(LU.bsrange)*len(PT)
    for i in range(0,LUnum):
        a_BS = int(action_codes[i]/len(PT))
        a_PT = action_codes[i]%len(PT)
        LUs[i].bs = int(LUs[i].bsrange[a_BS])
        LUs[i].P = PT[int(a_PT)]
    tnet_P = np.zeros([LUnum+Jamnum,BSnum+Evenum]) #P^T and P^J
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            if j < BSnum:
                if j == tdx.bs:
                    tnet_P[i][j]=tdx.P
            elif j>=BSnum:
                tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    for i in range(LUnum,LUnum+Jamnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    return tnet_P #update topo_P after LUs' actions

#act_E=[locE,miuE]       
def gene_revenue_Eve(topoR,topoR0, LUs): #caculate V_Eve and V_Eve^0 for all Eves
    tV = 0.0
    tV0 = 0.0
    for i in range(0,LUnum):        
        tV += max(topoR[i,BSnum]*Eves[0].open,topoR[i,BSnum+1]*Eves[1].open)-topoR[i,LUs[i].bs]
        tV0 += max(topoR0[i,BSnum]*Eves[0].open,topoR0[i,BSnum+1]*Eves[1].open)-topoR0[i,LUs[i].bs]
    tV*=a1
    tV0*=a1
    return tV, tV0, Eves[0].open*c_eveon*tslot+Eves[1].open*c_eveon*tslot
'''
zeta2 = 30
for i in range(0,LUnum):
    print(zeta2*LUs[i].P,",",topo_R[i,LUs[i].bs]-topo_R[i,BSnum:].max())
LV1,LV2=gene_revenue_LU(topo_R,topo_R0)
print(LV1.sum(),LV2.sum())'''

def gene_revenue_LU(topoR,topoR0, LUs): #caculate V_LU and V_LU^0 for single LU agent and LUs agent
    tV = np.array([])
    tV0 = np.array([])
    pcostL = np.array([])
    for i in range(0,LUnum):
        tV = np.append(tV, zeta1*(topoR[i,LUs[i].bs]-max(topoR[i,BSnum]*Eves[0].open,topoR[i,BSnum+1]*Eves[1].open)))
        tV0 = np.append(tV0, zeta1*(topoR0[i,LUs[i].bs]-max(topoR0[i,BSnum]*Eves[0].open,topoR0[i,BSnum+1]*Eves[1].open)))
        if topoR[i,LUs[i].bs]<R_min:
            QoScost = c_QoS*(topoR[i,LUs[i].bs]-R_min)
            tV+=QoScost
            tV0+=QoScost 
        pcostL = np.append(pcostL,zeta2*LUs[i].P)
    return tV, tV0, pcostL

def action_trans_Jam(action_code, LUs, Jams, BSs, Eves):#transform the Jams' actions code to real actions
    #action code is the action chosen by actor network
    #Jammers have 5*5 action, the code from 0~24
    a0=int(action_code/len(PJ)) #action of Jam1
    a1=action_code%len(PJ) #action of Jam2
    Jams[0].P=PJ[a0]
    Jams[1].P=PJ[int(a1)]
    tnet_P = np.zeros([LUnum+Jamnum,BSnum+Evenum]) #P^T and P^J
    for i in range(0,LUnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            if j < BSnum:
                if j == tdx.bs:
                    tnet_P[i][j]=tdx.P
            elif j>=BSnum:
                tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    for i in range(LUnum,LUnum+Jamnum):
        for j in range(0,BSnum+Evenum):
            tdx, tdy = dev_index([i,j], LUs, Jams, BSs, Eves)
            tnet_P[i][j]=tdx.P #power allocation only include the actual power allocated to corresponding channels   
    return tnet_P #update topo_P after LUs' actions

def col_formation(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    #print("mu_E:",mu_E)
    #print("mu_L:",mu_L)
    V_E, V_E0, pcostE = gene_revenue_Eve(topo_R,topo_R0, LUs)
    V_L,V_L0, pcostL = gene_revenue_LU(topo_R,topo_R0, LUs)
    if Jams[0].P+Jams[1].P==0:
        col_now = [0,0]
        if col_old==[0,0]:
            UJ=0.0
        else:
            UJ=-c_conf
        return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ
    sV_L = V_L.sum()
    sV_L0 = V_L0.sum()
    
    #print("raw utility of Eves:",V_E-V_E0)
    #print("raw utility of LUs:",sV_L-sV_L0)
    
    if col_old[0]==1:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)
    else:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
               
    if col_old[1]==1:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)
    else:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
    
    if UJL>=UJE and UJL>=0:
        col_now=[0,1]
        UJ=UJL
        V_L = V_L-mu_L*(V_L-V_L0)
        #print("incentive from LUs:",UJ)
    elif UJL<UJE and UJE>=0:
        col_now=[1,0]
        UJ=UJE
        V_E = V_E-mu_E*(V_E-V_E0)
        #print("incentive from Eves:",UJ)
    elif UJL<=0 and UJE<=0:
        col_now=[0,0]
        if col_old == [0,0]:
            UJ = 0
        else:
            UJ = -c_conf
    #print("Jamming power cost:",c_J*(Jams[0].P+Jams[1].P))
    
    #print(col_old,"->",col_now)
    
    return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ

muL = []
muE = []
def main(args, LUs, Jams, BSs, Eves):  
    #topo_h, topo_P, topo_R, topo_R0 = init_net_topo(LUs, Jams, BSs, Eves)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.state_dim = (LUnum+Jamnum)*(BSnum+Evenum)+2 #state_dim hidden_width action_dim
    
    replay_bufferEveopen = ReplayBuffer(args,"Eveopen",device)
    replay_bufferEvemu = ReplayBuffer(args,"Evemu",device)
    replay_bufferLUP = []
    for i in range(0,LUnum):
        replay_bufferLUP.append(ReplayBuffer(args,"LUBF",device))
    replay_bufferLUmu = ReplayBuffer(args,"LUmu",device)
    
    args.state_dim += 2
    replay_bufferJam = ReplayBuffer(args,"Jam",device)
    args.state_dim -= 2
    
    #Eveagent_open, Eveagent_mu, LUagents, LUagent_mu, Jamagent
    args.action_dim = 32
    Eveagent_open = PPO_discrete(args,device)
    #print(Eveagent_open.actor.device)
    #print(Eveagent_open.critic.device)
    args.action_dim = 10
    Eveagent_mu = PPO_discrete(args,device)
    LUagents=[]
    for i in range(0,LUnum):
        args.action_dim = len(LUs[i].bsrange)*len(PT)
        LUagents.append(PPO_discrete(args,device))
    args.action_dim = 10
    LUagent_mu = PPO_discrete(args, device)
    
    args.action_dim = len(PJ)**len(Jams)
    args.state_dim += 2
    Jamagent = PPO_discrete(args,device)
    args.state_dim -= 2
    
    state_norm = Normalization(args.state_dim, device)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(1, device)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma, device = device)
      
    args.max_episode_steps = 200
    total_steps = 0 
    evaluate_num = 0
    
    evaluate_rewards_E = []
    evaluate_rewards_L = []
    evaluate_rewards_J = []
    evaluate_rate_E = []
    evaluate_rate_T = []
    evaluate_rate_S = []
    
    muL = []
    muE = []
    while total_steps < args.max_train_steps:
        print("step:",total_steps," ",total_steps/200000," ",total_steps/201)
        topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
        LUs=copy.deepcopy(org_LUs)
        Jams=copy.deepcopy(org_Jams)
        BSs=copy.deepcopy(org_BSs)
        Eves =copy.deepcopy(org_Eves)
        done = False
        col_part = [0,0]
        s = gene_state(topo_h, topo_P, col_part, device)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        
        evaluate_rewardE, evaluate_rewardL, evaluate_rewardJ, eva_RE, eva_RT, eva_RS = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        while not done:
            if episode_steps>= args.max_episode_steps - 1:
                done = True
            episode_steps += 1
            a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
            topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
            E0s_ = gene_state(topo_h, topo_P, col_part, device)
            #print("E0s_dev:",E0s_.device)
            if args.use_state_norm:
                E0s_ = state_norm(E0s_)
            
            a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
            muE.append(a_Emu)
            
            a_LUs = []
            a_LUslogprob = []
            for i in range(0,LUnum):
                tempa,templp = LUagents[i].choose_action(E0s_)              
                a_LUs.append(tempa)
                a_LUslogprob.append(templp)
            topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
            E1s_ = gene_state(topo_h, topo_P, col_part, device)
            if args.use_state_norm:
                E1s_ = state_norm(E1s_)
            
            a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
            muL.append(a_Lmu)
            
            E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
            a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
            topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)

            topo_R,topo_R0 = update_rate(topo_h,topo_P)
            col_part_, r_E, r_L, r_L_sum, r_J =col_formation(a_Emu,a_Lmu,col_part,topo_R,topo_R0,LUs,Jams,Eves)
            col_part = col_part_
            s_ = gene_state(topo_h, topo_P, col_part, device)
            if args.use_state_norm:
                s_ = state_norm(s_)
            
            evaluate_rewardE += r_E
            evaluate_rewardL += r_L_sum
            evaluate_rewardJ += r_J
            for i in range(0,LUnum):
                eva_RE += max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
                eva_RT += topo_R[i,LUs[i].bs]
                eva_RS += topo_R[i,LUs[i].bs] - max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
            
            #col_now, V_E, V_L, V_L.sum(), UJ
            
            replay_bufferEveopen.store(s, a_Edir, a_Edirlogprob, r_E, s_, done)
            replay_bufferEvemu.store(E0s_, a_Emu, a_Emulogprob, r_E, s_, done)
            for i in range(0,LUnum):
                replay_bufferLUP[i].store(E0s_, a_LUs[i], a_LUslogprob[i], r_L[i], s_, done)
            replay_bufferLUmu.store(E1s_, a_Lmu, a_Lmulogprob, r_L_sum, s_, done)
            
            s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
            replay_bufferJam.store(E1s_J, a_Jam, a_Jamlogprob, r_J, s_J, done)
            
            s = s_
            total_steps += 1
            
            if replay_bufferEveopen.count == args.batch_size:
                Eveagent_open.update(replay_bufferEveopen, total_steps)
                replay_bufferEveopen.count = 0
                
                Eveagent_mu.update(replay_bufferEvemu, total_steps)
                replay_bufferEvemu.count = 0
                
                for i in range(0,LUnum):
                    LUagents[i].update(replay_bufferLUP[i], total_steps)
                    replay_bufferLUP[i].count=0
                    
                LUagent_mu.update(replay_bufferLUmu, total_steps)
                replay_bufferLUmu.count = 0
                
                Jamagent.update(replay_bufferJam, total_steps)
                replay_bufferJam.count = 0
        
        print("evaluate_num:",evaluate_num," evaluate_reward:",evaluate_rewardE/201, evaluate_rewardL/201, evaluate_rewardJ/201)
        # print("muE",muE)
        # print("muL",muL)
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_rewards_E.append(evaluate_rewardE/201)
            evaluate_rewards_L.append(evaluate_rewardL/201)
            evaluate_rewards_J.append(evaluate_rewardJ/201)
            evaluate_rate_E.append(eva_RE/201)
            evaluate_rate_T.append(eva_RT/201)
            evaluate_rate_S.append(eva_RS/201)
            # print("evaluate_num:",evaluate_num," evaluate_reward:",evaluate_rewardE/201, evaluate_rewardL/201, evaluate_rewardJ/201)
            # Save the rewards
            if evaluate_num % args.save_freq == 0:
                # np.save('TSSGrewardsE.npy', np.array(evaluate_rewards_E))
                # np.save('TSSGrewardsL.npy', np.array(evaluate_rewards_L))
                # np.save('TSSGrewardsJ.npy', np.array(evaluate_rewards_J))
                # np.save('TSSGrateE.npy', np.array(evaluate_rate_E))
                # np.save('TSSGrateT.npy', np.array(evaluate_rate_T))
                # np.save('TSSGrateS.npy', np.array(evaluate_rate_S))
                np.save('TSSGrewardsE0.npy', np.array(evaluate_rewards_E))
                np.save('TSSGrewardsL0.npy', np.array(evaluate_rewards_L))
                np.save('TSSGrewardsJ0.npy', np.array(evaluate_rewards_J))
                np.save('TSSGrateE0.npy', np.array(evaluate_rate_E))
                np.save('TSSGrateT0.npy', np.array(evaluate_rate_T))
                np.save('TSSGrateS0.npy', np.array(evaluate_rate_S))
                    

    #single test session for drawing
    col_record = np.array([])
    UE_record = np.array([])
    UL_record = np.array([])
    UJ_record = np.array([])
    rateE_record = np.array([])
    rateT_record = np.array([])
    rateS_record = np.array([])
    pT_record = np.array([])
    UEtemp = 0.0
    ULtemp = 0.0
    UJtemp = 0.0
    topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
    LUs = copy.deepcopy(org_LUs)
    Jams = copy.deepcopy(org_Jams)
    BSs = copy.deepcopy(org_BSs)
    Eves = copy.deepcopy(org_Eves)
    done = False
    col_part = [0, 0]
    s = gene_state(topo_h, topo_P, col_part, device)
    if args.use_state_norm:
        s = state_norm(s)
    episode_steps = 0
    while not done:
        rateEtemp = 0.0
        rateTtemp = 0.0
        rateStemp = 0.0
        pTtemp = 0.0
        for i in range(0,LUnum):
            rateEtemp += max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
            rateTtemp += topo_R[i,LUs[i].bs]
            rateStemp += topo_R[i,LUs[i].bs]-max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
            pTtemp += LUs[i].P
        rateE_record = np.append(rateE_record,rateEtemp)
        rateT_record = np.append(rateT_record,rateTtemp)
        rateS_record = np.append(rateS_record,rateStemp)
        pT_record = np.append(pT_record,pTtemp)
        UE_record = np.append(UE_record,UEtemp)
        UL_record = np.append(UL_record,ULtemp)
        UJ_record = np.append(UJ_record,UJtemp)
        col_record = np.append(col_record,col_part)
        if episode_steps >= args.max_episode_steps:
            done = True
        episode_steps += 1
        a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
        topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
        E0s_ = gene_state(topo_h, topo_P, col_part, device)
        if args.use_state_norm:
            E0s_ = state_norm(E0s_)
        a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
        a_LUs = []
        a_LUslogprob = []
        for i in range(0, LUnum):
            tempa, templp = LUagents[i].choose_action(E0s_)
            a_LUs.append(tempa)
            a_LUslogprob.append(templp)
        topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
        E1s_ = gene_state(topo_h, topo_P, col_part, device)
        if args.use_state_norm:
            E1s_ = state_norm(E1s_)
        a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
        
        E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
        a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
        
        
        topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
        topo_R, topo_R0 = update_rate(topo_h, topo_P)
        col_part_, r_E, r_L, r_L_sum, r_J = col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        UEtemp+=r_E
        ULtemp+=r_L_sum
        UJtemp+=r_J
        col_part = col_part_
        s_ = gene_state(topo_h, topo_P, col_part, device)
        if args.use_state_norm:
            s_ = state_norm(s_)
        # col_now, V_E, V_L, V_L.sum(), UJ
        s = s_
        
    # np.save('Single_col_record.npy', col_record)
    # np.save('Single_UE_record.npy', UE_record)
    # np.save('Single_UL_record.npy', UL_record)
    # np.save('Single_UJ_record.npy', UJ_record)
    # np.save('Single_rateE_record.npy', rateE_record)
    # np.save('Single_rateT_record.npy', rateT_record)
    # np.save('Single_rateS_record.npy', rateS_record)
    # np.save('Single_pT_record.npy', pT_record)
    
    np.save('Single_col_record0.npy', col_record)
    np.save('Single_UE_record0.npy', UE_record)
    np.save('Single_UL_record0.npy', UL_record)
    np.save('Single_UJ_record0.npy', UJ_record)
    np.save('Single_rateE_record0.npy', rateE_record)
    np.save('Single_rateT_record0.npy', rateT_record)
    np.save('Single_rateS_record0.npy', rateS_record)
    np.save('Single_pT_record0.npy', pT_record)
    return Eveagent_open, Eveagent_mu, LUagents, LUagent_mu, Jamagent


if __name__ == "__main__" :
    
    args = ARGS()
    Eveagent_open, Eveagent_mu, LUagents, LUagent_mu, Jamagent = main(args, LUs, Jams, BSs, Eves)
    np.save('Eveagent_open.npy',np.array([Eveagent_open]))                
    np.save('Eveagent_mu.npy',np.array([Eveagent_mu]))
    np.save('LUagents.npy',LUagents)
    np.save('LUagent_mu.npy',np.array([LUagent_mu]))
    np.save('Jamagent.npy',np.array([Jamagent]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")