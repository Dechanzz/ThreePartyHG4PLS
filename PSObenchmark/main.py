import numpy as np
import random
import math
import copy
import matplotlib
import matplotlib.pyplot as plt
from argsettings import ARGS
from pso import PSO
from plsenv import PLS, LU, BS, Eve, Jammer
import copy
import numpy as np
import torch
import math
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from ppo_continuous import PPO_continuous
from argsettings import ARGS
import matplotlib
import matplotlib.pyplot as plt

def fitness(x):
    """
    根据粒子位置计算适应值，可根据问题情况自定义
    """
    return x[0] * np.exp(x[1]) - x[2] * np.sin(x[1]) - x[3] * x[4]



if __name__ == "__main__":
    # args = ARGS(fitness)
    # pso = PSO(args)
    # pso.pso()
    # print(pso.g_best)
    # print(fitness([25.,25.,25.,25.,25.]))
    # print(fitness([25., 25., 1., 25., 25.]))
    # print(fitness([1., 1., 1., 1., 1.]))

    np.load.__defaults__ = (None, True, True, 'ASCII')
    # LUs = np.load('orgLUs.npy')
    # Jams = np.load('orgJams.npy')
    # BSs = np.load('orgBSs.npy')
    # Eves = np.load('orgEves.npy')
    Eveagent_open = np.load('Eveagent_open.npy')
    Eveagent_mu = np.load('Eveagent_mu.npy')
    LUagents = np.load('LUagents.npy')
    LUagent_mu = np.load('LUagent_mu.npy')
    Jamagent = np.load('Jamagent.npy')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    args = ARGS(fitness)
    pls0 = PLS()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #LU-Friendly JA   EV using PSO
    col_part_AN = [0, 1]
    EVutility_record = np.array([])
    LUutility_record = np.array([])
    LUutility_record_sum = np.array([])
    JAutility_record = np.array([])
    EVaction_record = np.array([])

    for i in range(0, 201):
        args.fitness = pls0.Eve_utility
        args.D = 1
        args.p_low = [1.0]  # 粒子位置的约束范围
        args.p_up = [32.0]
        temp_pso = PSO(args)
        temp_pso.pso()
        EVaction_record = np.append(EVaction_record, int(temp_pso.g_best))
        pls0.topo_h = pls0.action_trans_Eve(int(temp_pso.g_best), pls0.LUs, pls0.Jams, pls0.BSs, pls0.Eves)

        state0 = pls0.gene_state(pls0.topo_h, pls0.topo_P, col_part_AN, device)
        a_LUs = []
        for j in range(0, pls0.LUnum):
            tempa, templp = LUagents[j].choose_action(state0)
            a_LUs.append(tempa)
        pls0.topo_P = pls0.action_trans_LU(a_LUs)
        state1 = pls0.gene_state(pls0.topo_h, pls0.topo_P, col_part_AN, device)
        a_LUmu, a_LUmulp = LUagent_mu[0].choose_action(state1)
        state2 = pls0.gene_state_J(pls0.topo_h, pls0.topo_P, col_part_AN, 0.0, a_LUmu, device)
        a_Jam, a_Jamlp = Jamagent[0].choose_action(state2)
        pls0.topo_P = pls0.action_trans_Jam(a_Jam)
        pls0.topo_R, pls0.topo_R0 = pls0.update_rate(pls0.topo_h, pls0.topo_P)
        col_part_AN, r_E, r_L, r_L_sum, r_J = pls0.col_formation_AN(a_LUmu)
        EVutility_record = np.append(EVutility_record, r_E)
        LUutility_record = np.append(LUutility_record, r_L)
        LUutility_record_sum = np.append(LUutility_record_sum, r_L_sum)
        JAutility_record = np.append(JAutility_record, r_J)

    RE_AN = np.cumsum(EVutility_record)
    RJ_AN = np.cumsum(JAutility_record)
    np.save("sgEVutilANPSO.npy", EVutility_record)
    np.save("sgJAutilANPSO.npy", JAutility_record)
    np.save("EVutilANPSO.npy", RE_AN)
    np.save("JAutilANPSO.npy", RJ_AN)



    np.load.__defaults__ = (None, True, True, 'ASCII')
    # LUs = np.load('orgLUs.npy')
    # Jams = np.load('orgJams.npy')
    # BSs = np.load('orgBSs.npy')
    # Eves = np.load('orgEves.npy')
    Eveagent_open = np.load('Eveagent_open.npy')
    Eveagent_mu = np.load('Eveagent_mu.npy')
    LUagents = np.load('LUagents.npy')
    LUagent_mu = np.load('LUagent_mu.npy')
    Jamagent = np.load('Jamagent.npy')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    args = ARGS(fitness)
    pls0 = PLS()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # EV-Friendly JA   LU using PSO
    col_part_PE = [0, 1]
    EVutility_record = np.array([])
    LUutility_record = np.array([])
    LUutility_record_sum = np.array([])
    JAutility_record = np.array([])
    LUaction_record = np.array([])

    for i in range(0, 201):
        # EVaction_record = np.append(EVaction_record, int(temp_pso.g_best))
        state0 = pls0.gene_state(pls0.topo_h, pls0.topo_P, col_part_PE, device)
        a_EVs, a_EVslp = Eveagent_open[0].choose_action(state0)
        pls0.topo_h = pls0.action_trans_Eve(a_EVs, pls0.LUs, pls0.Jams, pls0.BSs, pls0.Eves)

        args.fitness = pls0.LU_utility
        args.D = pls0.LUnum
        args.p_low = []
        args.p_up = []
        for j in range(0, pls0.LUnum):
            temp = len(pls0.LUs[j].bsrange) * len(pls0.PT)
            args.p_low.append(0.0)  # 粒子位置的约束范围
            args.p_up.append(temp)
        temp_pso = PSO(args)
        temp_pso.pso()

        a_LUs = []
        for j in temp_pso.g_best:
            a_LUs.append(int(j))
        pls0.topo_P = pls0.action_trans_LU(a_LUs)

        state1 = pls0.gene_state(pls0.topo_h, pls0.topo_P, col_part_PE, device)
        a_LUmu, a_LUmulp = LUagent_mu[0].choose_action(state1)
        state2 = pls0.gene_state_J(pls0.topo_h, pls0.topo_P, col_part_PE, 0.0, a_LUmu, device)
        a_Jam, a_Jamlp = Jamagent[0].choose_action(state2)
        pls0.topo_P = pls0.action_trans_Jam(a_Jam)
        pls0.topo_R, pls0.topo_R0 = pls0.update_rate(pls0.topo_h, pls0.topo_P)
        col_part_PE, r_E, r_L, r_L_sum, r_J = pls0.col_formation_PE(a_LUmu)
        EVutility_record = np.append(EVutility_record, r_E)
        LUutility_record = np.append(LUutility_record, r_L)
        LUutility_record_sum = np.append(LUutility_record_sum, r_L_sum)
        JAutility_record = np.append(JAutility_record, r_J)

    RL_PE = np.cumsum(LUutility_record_sum)
    RJ_PE = np.cumsum(JAutility_record)
    np.save("sgLUutilPEPSO.npy", LUutility_record_sum)
    np.save("LUutilPEPSO.npy", RL_PE)
    np.save("sgJAUutilPEPSO.npy", JAutility_record)
    np.save("JAutilPEPSO.npy", RJ_PE)
    # pso = PSO(args)
    # pso.pso()
    # draw

