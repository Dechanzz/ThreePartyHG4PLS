import numpy as np
import random
import math
import torch
import copy
import matplotlib
import matplotlib.pyplot as plt

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

class PLS:
    def __init__(self, useold = True):
        self.LUnum = 20
        self.Jamnum = 2
        self.BSnum = 5
        self.Evenum = 5
        self.LUs = np.array([])
        self.Jams = np.array([])
        self.BSs = np.array([])
        self.Eves = np.array([])
        self.a_size=[1000,1000] #area size (m^2)
        self.tslot = 0.5 #time slot (s)
        #self.PT=np.array([0.0,0.025,0.05,0.075,0.1])
        self.PT=np.array([0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
        #self.PJ=np.array([0.0,0.025,0.05,0.075,0.1])
        self.PJ=np.array([0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
        self.W=10000000 #channel bandwidth (Hz)
        self.R_min = 2 # minimum QoS requirment of single LU (bps/Hz)
        self.v_UAV = 10 # velocity of UAV
        self.noise = 0.00000000316 #noise
        self.R_limit = 6
        self.a1 = 1
        self.zeta1 = 1
        self.zeta2 = 10
        self.c_J = 10
        self.c_conf = 0.5
        self.c_eveon = 0.1
        self.c_QoS = 0.5
        self.UAV_disl = 5 #safety distance between UAVs
        if useold: #else先不做了
            np.load.__defaults__ = (None, True, True, 'ASCII')
            self.LUs = np.load('orgLUs.npy')
            self.Jams = np.load('orgJams.npy')
            self.BSs = np.load('orgBSs.npy')
            self.Eves = np.load('orgEves.npy')
            np.load.__defaults__ = (None, False, True, 'ASCII')
            # print("LUnum:", len(self.LUs))
            # topo_limit = self.maxtrans_rates()
            # self.gene_action(topo_limit, LUs)
            for i in self.LUs:
                i.P = self.PT[np.random.randint(1, len(self.PT))]
                # i.bs = int(i.bsrange[np.random.randint(0, len(i.bsrange))])
                i.bs = np.random.randint(1, self.BSnum)
            for i in self.Jams:
                i.P = self.PJ[np.random.randint(1, len(self.PJ))]
        self.topo_h, self.topo_P, self.topo_R, self.topo_R0 = self.init_net_topo(self.LUs, self.Jams, self.BSs, self.Eves)


    def initloc2D(self, n, area_size):  # create num locs in map of area_size
        x = area_size[0]
        y = area_size[1]
        tr = np.array([])
        for i in range(0, n):
            tx = np.random.randint(0, x)
            ty = np.random.randint(0, y)
            tr = np.append(tr, np.array([tx, ty]))
        return tr

    def cacu_cgain(self, devx, devy):  # return channel gain of devx and devy
        locx = devx.loc
        locy = devy.loc
        return (np.linalg.norm(locx - locy)) ** (-2)

    def dev_index(self, ind, LUs, Jams, BSs, Eves, slct=-1):  # serch correspond devx and devy of index ind, if slct = 0: search sender, slct = 1: search receiver
        if len(ind) == 2:
            if ind[0] > self.LUnum - 1:
                devx = Jams[ind[0] - self.LUnum]
            else:
                devx = LUs[ind[0]]
            if ind[1] > self.BSnum - 1:
                devy = Eves[ind[1] - self.BSnum]
            else:
                devy = BSs[ind[1]]
            return devx, devy
        elif len(ind) == 1:
            if slct == 0:
                if ind[0] > self.LUnum - 1:
                    devx = Jams[ind[0] - self.LUnum]
                else:
                    devx = LUs[ind[0]]
            elif slct == 1:
                if ind[0] > self.BSnum - 1:
                    devx = Eves[ind[0] - self.BSnum]
                else:
                    devx = BSs[ind[0]]
            return devx

    def cacu_rate(self, wid, sig, nois):
        return wid * math.log2(1 + sig / nois)

    def init_net_topo(self, LUs, Jams, BSs, Eves):  # initialize the network topology graph of channel gain and power allocation and rates
        tnet_h = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # h_ij
        tnet_P = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # P^T and P^J
        tnet_R = np.zeros([self.LUnum, self.BSnum + self.Evenum])  # R^T and R^E
        tnet_R0 = np.zeros([self.LUnum, self.BSnum + self.Evenum])  # R without jamming
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
                tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
                if j < self.BSnum:
                    if j == tdx.bs:
                        tnet_P[i][j] = tdx.P
                elif j >= self.BSnum:
                    tnet_P[i][
                        j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        for i in range(self.LUnum, self.LUnum + self.Jamnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
                tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
                tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                if tnet_P[i][j] != 0.0:
                    pJg = 0.0
                    for k in range(self.LUnum, self.LUnum + self.Jamnum):
                        pJg += tnet_P[k][j] * tnet_h[k][j]
                    tnet_R[i][j] = math.log2(1 + tnet_P[i][j] * tnet_h[i][j] / (self.noise + pJg))
                    tnet_R0[i][j] = math.log2(1 + tnet_P[i][j] * tnet_h[i][j] / (self.noise))
        return tnet_h, tnet_P, tnet_R, tnet_R0

    def update_rate(self, topoh, topoP):
        tnet_R = np.zeros([self.LUnum, self.BSnum + self.Evenum])  # R^T and R^E
        tnet_R0 = np.zeros([self.LUnum, self.BSnum + self.Evenum])  # R without jamming
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                if topoP[i][j] != 0.0:
                    pJg = 0.0
                    for k in range(self.LUnum, self.LUnum + self.Jamnum):
                        pJg += topoP[k][j] * topoh[k][j]
                    tnet_R[i][j] = math.log2(1 + topoP[i][j] * topoh[i][j] / (self.noise + pJg))
                    tnet_R0[i][j] = math.log2(1 + topoP[i][j] * topoh[i][j] / (self.noise))
        return tnet_R, tnet_R0

    def maxtrans_rates(self):  #
        tnet_R = np.zeros([self.LUnum, self.BSnum])  # R^T and R^E
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                tnet_R[i][j] = math.log2(1 + self.PT[-1] * self.cacu_cgain(tdx, tdy) / self.noise)
        return tnet_R

    def gene_action(self,topo_limit, LUs):
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum):
                LUs[i].bsrange = np.append(LUs[i].bsrange, j)

    def testnum(self, ary, lim):
        tres = 0
        for i in ary:
            if i >= lim:
                tres += 1
        return tres

    # environment
    def gene_state(self, topo_h, topo_P, col_parti, device):
        rtrn_state = np.append((topo_h * topo_P).flatten(), np.array(col_parti))
        rtrn_state = torch.tensor(rtrn_state).to(device)
        return rtrn_state

    def gene_state_J(self, topo_h, topo_P, col_parti, mu1, mu2, device):
        rtrn_state = np.append((topo_h * topo_P).flatten(), np.array(col_parti))
        rtrn_state = np.append(rtrn_state, [mu1, mu2])
        rtrn_state = torch.tensor(rtrn_state).to(device)
        return rtrn_state

    def action_trans_Eve(self, action_code, LUs, Jams, BSs, Eves):  # transform the Eve actions code to real actions
        # action code is the action chosen by actor network
        # Evesdroppers have 4*4 action, the code from 0~15
        a_str = bin(int(action_code))[2:].zfill(self.Evenum)
        for i in range(0, self.Evenum):
            if a_str[i] == '0':
                Eves[i].open = 0
            elif a_str[i] == '1':
                Eves[i].open = 1
        tnet_h = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # h_ij
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
                tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
        for i in range(self.LUnum, self.LUnum + self.Jamnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
                tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
        return tnet_h  # updated topo_h after Eves' actions

    def action_trans_LU(self, action_codes):  # transform the LU action codes to real actions
        # action codes are the action arry of length len(LUs) chosen by actor networks
        # single LU has actions of len(LU.bsrange)*len(PT)
        for i in range(0, self.LUnum):
            a_BS = int(action_codes[i] / len(self.PT))
            a_PT = action_codes[i] % len(self.PT)
            if a_BS >= len(self.LUs[i].bsrange):
                a_BS = len(self.LUs[i].bsrange) - 1
            elif a_BS <= 0:
                a_BS = 0
            if a_PT >= len(self.PT):
                a_PT = len(self.PT)
            elif a_PT <= 0:
                a_PT = 0
            self.LUs[i].bs = int(self.LUs[i].bsrange[a_BS])
            self.LUs[i].P = self.PT[int(a_PT)]
        tnet_P = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # P^T and P^J
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                if j < self.BSnum:
                    if j == tdx.bs:
                        tnet_P[i][j] = tdx.P
                elif j >= self.BSnum:
                    tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        for i in range(self.LUnum, self.LUnum + self.Jamnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        return tnet_P  # update topo_P after LUs' actions

    def gene_revenue_Eve(self, topoR, topoR0):  # caculate V_Eve and V_Eve^0 for all Eves
        tV = 0.0
        tV0 = 0.0
        for i in range(0, self.LUnum):
            tV += max(topoR[i, self.BSnum] * self.Eves[0].open, topoR[i, self.BSnum + 1] * self.Eves[1].open) - topoR[i, self.LUs[i].bs]
            tV0 += max(topoR0[i, self.BSnum] * self.Eves[0].open, topoR0[i, self.BSnum + 1] * self.Eves[1].open) - topoR0[i, self.LUs[i].bs]
        tV *= self.a1
        tV0 *= self.a1
        return tV, tV0, self.Eves[0].open * self.c_eveon * self.tslot + self.Eves[1].open * self.c_eveon * self.tslot

    def gene_revenue_LU(self, topoR, topoR0):  # caculate V_LU and V_LU^0 for single LU agent and LUs agent
        tV = np.array([])
        tV0 = np.array([])
        pcostL = np.array([])
        for i in range(0, self.LUnum):
            tV = np.append(tV, self.zeta1 * (topoR[i, self.LUs[i].bs] - max(topoR[i, self.BSnum] * self.Eves[0].open, topoR[i, self.BSnum + 1] * self.Eves[1].open)))
            tV0 = np.append(tV0, self.zeta1 * (topoR0[i, self.LUs[i].bs] - max(topoR0[i, self.BSnum] * self.Eves[0].open, topoR0[i, self.BSnum + 1] * self.Eves[1].open)))
            if topoR[i, self.LUs[i].bs] < self.R_min:
                QoScost = self.c_QoS * (topoR[i, self.LUs[i].bs] - self.R_min)
                tV += QoScost
                tV0 += QoScost
            pcostL = np.append(pcostL, self.zeta2 * self.LUs[i].P)
        return tV, tV0, pcostL

    def action_trans_Jam(self, action_code):  # transform the Jams' actions code to real actions
        # action code is the action chosen by actor network
        # Jammers have 5*5 action, the code from 0~24
        a0 = int(action_code / len(self.PJ))  # action of Jam1
        a1 = action_code % len(self.PJ)  # action of Jam2
        self.Jams[0].P = self.PJ[a0]
        self.Jams[1].P = self.PJ[int(a1)]
        tnet_P = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # P^T and P^J
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                if j < self.BSnum:
                    if j == tdx.bs:
                        tnet_P[i][j] = tdx.P
                elif j >= self.BSnum:
                    tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        for i in range(self.LUnum, self.LUnum + self.Jamnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        return tnet_P  # update topo_P after LUs' actions

    def col_formation(self, mu_E, mu_L, col_old, topo_R, topo_R0, LUs, Jams, Eves):  # the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
        V_E, V_E0, pcostE = self.gene_revenue_Eve(topo_R, topo_R0, LUs)
        V_L, V_L0, pcostL = self.gene_revenue_LU(topo_R, topo_R0, LUs)
        if Jams[0].P + Jams[1].P == 0:
            col_now = [0, 0]
            if col_old == [0, 0]:
                UJ = 0.0
            else:
                UJ = -self.c_conf
            return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ
        sV_L = V_L.sum()
        sV_L0 = V_L0.sum()
        if col_old[0] == 1:
            UJE = mu_E * (V_E - V_E0) - self.c_J * (Jams[0].P + Jams[1].P)
        else:
            UJE = mu_E * (V_E - V_E0) - self.c_J * (Jams[0].P + Jams[1].P) - self.c_conf
        if col_old[1] == 1:
            UJL = mu_L * (sV_L - sV_L0) - self.c_J * (Jams[0].P + Jams[1].P)
        else:
            UJL = mu_L * (sV_L - sV_L0) - self.c_J * (Jams[0].P + Jams[1].P) - self.c_conf
        if UJL >= UJE and UJL >= 0:
            col_now = [0, 1]
            UJ = UJL
            V_L = V_L - mu_L * (V_L - V_L0)
            # print("incentive from LUs:",UJ)
        elif UJL < UJE and UJE >= 0:
            col_now = [1, 0]
            UJ = UJE
            V_E = V_E - mu_E * (V_E - V_E0)
            # print("incentive from Eves:",UJ)
        elif UJL <= 0 and UJE <= 0:
            col_now = [0, 0]
            if col_old == [0, 0]:
                UJ = 0
            else:
                UJ = -self.c_conf
        return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ

    def col_formation_PE(self, mu_E):  # the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
        V_E, V_E0, pcostE = self.gene_revenue_Eve(self.topo_R, self.topo_R0)
        V_L, V_L0, pcostL = self.gene_revenue_LU(self.topo_R, self.topo_R0)
        # unit incentive cannot be calculated by DNN, thus we use a fixed one
        mu_E = 0.01 # can be changed
        if V_E - V_E0 >=0:
            UJE = mu_E * (V_E - V_E0) - self.c_J * (self.Jams[0].P + self.Jams[1].P)
        else:
            UJE = -1 * self.c_J * (self.Jams[0].P + self.Jams[1].P)
        col_now = [1, 0]
        UJ = UJE
        V_E = V_E - mu_E * (V_E - V_E0)
        return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ

    def col_formation_AN(self, mu_L):  # the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
        V_E, V_E0, pcostE = self.gene_revenue_Eve(self.topo_R, self.topo_R0)
        V_L, V_L0, pcostL = self.gene_revenue_LU(self.topo_R, self.topo_R0)
        # unit incentive cannot be calculated by DNN, thus we use a fixed one
        mu_L = 0.01 # can be changed
        sV_L = V_L.sum()
        sV_L0 = V_L0.sum()
        if sV_L - sV_L0 >=0 :
            UJL = mu_L * (sV_L - sV_L0) - self.c_J * (self.Jams[0].P + self.Jams[1].P)
        else:
            UJL = - self.c_J * (self.Jams[0].P + self.Jams[1].P)
        col_now = [0, 1]
        UJ = UJL
        V_L = V_L - mu_L * (V_L - V_L0)
        return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ

    def LU_utility(self, xLU):  # caculate V_LU and V_LU^0 for single LU agent and LUs agent
        LFJ = False
        tempLU = np.zeros((self.LUnum,2))
        for i in range(0, self.LUnum):
            a_BS = int(int(xLU[i]) / len(self.PT))
            if a_BS >= len(self.LUs[i].bsrange):
                a_BS = len(self.LUs[i].bsrange)-1
            elif a_BS <= 0:
                a_BS = 0
            a_PT = int(xLU[i]) % len(self.PT)
            if a_PT >= len(self.PT):
                a_PT = len(self.PT)
            elif a_PT <= 0:
                a_PT = 0
            tempLU[i][0] = int(self.LUs[i].bsrange[a_BS])
            tempLU[i][1] = self.PT[int(a_PT)]
        tnet_P = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # P^T and P^J
        for i in range(0, self.LUnum):
            for j in range(0, self.BSnum + self.Evenum):
                # tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                if j < self.BSnum:
                    if j == tempLU[i][0]:
                        tnet_P[i][j] = tempLU[i][1]
                elif j >= self.BSnum:
                    tnet_P[i][j] = tempLU[i][1]  # power allocation only include the actual power allocated to corresponding channels
        for i in range(self.LUnum, self.LUnum + self.Jamnum):
            for j in range(0, self.BSnum + self.Evenum):
                tdx, tdy = self.dev_index([i, j], self.LUs, self.Jams, self.BSs, self.Eves)
                tnet_P[i][j] = tdx.P  # power allocation only include the actual power allocated to corresponding channels
        temptopoR, temptopoR0  = self.update_rate(self.topo_h, tnet_P)
        tV = np.array([])
        tV0 = np.array([])
        pcostL = np.array([])
        for i in range(0, self.LUnum):
            tV = np.append(tV, self.zeta1 * (temptopoR[i, self.LUs[i].bs] - max(temptopoR[i, self.BSnum] * self.Eves[0].open, temptopoR[i, self.BSnum + 1] * self.Eves[1].open)))
            tV0 = np.append(tV0, self.zeta1 * (temptopoR0[i, self.LUs[i].bs] - max(temptopoR0[i, self.BSnum] * self.Eves[0].open, temptopoR0[i, self.BSnum + 1] * self.Eves[1].open)))
            if temptopoR[i, self.LUs[i].bs] < self.R_min:
                QoScost = self.c_QoS * (temptopoR[i, self.LUs[i].bs] - self.R_min)
                tV += QoScost
                tV0 += QoScost
            pcostL = np.append(pcostL, self.zeta2 * self.LUs[i].P)
        if not LFJ:
            return np.sum(tV-pcostL)

    def Eve_utility(self, xEve):  # caculate V_Eve and V_Eve^0 for all Eves
        topoR = self.topo_R
        topoR0 = self.topo_R0
        EFJ = False
        Eaction_code = int(xEve)
        temptopoh = self.action_trans_Eve(Eaction_code, self.LUs, self.Jams, self.BSs, self.Eves)
        a_str = bin(int(xEve))[2:].zfill(self.Evenum)
        Etemp = np.zeros(self.Evenum)
        for i in range(0, self.Evenum):
            if a_str[i] == '0':
                Etemp[i] = 0
            elif a_str[i] == '1':
                Etemp[i] = 1
        # tnet_h = np.zeros([self.LUnum + self.Jamnum, self.BSnum + self.Evenum])  # h_ij
        # for i in range(0, self.LUnum):
        #     for j in range(0, self.BSnum + self.Evenum):
        #         tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
        #         tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
        # for i in range(self.LUnum, self.LUnum + self.Jamnum):
        #     for j in range(0, self.BSnum + self.Evenum):
        #         tdx, tdy = self.dev_index([i, j], LUs, Jams, BSs, Eves)
        #         tnet_h[i][j] = self.cacu_cgain(tdx, tdy)  # channel gain include all channel gains
        # #return tnet_h  # updated topo_h after Eves' actions
        tV = 0.0
        tV0 = 0.0
        for i in range(0, self.LUnum):
            tV += max(topoR[i, self.BSnum] * Etemp[0], topoR[i, self.BSnum + 1] * Etemp[1]) - topoR[i, self.LUs[i].bs]
            tV0 += max(topoR0[i, self.BSnum] * Etemp[0], topoR0[i, self.BSnum + 1] * Etemp[1]) - topoR0[i, self.LUs[i].bs]
        tV *= self.a1
        tV0 *= self.a1
        if not EFJ:
            return tV-self.Eves[0].open * self.c_eveon * self.tslot + self.Eves[1].open * self.c_eveon * self.tslot
        # return tV, tV0, self.Eves[0].open * self.c_eveon * self.tslot + self.Eves[1].open * self.c_eveon * self.tslot


    # def JA_utility(self, mu_E, mu_L, col_old, topo_R, topo_R0, LUs, Jams, Eves):  # the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    #     V_E, V_E0, pcostE = self.gene_revenue_Eve(topo_R, topo_R0, LUs)
    #     V_L, V_L0, pcostL = self.gene_revenue_LU(topo_R, topo_R0, LUs)
    #     if Jams[0].P + Jams[1].P == 0:
    #         col_now = [0, 0]
    #         if col_old == [0, 0]:
    #             UJ = 0.0
    #         else:
    #             UJ = -self.c_conf
    #         return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ
    #     sV_L = V_L.sum()
    #     sV_L0 = V_L0.sum()
    #     if col_old[0] == 1:
    #         UJE = mu_E * (V_E - V_E0) - self.c_J * (Jams[0].P + Jams[1].P)
    #     else:
    #         UJE = mu_E * (V_E - V_E0) - self.c_J * (Jams[0].P + Jams[1].P) - self.c_conf
    #
    #     if col_old[1] == 1:
    #         UJL = mu_L * (sV_L - sV_L0) - self.c_J * (Jams[0].P + Jams[1].P)
    #     else:
    #         UJL = mu_L * (sV_L - sV_L0) - self.c_J * (Jams[0].P + Jams[1].P) - self.c_conf
    #
    #     if UJL >= UJE and UJL >= 0:
    #         col_now = [0, 1]
    #         UJ = UJL
    #         V_L = V_L - mu_L * (V_L - V_L0)
    #         # print("incentive from LUs:",UJ)
    #     elif UJL < UJE and UJE >= 0:
    #         col_now = [1, 0]
    #         UJ = UJE
    #         V_E = V_E - mu_E * (V_E - V_E0)
    #         # print("incentive from Eves:",UJ)
    #     elif UJL <= 0 and UJE <= 0:
    #         col_now = [0, 0]
    #         if col_old == [0, 0]:
    #             UJ = 0
    #         else:
    #             UJ = -self.c_conf
    #
    #     return col_now, V_E - pcostE, V_L - pcostL, V_L.sum() - pcostL.sum(), UJ