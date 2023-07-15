#This file is a supplimentary code for comparing the parameter influences in simulation, please:
#option 1: add the following code ../main.py, and reorganize the code for running;
#option 2: use consloe mode in PyCharm, after running ../main.py, run these codes in console
#option 3: use scientific mode in PyCharm
#option 4 (most recommanded): run all codes in Spyder, while #%% needs to be used for sepreating the codes

#%% power consumption of LU vs zeta2
#use gene_revenue_LU
def TE_alloc(topoh, topoP):
    finalp = 0.0
    for i in range(0,LUnum):
        ptemp = 0.0
        TEmax = 0.0
        for p in PT:           
            for j in range(0,BSnum):
                pJg = 0.0
                for k in range(LUnum,LUnum+Jamnum):
                    pJg += topoP[k][j]*topoh[k][j]
                RT = math.log2(1+p*topoh[i][j]/(noise+pJg))                              
                TE = RT/p
                if TE>TEmax:
                    TEmax = TE
                    ptemp = p
        finalp+=ptemp       
    return finalp

def SG_alloc(topoh, topoP):
    finalp = 0.0
    for i in range(0,LUnum):
        ptemp = 0.0
        RSmax = 0.0
        for p in PT:           
            for j in range(0,BSnum):
                pJg = 0.0
                for k in range(LUnum,LUnum+Jamnum):
                    pJg += topoP[k][j]*topoh[k][j]
                RT = math.log2(1+p*topoh[i][j]/(noise+pJg))  
                REmax = 0.0
                for e in range(BSnum,BSnum+Evenum):
                    pJg = 0.0
                    for k in range(LUnum,LUnum+Jamnum):
                        pJg += topoP[k][e]*topoh[k][e]
                    RE = math.log2(1+p*topoh[i][e]/(noise+pJg))
                    if RE>REmax:
                        REmax = RE
                RS = RT-REmax
                if RS>RSmax:
                    RSmax = RS
                    ptemp = p
        finalp+=ptemp       
    return finalp

def SE_alloc(topoh, topoP):
    finalp = 0.0
    for i in range(0,LUnum):
        ptemp = 0.0
        SEmax = 0.0
        for p in PT:           
            for j in range(0,BSnum):
                pJg = 0.0
                for k in range(LUnum,LUnum+Jamnum):
                    pJg += topoP[k][j]*topoh[k][j]
                RT = math.log2(1+p*topoh[i][j]/(noise+pJg))  
                REmax = 0.0
                for e in range(BSnum,BSnum+Evenum):
                    pJg = 0.0
                    for k in range(LUnum,LUnum+Jamnum):
                        pJg += topoP[k][e]*topoh[k][e]
                    RE = math.log2(1+p*topoh[i][e]/(noise+pJg))
                    if RE>REmax:
                        REmax = RE
                RS = RT-REmax
                SE = RS/p
                if SE>SEmax:
                    SEmax = SE
                    ptemp = p
        finalp+=ptemp       
    return finalp

c_conf = 0.5
R_min = 2

pcost = [0.0,0.0,0.0,0.0,0.0,0.0]
pcosttemp = 0.0

pcost_TG = [0.0,0.0,0.0,0.0,0.0,0.0]
pcosttemp_TG = 0.0

pcost_TE = [0.0,0.0,0.0,0.0,0.0,0.0]
pcosttemp_TE = 0.0

pcost_SG = [0.0,0.0,0.0,0.0,0.0,0.0]
pcosttemp_SG = 0.0

pcost_SE = [0.0,0.0,0.0,0.0,0.0,0.0]
pcosttemp_SE = 0.0

UEtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
ULtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
UJtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
UErec = [[],[],[],[],[],[]]
ULrec = [[],[],[],[],[],[]]
UJrec = [[],[],[],[],[],[]]

rateE_record = np.array([])
rateT_record = np.array([])
rateS_record = np.array([])


topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]

s = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0

while not done:
    
    rateEtemp = 0.0
    rateTtemp = 0.0
    rateStemp = 0.0
    # pTtemp = 0.0
    # pJtemp = 0.0
    
    for i in range(0,LUnum):
        rateEtemp += max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
        rateTtemp += topo_R[i,LUs[i].bs]
        rateStemp += topo_R[i,LUs[i].bs]-max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
        
    rateE_record = np.append(rateE_record,rateEtemp)
    rateT_record = np.append(rateT_record,rateTtemp)
    rateS_record = np.append(rateS_record,rateStemp)

    if episode_steps >= args.max_episode_steps:
        done = True
    episode_steps += 1
    
    #TSSG
    a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
    topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
    E0s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
    Emu_record = np.append(Emu_record,a_Emu)
    a_LUs = []
    a_LUslogprob = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s_)
        a_LUs.append(tempa)
        a_LUslogprob.append(templp)
    topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
    E1s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
    Lmu_record = np.append(Lmu_record,a_Lmu)
    E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
    a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
    topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
    topo_R, topo_R0 = update_rate(topo_h, topo_P)
    
    zeta2 = 5
    
    pTE=TE_alloc(topo_h,topo_P)
    pSG=SG_alloc(topo_h,topo_P)
    pSE=SE_alloc(topo_h,topo_P)
    
    for i in range(0,6):
        taa, tbb, pcosttemp = gene_revenue_LU(topo_R,topo_R0, LUs)
        pcost[i]+=pcosttemp.sum()
        
        pcosttemp_TG = 0.1*zeta2*LUnum
        pcost_TG[i]+=pcosttemp_TG
              
        pcost_TE[i]+=pTE*zeta2
        pcost_SG[i]+=pSG*zeta2
        pcost_SE[i]+=pSE*zeta2
                
        col_part_, r_E, r_L, r_L_sum, r_J = col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        UEtemp[i]+=r_E
        ULtemp[i]+=r_L_sum
        UJtemp[i]+=r_J
        UErec[i].append(UEtemp[i])
        ULrec[i].append(ULtemp[i])
        UJrec[i].append(UJtemp[i])
                
        zeta2+=2
    
    col_part = col_part_
    
    #TSSG
    tempmes = mesmerize(topo_h)
    topo_h*=tempmes
    s_ = gene_state(topo_h, topo_P, col_part, device)
    s = s_

#%% power consumption of EVs vs ck
#use gene_revenue_Eve
c_eveon = 0.1
c_conf = 0.5
R_min = 2
def REE_alloc(topoh, topoP):
    Enum = np.zeros([1,Evenum])
    finalnum = 0     
    for i in range(0,LUnum):
        flag = 0
        REmax = 0.0  
        for j in range(BSnum, BSnum+Evenum):
            pJg = 0.0            
            for k in range(LUnum,LUnum+Jamnum):
                pJg += topoP[k][j]*topoh[k][j]
            RE = math.log2(1+topoP[i][j]*topoh[i][j]/(noise+pJg))                              
            if RE>REmax:
                REmax = RE
                flag = j
        Enum[0][flag-BSnum]+=1 
    for i in Enum[0]:
        if i > 0:
            finalnum+=1
    return finalnum

pcost = [0.0,0.0,0.0,0.0,0.0]
pcosttemp = 0.0

pcost_EG = [0.0,0.0,0.0,0.0,0.0]
pcosttemp_EG = 0.0

pcost_REE = [0.0,0.0,0.0,0.0,0.0]
pcosttemp_REE = 0.0

UEtemp = [0.0,0.0,0.0,0.0,0.0]
ULtemp = [0.0,0.0,0.0,0.0,0.0]
UJtemp = [0.0,0.0,0.0,0.0,0.0]
UErec = [[],[],[],[],[]]
ULrec = [[],[],[],[],[]]
UJrec = [[],[],[],[],[]]

rateE_record = np.array([])
rateT_record = np.array([])
rateS_record = np.array([])

topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]

s = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0

while not done:
    
    rateEtemp = 0.0
    rateTtemp = 0.0
    rateStemp = 0.0
    
    for i in range(0,LUnum):
        rateEtemp += max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
        rateTtemp += topo_R[i,LUs[i].bs]
        rateStemp += topo_R[i,LUs[i].bs]-max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
  
    rateE_record = np.append(rateE_record,rateEtemp)
    rateT_record = np.append(rateT_record,rateTtemp)
    rateS_record = np.append(rateS_record,rateStemp)

    if episode_steps >= args.max_episode_steps:
        done = True
    episode_steps += 1
    
    #TSSG
    a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
    topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
    E0s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
    Emu_record = np.append(Emu_record,a_Emu)
    a_LUs = []
    a_LUslogprob = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s_)
        a_LUs.append(tempa)
        a_LUslogprob.append(templp)
    topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
    E1s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
    Lmu_record = np.append(Lmu_record,a_Lmu)
    E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
    a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
    topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
    topo_R, topo_R0 = update_rate(topo_h, topo_P)
    
    c_eveon = 0.1
    
    EVon=REE_alloc(topo_h,topo_P)

    for i in range(0,5):
        taa, tbb, pcosttemp = gene_revenue_Eve(topo_R,topo_R0, LUs)
        pcost[i]+=pcosttemp
        
        pcosttemp_EG = c_eveon*Evenum*tslot
        pcost_EG[i]+=pcosttemp_EG
        
        pcost_REE[i]+=c_eveon*EVon*tslot       
                
        col_part_, r_E, r_L, r_L_sum, r_J = col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        UEtemp[i]+=r_E
        ULtemp[i]+=r_L_sum
        UJtemp[i]+=r_J
        UErec[i].append(UEtemp[i])
        ULrec[i].append(ULtemp[i])
        UJrec[i].append(UJtemp[i])
                
        c_eveon+=0.1
    
    col_part = col_part_
    
    #TSSG
    tempmes = mesmerize(topo_h)
    topo_h*=tempmes
    s_ = gene_state(topo_h, topo_P, col_part, device)

    s = s_

#%% power consumption of JAs vs eta_J (c_J) 
#use gene_revenue_Eve  
def col_formation(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    #print("mu_E:",mu_E)
    #print("mu_L:",mu_L)
    pcostJ = 0.0
    V_E, V_E0, pcostE = gene_revenue_Eve(topo_R,topo_R0, LUs)
    V_L,V_L0, pcostL = gene_revenue_LU(topo_R,topo_R0, LUs)
    if Jams[0].P+Jams[1].P==0:
        col_now = [0,0]
        if col_old==[0,0]:
            UJ=0.0
        else:
            UJ=-c_conf
            pcostJ = c_conf
        return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ, pcostJ
    sV_L = V_L.sum()
    sV_L0 = V_L0.sum()
    
    #print("raw utility of Eves:",V_E-V_E0)
    #print("raw utility of LUs:",sV_L-sV_L0)
    
    if col_old[0]==1:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)
        pcostJE=c_J*(Jams[0].P+Jams[1].P)
    else:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
        pcostJE=(c_J*(Jams[0].P+Jams[1].P)+c_conf)
    if col_old[1]==1:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)
        pcostJL=c_J*(Jams[0].P+Jams[1].P)
    else:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
        pcostJL=(c_J*(Jams[0].P+Jams[1].P)+c_conf)
    if UJL>=UJE and UJL>=0:
        col_now=[0,1]
        UJ=UJL
        V_L = V_L-mu_L*(V_L-V_L0)
        pcostJ = pcostJL
        #print("incentive from LUs:",UJ)
    elif UJL<UJE and UJE>=0:
        col_now=[1,0]
        UJ=UJE
        V_E = V_E-mu_E*(V_E-V_E0)
        pcostJ = pcostJE
        #print("incentive from Eves:",UJ)
    elif UJL<=0 and UJE<=0:
        col_now=[0,0]
        if col_old == [0,0]:
            UJ = 0
        else:
            UJ = -c_conf
            pcostJ = c_conf
    return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ, pcostJ

def JAIM(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    costJ = 0.0
    V_E, V_E0, pcostE = gene_revenue_Eve(topo_R,topo_R0, LUs)
    V_L,V_L0, pcostL = gene_revenue_LU(topo_R,topo_R0, LUs)
    V_JE = 0.0
    V_JL = 0.0
    if Jams[0].P+Jams[1].P==0:
        col_now = [0,0]
        if col_old==[0,0]:
            UJ=0.0
        else:
            UJ=-c_conf
            costJ=c_conf
        return costJ, UJ
    sV_L = V_L.sum()
    sV_L0 = V_L0.sum()
    
    if col_old[0]==1:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)
        V_JE = mu_E*(V_E-V_E0)
        costJE=c_J*(Jams[0].P+Jams[1].P)
    else:
        UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
        V_JE = mu_E*(V_E-V_E0)
        costJE=(c_J*(Jams[0].P+Jams[1].P)+c_conf)             
    if col_old[1]==1:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)
        V_JL = mu_L*(sV_L-sV_L0)
        costJL=c_J*(Jams[0].P+Jams[1].P)
    else:
        UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)-c_conf
        V_JL = mu_L*(sV_L-sV_L0)
        costJL=(c_J*(Jams[0].P+Jams[1].P)+c_conf)    
    if V_JL>=V_JE and V_JL>=0:
        col_now=[0,1]
        UJ=UJL
        V_L = V_L-mu_L*(V_L-V_L0)
        costJ = costJL
    elif V_JL<V_JE and V_JE>=0:
        col_now=[1,0]
        UJ=UJE
        V_E = V_E-mu_E*(V_E-V_E0)
        costJ = costJE
    elif UJL<=0 and UJE<=0:
        col_now=[0,0]
        if col_old == [0,0]:
            UJ = 0
        else:
            UJ = -c_conf   
            costJ=c_conf    
    return  costJ, UJ

c_eveon = 0.1
c_conf = 0.5
R_min = 2
c_J = 5

UEtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
ULtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
UJtemp = [0.0,0.0,0.0,0.0,0.0,0.0]
UErec = [[],[],[],[],[],[]]
ULrec = [[],[],[],[],[],[]]
UJrec = [[],[],[],[],[],[]]

cJsum = [0.0,0.0,0.0,0.0,0.0,0.0]
cJtemp = 0.0

UJtemp_IM = [0.0,0.0,0.0,0.0,0.0,0.0]
UJrec_IM = [[],[],[],[],[],[]]
cJsum_IM = [0.0,0.0,0.0,0.0,0.0,0.0]
cJtemp_IM = 0.0

topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]

s = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0

while not done:
    if episode_steps >= args.max_episode_steps:
        done = True
    episode_steps += 1
    
    #TSSG
    a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
    topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
    E0s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
    a_LUs = []
    a_LUslogprob = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s_)
        a_LUs.append(tempa)
        a_LUslogprob.append(templp)
    topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
    E1s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
    Lmu_record = np.append(Lmu_record,a_Lmu)
    E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
    a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
    topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
    topo_R, topo_R0 = update_rate(topo_h, topo_P)
    
    c_J = 5
    for i in range(0,6):   
        col_part_, r_E, r_L, r_L_sum, r_J, cJtemp= col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        UEtemp[i]+=r_E
        ULtemp[i]+=r_L_sum
        UJtemp[i]+=r_J
        UErec[i].append(UEtemp[i])
        ULrec[i].append(ULtemp[i])
        UJrec[i].append(UJtemp[i])       
        cJsum[i]+=cJtemp
        cJtemp_IM, UJtemp_IM[i] = JAIM(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        cJsum_IM[i]+=cJtemp_IM
        UJrec_IM[i].append(UJtemp_IM[i])
        c_J += 2
    
    col_part = col_part_
    
    #TSSG
    tempmes = mesmerize(topo_h)
    topo_h*=tempmes
    s_ = gene_state(topo_h, topo_P, col_part, device)
    s = s_
