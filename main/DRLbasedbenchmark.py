#This file is a supplimentary code for DRL-based benchmarks, please:
#option 1: add the following code ../main.py, and reorganize the code for running;
#option 2: use consloe mode in PyCharm, after running ../main.py, run these codes in console
#option 3: use scientific mode in PyCharm
#option 4 (most recommanded): run all codes in Spyder, while #%% needs to be used for sepreating the codes

#%%
def reducepoints(ary,rd):
    ary = np.array(ary)
    resary = np.array([])
    for i in range(0,len(ary),rd):
        resary = np.append(resary,ary[i])
    return resary

def col_formation(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
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
    return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ
E_record = np.array([])
UL_record = np.array([])
UJ_record = np.array([])

UE_record_PE = np.array([])
UL_record_PE = np.array([])
UJ_record_PE = np.array([])

UE_record_AN = np.array([])
UL_record_AN = np.array([])
UJ_record_AN = np.array([])

rateE_record = np.array([])
rateT_record = np.array([])
rateS_record = np.array([])

rateE_record_PE = np.array([])
rateT_record_PE = np.array([])
rateS_record_PE = np.array([])

rateE_record_AN = np.array([])
rateT_record_AN = np.array([])
rateS_record_AN = np.array([])

pT_record = np.array([])
pJ_record = np.array([])
Emu_record = np.array([])
Lmu_record = np.array([])

pT_record_PE = np.array([])
pJ_record_PE = np.array([])
Emu_record_PE = np.array([])
Lmu_record_PE = np.array([])

pT_record_AN = np.array([])
pJ_record_AN = np.array([])
Emu_record_AN = np.array([])
Lmu_record_AN = np.array([])

UEtemp = 0.0
ULtemp = 0.0
UJtemp = 0.0

UEtemp_PE = 0.0
ULtemp_PE = 0.0
UJtemp_PE = 0.0

UEtemp_AN = 0.0
ULtemp_AN = 0.0
UJtemp_AN = 0.0

topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
topo_h_PE, topo_P_PE, topo_R_PE, topo_R0_PE = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
topo_h_AN, topo_P_AN, topo_R_AN, topo_R0_AN = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

LUs_PE = copy.deepcopy(org_LUs)
Jams_PE = copy.deepcopy(org_Jams)
BSs_PE = copy.deepcopy(org_BSs)
Eves_PE = copy.deepcopy(org_Eves)

LUs_AN = copy.deepcopy(org_LUs)
Jams_AN = copy.deepcopy(org_Jams)
BSs_AN = copy.deepcopy(org_BSs)
Eves_AN = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]
col_part_PE = [1, 0]
col_part_AN = [0, 1]

s = gene_state(topo_h, topo_P, col_part, device)
s_PE = gene_state(topo_h, topo_P, col_part, device)
s_AN = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0
def mesmerize(topo_h):
    xx = topo_h.shape[0]
    yy = topo_h.shape[1]
    tempmsmr = np.zeros(topo_h.shape)
    for i in range(0,xx):
        for j in range(0,yy):
            msmr = np.random.normal(1,0.5)    
            while msmr<=0.5 or msmr>=1.5:
                msmr = np.random.normal(1,0.5)
            tempmsmr[i][j] = msmr
    return tempmsmr

def col_formation_PE(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    V_E, V_E0, pcostE = gene_revenue_Eve(topo_R,topo_R0, LUs)
    V_L,V_L0, pcostL = gene_revenue_LU(topo_R,topo_R0, LUs)
    UJE=mu_E*(V_E-V_E0)-c_J*(Jams[0].P+Jams[1].P)
    col_now=[1,0]
    UJ=UJE
    V_E = V_E-mu_E*(V_E-V_E0)
    return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ

def col_formation_AN(mu_E,mu_L,col_old,topo_R,topo_R0,LUs,Jams,Eves): #the final stage of the TSSG, generate the utility(reward) of all players and the coalition partition
    V_E, V_E0, pcostE = gene_revenue_Eve(topo_R,topo_R0, LUs)
    V_L,V_L0, pcostL = gene_revenue_LU(topo_R,topo_R0, LUs)
    sV_L = V_L.sum()
    sV_L0 = V_L0.sum()
    UJL=mu_L*(sV_L-sV_L0)-c_J*(Jams[0].P+Jams[1].P)
    col_now=[0,1]
    UJ=UJL
    V_L = V_L-mu_L*(V_L-V_L0)
    return col_now, V_E-pcostE, V_L-pcostL, V_L.sum()-pcostL.sum(), UJ

#%%
col_record = np.array([])
UE_record = np.array([])
UL_record = np.array([])
UJ_record = np.array([])

UE_record_PE = np.array([])
UL_record_PE = np.array([])
UJ_record_PE = np.array([])

UE_record_AN = np.array([])
UL_record_AN = np.array([])
UJ_record_AN = np.array([])

rateE_record = np.array([])
rateT_record = np.array([])
rateS_record = np.array([])

rateE_record_PE = np.array([])
rateT_record_PE = np.array([])
rateS_record_PE = np.array([])

rateE_record_AN = np.array([])
rateT_record_AN = np.array([])
rateS_record_AN = np.array([])

pT_record = np.array([])
pJ_record = np.array([])


pT_record_PE = np.array([])
pJ_record_PE = np.array([])
Emu_record_PE = np.array([])
Lmu_record_PE = np.array([])

pT_record_AN = np.array([])
pJ_record_AN = np.array([])
Emu_record_AN = np.array([])
Lmu_record_AN = np.array([])

# xTnm_record = np.array([])
# xTnm_record_PE = np.array([])
# xTnm_record_AN = np.array([])

# xSWk_record = np.array([])
# xSWk_record_PE = np.array([])
# xSWk_record_AN = np.array([])

EVagent_action_record = np.array([])
EVagentmu_action_record = np.array([])
LUagent_action_record = np.array([])
LUagentmu_action_record = np.array([])
JAagent_action_record = np.array([])

EVagent_action_record_PE = np.array([])
EVagentmu_action_record_PE = np.array([])
LUagent_action_record_PE = np.array([])
LUagentmu_action_record_PE = np.array([])
JAagent_action_record_PE = np.array([])

EVagent_action_record_AN = np.array([])
EVagentmu_action_record_AN = np.array([])
LUagent_action_record_AN = np.array([])
LUagentmu_action_record_AN = np.array([])
JAagent_action_record_AN = np.array([])

rand_record = np.array([])


UEtemp = 0.0
ULtemp = 0.0
UJtemp = 0.0

UEtemp_PE = 0.0
ULtemp_PE = 0.0
UJtemp_PE = 0.0

UEtemp_AN = 0.0
ULtemp_AN = 0.0
UJtemp_AN = 0.0

topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
topo_h_PE, topo_P_PE, topo_R_PE, topo_R0_PE = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)
topo_h_AN, topo_P_AN, topo_R_AN, topo_R0_AN = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

LUs_PE = copy.deepcopy(org_LUs)
Jams_PE = copy.deepcopy(org_Jams)
BSs_PE = copy.deepcopy(org_BSs)
Eves_PE = copy.deepcopy(org_Eves)

LUs_AN = copy.deepcopy(org_LUs)
Jams_AN = copy.deepcopy(org_Jams)
BSs_AN = copy.deepcopy(org_BSs)
Eves_AN = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]
col_part_PE = [1, 0]
col_part_AN = [0, 1]

s = gene_state(topo_h, topo_P, col_part, device)
s_PE = gene_state(topo_h, topo_P, col_part, device)
s_AN = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0

while not done:
    
    rateEtemp = 0.0
    rateTtemp = 0.0
    rateStemp = 0.0
    pTtemp = []
    pJtemp = []
    
    rateEtemp_PE = 0.0
    rateTtemp_PE = 0.0
    rateStemp_PE = 0.0
    pTtemp_PE = []
    pJtemp_PE = []
    
    rateEtemp_AN = 0.0
    rateTtemp_AN = 0.0
    rateStemp_AN = 0.0
    pTtemp_AN = []
    pJtemp_AN = []
    
    
    for i in range(0,LUnum):
        rateEtemp += max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
        rateTtemp += topo_R[i,LUs[i].bs]
        rateStemp += topo_R[i,LUs[i].bs]-max(topo_R[i,BSnum]*Eves[0].open,topo_R[i,BSnum+1]*Eves[1].open)
        pTtemp.append(LUs[i].P)
        
        rateEtemp_PE += max(topo_R_PE[i,BSnum]*Eves_PE[0].open,topo_R_PE[i,BSnum+1]*Eves_PE[1].open)
        rateTtemp_PE += topo_R_PE[i,LUs_PE[i].bs]
        rateStemp_PE += topo_R_PE[i,LUs_PE[i].bs]-max(topo_R_PE[i,BSnum]*Eves_PE[0].open,topo_R_PE[i,BSnum+1]*Eves_PE[1].open)
        pTtemp_PE.append(LUs_PE[i])
        
        rateEtemp_AN += max(topo_R_AN[i,BSnum]*Eves_AN[0].open,topo_R_AN[i,BSnum+1]*Eves_AN[1].open)
        rateTtemp_AN += topo_R_AN[i,LUs_AN[i].bs]
        rateStemp_AN += topo_R_AN[i,LUs_AN[i].bs]-max(topo_R_AN[i,BSnum]*Eves_AN[0].open,topo_R_AN[i,BSnum+1]*Eves_AN[1].open)
        pTtemp_AN.append(LUs_AN[i].P)
    
    for i in range(0,Jamnum):
        pJtemp.append(Jams[i].P)
        pJtemp_PE.append(Jams_PE[i].P)
        pJtemp_AN.append(Jams_AN[i].P)
        
    rateE_record = np.append(rateE_record,rateEtemp)
    rateT_record = np.append(rateT_record,rateTtemp)
    rateS_record = np.append(rateS_record,rateStemp)
    pT_record = np.append(pT_record,pTtemp)
    pJ_record = np.append(pT_record,pTtemp)
    UE_record = np.append(UE_record,UEtemp)
    UL_record = np.append(UL_record,ULtemp)
    UJ_record = np.append(UJ_record,UJtemp)
    col_record = np.append(col_record,col_part)
    
    rateE_record_PE = np.append(rateE_record_PE,rateEtemp_PE)
    rateT_record_PE = np.append(rateT_record_PE,rateTtemp_PE)
    rateS_record_PE = np.append(rateS_record_PE,rateStemp_PE)
    pT_record_PE = np.append(pT_record_PE,pTtemp_PE)
    pJ_record_PE = np.append(pJ_record_PE,pJtemp_PE)
    UE_record_PE = np.append(UE_record_PE,UEtemp_PE)
    UL_record_PE = np.append(UL_record_PE,ULtemp_PE)
    UJ_record_PE = np.append(UJ_record_PE,UJtemp_PE)
    
    rateE_record_AN = np.append(rateE_record_AN,rateEtemp_AN)
    rateT_record_AN = np.append(rateT_record_AN,rateTtemp_AN)
    rateS_record_AN = np.append(rateS_record_AN,rateStemp_AN)
    pT_record_AN = np.append(pT_record_AN,pTtemp_AN)
    pJ_record_AN = np.append(pJ_record_AN,pJtemp_AN)
    UE_record_AN = np.append(UE_record_AN,UEtemp_AN)
    UL_record_AN = np.append(UL_record_AN,ULtemp_AN)
    UJ_record_AN = np.append(UJ_record_AN,UJtemp_AN)
    
    if episode_steps >= args.max_episode_steps:
        done = True
    episode_steps += 1
    
    #TSSG
    a_Edir, a_Edirlogprob = Eveagent_open.choose_action(s)
    EVagent_action_record = np.append(EVagent_action_record, a_Edir)
    topo_h = action_trans_Eve(a_Edir, LUs, Jams, BSs, Eves)
    E0s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Emu, a_Emulogprob = Eveagent_mu.choose_action(E0s_)
    EVagentmu_action_record = np.append(EVagentmu_action_record, a_Emu)
    Emu_record = np.append(Emu_record,a_Emu)
    a_LUs = []
    a_LUslogprob = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s_)
        a_LUs.append(tempa)
        a_LUslogprob.append(templp)
    LUagent_action_record = np.append(LUagent_action_record, a_LUs)
    topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
    E1s_ = gene_state(topo_h, topo_P, col_part, device)
    a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
    LUagentmu_action_record = np.append(LUagentmu_action_record, a_Lmu)
    Lmu_record = np.append(Lmu_record,a_Lmu)
    E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
    a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
    JAagent_action_record = np.append(JAagent_action_record, a_Jam)
    topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
    topo_R, topo_R0 = update_rate(topo_h, topo_P)
    col_part_, r_E, r_L, r_L_sum, r_J = col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
    UEtemp+=r_E
    ULtemp+=r_L_sum
    UJtemp+=r_J
    col_part = col_part_
    
    #Proactive Eavesdropping
    a_Edir_PE, a_Edirlogprob_PE = Eveagent_open.choose_action(s_PE)
    EVagent_action_record_PE = np.append(EVagent_action_record_PE, a_Edir_PE)
    topo_h_PE = action_trans_Eve(a_Edir_PE, LUs_PE, Jams_PE, BSs_PE, Eves_PE)
    E0s__PE = gene_state(topo_h_PE, topo_P_PE, col_part_PE, device)
    a_Emu_PE, a_Emulogprob_PE = Eveagent_mu.choose_action(E0s__PE)
    EVagentmu_action_record_PE = np.append(EVagentmu_action_record_PE, a_Emu_PE)
    Emu_record_PE = np.append(Emu_record_PE,a_Emu_PE)
    a_LUs_PE = []
    a_LUslogprob_PE = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s__PE)
        a_LUs_PE.append(tempa)
        a_LUslogprob_PE.append(templp)
    LUagent_action_record_PE = np.append(LUagent_action_record_PE, a_LUs_PE)
    topo_P_PE = action_trans_LU(a_LUs_PE, LUs_PE, Jams_PE, BSs_PE, Eves_PE)
    E1s__PE = gene_state(topo_h_PE, topo_P_PE, col_part_PE, device)
    a_Lmu_PE, a_Lmulogprob_PE = LUagent_mu.choose_action(E1s__PE)
    LUagentmu_action_record_PE = np.append(LUagentmu_action_record_PE, a_Lmu_PE)
    Lmu_record_PE = np.append(Lmu_record_PE,a_Lmu_PE)
    E1s_J_PE = gene_state_J(topo_h_PE, topo_P_PE, col_part_PE, a_Emu_PE, a_Lmu_PE, device)
    a_Jam_PE, a_Jamlogprob_PE = Jamagent.choose_action(E1s_J_PE)
    JAagent_action_record_PE = np.append(JAagent_action_record_PE, a_Jam_PE)
    topo_P_PE = action_trans_Jam(a_Jam_PE, LUs_PE, Jams_PE, BSs_PE, Eves_PE)
    topo_R_PE, topo_R0_PE = update_rate(topo_h_PE, topo_P_PE)
    col_part__PE, r_E_PE, r_L_PE, r_L_sum_PE, r_J_PE = col_formation_PE(a_Emu_PE, a_Lmu_PE, col_part_PE, topo_R_PE, topo_R0_PE, LUs_PE, Jams_PE, Eves_PE)
    UEtemp_PE+=r_E_PE
    ULtemp_PE+=r_L_sum_PE
    UJtemp_PE+=r_J_PE
    col_part_PE = col_part__PE
    
    #Artifical Noise
    a_Edir_AN, a_Edirlogprob_AN = Eveagent_open.choose_action(s_AN)
    EVagent_action_record_AN = np.append(EVagent_action_record_AN, a_Edir_AN)
    topo_h_AN = action_trans_Eve(a_Edir_AN, LUs_AN, Jams_AN, BSs_AN, Eves_AN)
    E0s__AN = gene_state(topo_h_AN, topo_P_AN, col_part_AN, device)
    a_Emu_AN, a_Emulogprob_AN = Eveagent_mu.choose_action(E0s__AN)
    EVagentmu_action_record_AN = np.append(EVagentmu_action_record_AN, a_Emu_AN)
    Emu_record_AN = np.append(Emu_record_AN,a_Emu_AN)
    a_LUs_AN = []
    a_LUslogprob_AN = []
    for i in range(0, LUnum):
        tempa, templp = LUagents[i].choose_action(E0s__AN)
        a_LUs_AN.append(tempa)
        a_LUslogprob_AN.append(templp)
    LUagent_action_record_AN = np.append(LUagent_action_record_AN, a_LUs_AN)
    topo_P_AN = action_trans_LU(a_LUs_AN, LUs_AN, Jams_AN, BSs_AN, Eves_AN)
    E1s__AN = gene_state(topo_h_AN, topo_P_AN, col_part_AN, device)
    a_Lmu_AN, a_Lmulogprob_AN = LUagent_mu.choose_action(E1s__AN)
    LUagentmu_action_record_AN = np.append(LUagentmu_action_record_AN, a_Lmu_AN)
    Lmu_record_AN = np.append(Lmu_record_AN,a_Lmu_AN)
    E1s_J_AN = gene_state_J(topo_h_AN, topo_P_AN, col_part_AN, a_Emu_AN, a_Lmu_AN, device)
    a_Jam_AN, a_Jamlogprob_AN = Jamagent.choose_action(E1s_J_AN)
    JAagent_action_record_AN = np.append(JAagent_action_record_AN, a_Jam_AN)
    topo_P_AN = action_trans_Jam(a_Jam_AN, LUs_AN, Jams_AN, BSs_AN, Eves_AN)
    topo_R_AN, topo_R0_AN = update_rate(topo_h_AN, topo_P_AN)
    col_part__AN, r_E_AN, r_L_AN, r_L_sum_AN, r_J_AN = col_formation_AN(a_Emu_AN, a_Lmu_AN, col_part_AN, topo_R_AN, topo_R0_AN, LUs_AN, Jams_AN, Eves_AN)
    UEtemp_AN+=r_E_AN
    ULtemp_AN+=r_L_sum_AN
    UJtemp_AN+=r_J_AN
    col_part_AN = col_part__AN
    
    #TSSG
    tempmes = mesmerize(topo_h)
    rand_record = np.append(rand_record, tempmes)
    topo_h*=tempmes
    s_ = gene_state(topo_h, topo_P, col_part, device)
    # col_now, V_E, V_L, V_L.sum(), UJ
    s = s_
    
    #PE
    topo_h_PE*=tempmes
    s__PE = gene_state(topo_h_PE, topo_P_PE, col_part_PE, device)
    # col_now, V_E, V_L, V_L.sum(), UJ
    s_PE = s__PE
    
    #AN
    topo_h_AN*=tempmes
    s__AN = gene_state(topo_h_AN, topo_P_AN, col_part_AN, device)
    # col_now, V_E, V_L, V_L.sum(), UJ
    s_AN = s__AN

# EVagent_action_record = np.array([])
# EVagentmu_action_record = np.array([])
# LUagent_action_record = np.array([])
# LUagentmu_action_record = np.array([])
# JAagent_action_record = np.array([])

np.save("randrecord.npy",rand_record)
np.save("EVagent_action_record.npy",EVagent_action_record)
np.save("EVagentmu_action_record.npy",EVagentmu_action_record)
np.save("LUagent_action_record.npy",LUagent_action_record)
np.save("LUagentmu_action_record.npy",LUagentmu_action_record)
np.save("JAagent_action_record.npy",JAagent_action_record)

np.save("EVagent_action_record_PE.npy",EVagent_action_record_PE)
np.save("EVagentmu_action_record_PE.npy",EVagentmu_action_record_PE)
np.save("LUagent_action_record_PE.npy",LUagent_action_record_PE)
np.save("LUagentmu_action_record_PE.npy",LUagentmu_action_record_PE)
np.save("JAagent_action_record_PE.npy",JAagent_action_record_PE)

np.save("EVagent_action_record_AN.npy",EVagent_action_record_AN)
np.save("EVagentmu_action_record_AN.npy",EVagentmu_action_record_AN)
np.save("LUagent_action_record_AN.npy",LUagent_action_record_AN)
np.save("LUagentmu_action_record_AN.npy",LUagentmu_action_record_AN)
np.save("JAagent_action_record_AN.npy",JAagent_action_record_AN)

np.save("SUE0.npy",UE_record)
np.save("SUL0.npy",UL_record)
np.save("SUJ0.npy",UJ_record)
np.save("SUE_PE0.npy",UE_record_PE)
np.save("SUL_PE0.npy",UL_record_PE)
np.save("SUJ_PE0.npy",UJ_record_PE)
np.save("SUE_AN0.npy",UE_record_AN)
np.save("SUL_AN0.npy",UL_record_AN)
np.save("SUJ_AN0.npy",UJ_record_AN)

#%% avgU of LUs EVs JAs vs c_conf
# Y: avg UE X: c_conf
# c_conf: [0.1,0.6] 0,0.1,0.2,0.3,0.4,0.5,0.6

c_conf = 0.3

UEtemp = [0.0,0.0,0.0,0.0,0.0]
ULtemp = [0.0,0.0,0.0,0.0,0.0]
UJtemp = [0.0,0.0,0.0,0.0,0.0]
UErec = [[],[],[],[],[]]
ULrec = [[],[],[],[],[]]
UJrec = [[],[],[],[],[]]

topo_h, topo_P, topo_R, topo_R0 = init_net_topo(org_LUs, org_Jams, org_BSs, org_Eves)

LUs = copy.deepcopy(org_LUs)
Jams = copy.deepcopy(org_Jams)
BSs = copy.deepcopy(org_BSs)
Eves = copy.deepcopy(org_Eves)

done = False

col_part = [0, 0]

s = gene_state(topo_h, topo_P, col_part, device)

episode_steps = 0
c_conf=0.3
for j in range(0,5):
    c_eveon = 0.1
    R_min = 2
    c_J = 5
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
        #Emu_record = np.append(Emu_record,a_Emu)
        a_LUs = []
        a_LUslogprob = []
        for i in range(0, LUnum):
            tempa, templp = LUagents[i].choose_action(E0s_)
            a_LUs.append(tempa)
            a_LUslogprob.append(templp)
        topo_P = action_trans_LU(a_LUs, LUs, Jams, BSs, Eves)
        E1s_ = gene_state(topo_h, topo_P, col_part, device)
        a_Lmu, a_Lmulogprob = LUagent_mu.choose_action(E1s_)
        #Lmu_record = np.append(Lmu_record,a_Lmu)
        E1s_J = gene_state_J(topo_h, topo_P, col_part, a_Emu, a_Lmu, device)
        a_Jam, a_Jamlogprob = Jamagent.choose_action(E1s_J)
        topo_P = action_trans_Jam(a_Jam, LUs, Jams, BSs, Eves)
        topo_R, topo_R0 = update_rate(topo_h, topo_P)
        #print("c_conf=",c_conf)
        col_part_, r_E, r_L, r_L_sum, r_J = col_formation(a_Emu, a_Lmu, col_part, topo_R, topo_R0, LUs, Jams, Eves)
        UEtemp[j]+=r_E
        ULtemp[j]+=r_L_sum
        UJtemp[j]+=r_J
        UErec[j].append(UEtemp[j])
        ULrec[j].append(ULtemp[j])
        UJrec[j].append(UJtemp[j])
  
        col_part = col_part_
        
        #TSSG
        tempmes = mesmerize(topo_h)
        topo_h*=tempmes
        s_ = gene_state(topo_h, topo_P, col_part, device)
        # col_now, V_E, V_L, V_L.sum(), UJ
        s = s_
    c_conf+=0.1

