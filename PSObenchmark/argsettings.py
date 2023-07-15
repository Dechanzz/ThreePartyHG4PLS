# D, N, M, p_low, p_up, v_low, v_high, w = 1., c1 = 2., c2 = 2.
# 5, 100, 50, low, up, -1, 1, w = 0.9

class ARGS:
    def __init__(self, fitness):
        self.D = 5  # 粒子维度
        self.N = 100  # 粒子群规模，初始化种群个数
        self.M = 50  # 最大迭代次数
        self.p_low = [1, 1, 1, 1, 1] # 粒子位置的约束范围
        self.p_up = [25, 25, 25, 25, 25]
        self.v_low = -10
        self.v_high = 10
        self.w = 0.9  # 惯性权值
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 2.0  # 群体学习因子
        self.fitness = fitness
