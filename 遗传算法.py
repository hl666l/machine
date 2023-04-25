# 遗传算法
"""
    1.初始化一个种群，种群中的个体DNA表示
    2.种群中的个体进行交叉变异产生后代
    3.根据后代中每个个体适应度进行自然选择、优胜劣汰
    4.不断迭代产生最优种群

"""
import numpy as np

DNA_SIZE = 8  # 基因个数
POP_SIZE = 200  # 种群个体个数
CROSSVER_RATE = 0.9  # 交叉概率
MUTATION_RATE = 0.01  # 突变率
N_GENERATIONS = 5000  # 迭代次数
X_BOUND = [-3, 3]  # x的取值范围
Y_BOUND = [-3, 3]  # y的取值范围


#  求二元函数的最大值
def F(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)


# 得到最大适应度
def get_fitness(pop):
    """
    种群所有元素减去最小的
    :param pop:种群
    :return: 返回适应度
    """
    x, y = translateDNA(pop)
    pred = F(x, y)  # 计算这个基因对应的F值
    return (pred - np.min(pred)) + 1e-3


def translateDNA(pop):
    '''
    解码
    :param pop: 种群矩阵，一行表示一个二进制编码的个体（可能解），行数为种群中个体数目
    :return: 返回的x,y 是一个行 为种群大小 列为 1 的矩阵 每一个值代表[-3,3]上x,y的可能取值（十进制数）
    '''
    x_pop = pop[:, 1::2]  # pop中的奇数列表示x
    y_pop = pop[:, 0::2]  # pop中的偶数列表示y
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]  # 计算基因对应的x值
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]  # 计算基因对应的y值
    return x, y


# 交叉、变异
def crossover_and_mutation(pop, CROSSVER_RATE=0.8):  # 计算交叉，变异基因
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（代表一个个体的一个二进制0，1串）
        if np.random.rand() < CROSSVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 在种群中选择另一个个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]
        mutation(child)
        new_pop.append(child)
    return new_pop


def mutation(child, MUTATION_RATE=0.1):  # 计算变异基因
    if np.random.rand() < MUTATION_RATE:
        mutate_points = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_points] = child[mutate_points] ^ 1  # 将变异点位置的二进制反转


def select(pop, fitness):  # 自然选择，优胜劣汰
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness) / fitness.sum())
    return pop[idx]


def print_info(pop):  # 输出
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    # print('此时种群',pop)
    # print('max_fitness:',fitness[max_fitness_index])
    x, y = translateDNA(pop)
    # print('最优基因型：',pop[max_fitness_index])
    # print('(x,y):',x[max_fitness_index],y[max_fitness_index])
    print('max_fitness:%s,函数最大值:%s' % (fitness[max_fitness_index], F(x[max_fitness_index], y[max_fitness_index])))


if __name__ == '__main__':
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # 初始化种群，生成元素为0-1，尺寸为200x16的矩阵
    for i in range(N_GENERATIONS):
        x, y = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop))  # 交叉变异
        fitness = get_fitness(pop)  # 得到适应度
        pop = select(pop, fitness)  # 优胜劣汰
        if i % 100 == 0:
            print('第%s次迭代:' % i)
            print_info(pop)
