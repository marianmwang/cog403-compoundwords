import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def random_chance_MRR(rows):
    ranks = 0
    for _ in range(rows):
        ranks += 1 / random.randint(1, rows)
    mrr = ranks / rows
    return mrr

def plot_MRR(dfs, legend):
    MRR = np.zeros((len(dfs), 5))

    for i, data in enumerate(dfs):
        MRR[i][0] = data["MRR_avg"][0]
        MRR[i][1] = data["MRR_LSA"][0]
        MRR[i][2] = data["MRR_COS"][0]
        MRR[i][3] = data["MRR_alpha"][0]
        MRR[i][4] = random_chance_MRR(data.shape[0])
    

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(["avg", "LSA", "Cos", "alpha", "chance"], MRR.T, 'o')
    ax.legend(legend, loc="upper right", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Combination Type", fontsize=10)
    ax.set_ylabel("MRR", fontsize=10)

    plt.show()


def plot_alpha(dfs, legend):
    alpha = []

    for data in dfs:
        alpha.append(float(data["alpha"][0]))
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(legend, alpha, 'o')
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel("Glove Embedding", fontsize=10)
    ax.set_ylabel("Best Alpha Value", fontsize=10)
    plt.show()


if __name__ == "__main__":
    # MRR analysis
    glove_paths = [
        'ladec_glove6b50d_combinations.csv',
        'ladec_glove6b100d_combinations.csv',
        'ladec_glove6b200d_combinations.csv',
        'ladec_glove6b300d_combinations.csv',
        'ladec_glove42b300d_combinations.csv',
        'ladec_glove840b300d_combinations.csv',
        'ladec_gloveTwitter27b25d_combinations.csv',
        'ladec_gloveTwitter27b50d_combinations.csv',
        'ladec_gloveTwitter27b100d_combinations.csv',
        'ladec_gloveTwitter27b200d_combinations.csv',
    ]

    legend = [path.split("_")[1][5:] for path in glove_paths]
    DFs = [pd.read_csv(path) for path in glove_paths]

    # MRR analysis
    print
    plot_MRR(DFs, legend)

    #  Alpha analysis
    plot_alpha(DFs, legend)


