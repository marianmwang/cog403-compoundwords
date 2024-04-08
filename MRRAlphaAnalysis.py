import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_MRR(dfs, legend):
    MRR = np.zeros((len(dfs), 4))

    for i, data in enumerate(dfs):
        # alpha_label = list(filter(lambda x: "MRR_alpha_" in x, data.columns))[0]
        MRR[i][0] = data["MRR_avg"][0]
        MRR[i][1] = data["MRR_LSA"][0]
        MRR[i][2] = data["MRR_COS"][0]
        MRR[i][3] = data["MRR_alpha"][0]
    

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(["avg", "LSA", "Cos", "alpha"], MRR.T, 'o')
    ax.legend(legend, loc="upper right", bbox_to_anchor=(0.9, 1))
    ax.set_xlabel("Combination Type", fontsize=10)
    ax.set_ylabel("MRR", fontsize=10)

    plt.show()

def plot_alpha(dfs, legend):
    alpha = []

    for data in dfs:
        alpha_value = list(filter(lambda x: "alpha_combination_" in x, data.columns))[0]
        alpha.append(float(alpha_value[18:]))
    
    print(alpha)
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

    plot_MRR(DFs, legend)

    #  Alpha analysis
    plot_alpha(DFs, legend)


