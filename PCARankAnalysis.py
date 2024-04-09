import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Helper functions for parsing CSV columns
def to_float_array(row):
    nums = row.strip().split()
    if nums[0] == "[":
        nums = nums[1:]
    elif nums[0][0] == "[":
        nums[0] = nums[0][1:]

    if nums[-1] == "]":
        nums = nums[:-1]
    elif nums[-1][-1] == "]":
        nums[-1] = nums[-1][:-1]

    return np.asarray(nums, dtype=float)


def get_word_embeddings(S, column_name):
    shape = S[column_name].shape
    # Change shape
    S_new = np.zeros([shape[0], 300])
    S_float = S[column_name].apply(to_float_array)
    for i in range(shape[0]):
        S_new[i] = S_float[i]
    return S_new


def concat_word_embeddings(D):
    # Parse word embeddings
    D_stim = get_word_embeddings(D, "stim_embedding")
    D_avg = get_word_embeddings(D, "average_combination")
    D_LSA = get_word_embeddings(D, "LSA_combination")
    D_Cos = get_word_embeddings(D, "COS_combination")
    D_alpha = get_word_embeddings(D, "alpha_combination")
    D_stim = np.concatenate((D_stim, D_avg, D_LSA, D_Cos, D_alpha))

    return D_stim


def plot_rows(D, rows, stim_label):
    TOTAL_ROWS = 2486

    # Use PCA
    embedding = PCA(n_components=2)
    embedding_lowdim = embedding.fit_transform(D)

    # Plotting
    plt.figure(figsize=(20, 10))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(rows))))
    for i in rows:
        color = next(colors)
        plt.scatter(embedding_lowdim[i,0], embedding_lowdim[i,1], color=color, marker='o')
        # Average combination
        plt.scatter(embedding_lowdim[TOTAL_ROWS+i,0], embedding_lowdim[TOTAL_ROWS+i,1], color=color, marker='s', label="Average Combination")
        # LSA combination
        plt.scatter(embedding_lowdim[2*TOTAL_ROWS+i,0], embedding_lowdim[2*TOTAL_ROWS+i,1], color=color, marker='v', label="LSA Combination")
        # COS combination
        plt.scatter(embedding_lowdim[3*TOTAL_ROWS+i,0], embedding_lowdim[3*TOTAL_ROWS+i,1], color=color, marker='P', label="COS Combination")
        # Alpha combination
        plt.scatter(embedding_lowdim[4*TOTAL_ROWS+i,0], embedding_lowdim[4*TOTAL_ROWS+i,1], color=color, marker='X', label="Alpha Combination")

    for i in rows:
        plt.text(embedding_lowdim[i,0],embedding_lowdim[i,1],stim_label[i],fontsize=10)

    plt.show()


if __name__ == "__main__":
    # Load and parse data from csv
    # CHANGE THIS PATH FOR DIFFERENT DATASETS
    D = pd.read_csv('ladec_glove42b300d_combinations.csv')

    # Define potential rows to plot
    FISH_ROWS = np.where(D["c2"]=="fish")[0][:10]
    LSA_ROWS = np.where(D["Rank_LSA"] < 10)[0][:10]
    LSA_HEAD_ROWS = D.sort_values("Rank_LSA").head(5).index.to_numpy()
    AVG_HEAD_ROWS = D.sort_values("Rank_avg").head(5).index.to_numpy()
    COS_HEAD_ROWS = D.sort_values("Rank_COS").head(5).index.to_numpy()
    ALPHA_HEAD_ROWS = D.sort_values("Rank_alpha").head(5).index.to_numpy()
    LSA_TAIL_ROWS = D.sort_values("Rank_LSA").tail(5).index.to_numpy()
    AVG_TAIL_ROWS = D.sort_values("Rank_avg").tail(5).index.to_numpy()
    COS_TAIL_ROWS = D.sort_values("Rank_COS").tail(5).index.to_numpy()
    ALPHA_TAIL_ROWS = D.sort_values("Rank_alpha").tail(5).index.to_numpy()


    # Concatenate word embeddings
    D_embeddings = concat_word_embeddings(D)

    # Plot rows: Change second parameter to the desired rows to plot
    plot_rows(D_embeddings, LSA_HEAD_ROWS, D["stim"])