import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# Generate a matrix of cos similarities for all pairs of stim_embedding value
def generate_similarity_matrix(df):
    # Get the stim_embeddings column
    stim_embeddings = np.stack(df['stim_embedding'].values)
    # Get cos similarity of all pairs of values as a matrix
    similarity_matrix = cosine_similarity(stim_embeddings)
    return similarity_matrix


# For the given data (in dataframe), calculate the rank and mrr between the
# given combination column and the stim_embedding column
def calculate_ranks_and_mrr(data, combination_col, similarity_matrix):
    ranks = []  # Stores each rows' rank
    mrr_sum = 0  # sum of reciprocal rank is 0 at start

    # Loop through each row in the data, idx is the row index
    for idx, row in data.iterrows():
        # Compute the cos simlarity of combination_col and stim_embedding
        current_combination_similarity = cosine_similarity(
            np.array(row[combination_col]).reshape(1, -1),
            np.array(row['stim_embedding']).reshape(1, -1)
        )[0][0]

        # Get all similarities of the current stim_embedding to all others
        # stim_embeddings from the similarity matrix
        all_stim_similarities = similarity_matrix[idx]

        # Calculate the rank of the current combination embedding's similarity
        rank = (all_stim_similarities > current_combination_similarity).sum()+1
        ranks.append(rank)

        # Get the reciprocal ranks
        mrr_sum += 1 / rank
    # Calculate MRR
    mrr = mrr_sum / len(data)
    # Align ranks with proper row index, otherwise null values might appear
    ranks = pd.Series(ranks, index=data.index)

    return ranks, mrr

# CHANGE THIS PATH FOR DIFFERENT DATASETS
glove_path = 'data/ladec_gloveTwitter27B200d_combinations.pkl'
# Load the dataframe from the pickle file
data = pd.read_pickle(glove_path)
# Reset the index for proper alignment, otherwise might have null values
data.reset_index(drop=True, inplace=True)

# Create a new column for each alpha combination from 0 to 1 in steps of 0.01
for step in range(101):
  alpha = 0.01 * step
  data["alpha_combination_"+str(alpha)] = data.apply(
      lambda row: alpha*row["c1_embedding"]+(1-alpha)*row["c2_embedding"],axis=1
      )
  
test_mrrs = []
# Perform 5-fold cross validation
for i in range(5):
  # Split data 80/20 train/test
  train, test = train_test_split(data, test_size=0.2)

  # Reset the index for proper alignment, otherwise might have null values
  train.reset_index(drop=True, inplace=True)
  test.reset_index(drop=True, inplace=True)

  mrrs = []
  best_alpha = 0
  best_mrr = 0
  train_similarity_matrix = generate_similarity_matrix(train)
  # try all alphas on the training set to get best alpha
  for step in range(101):
    alpha = 0.01 * step
    _, mrr = calculate_ranks_and_mrr(train, "alpha_combination_"+str(alpha), train_similarity_matrix)
    mrrs.append(mrr)
    if (mrr > best_mrr):
      best_alpha = alpha
      best_mrr = mrr
  # use best alpha on test set and store MRR
  ranks, mrr = calculate_ranks_and_mrr(test, "alpha_combination_"+str(best_alpha), generate_similarity_matrix(test))
  data["ranks_alpha_combination"+str(best_alpha)] = ranks
  test_mrrs.append((best_alpha, mrr))

# Print best alpha and MRRs for each fold
print(test_mrrs)