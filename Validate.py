import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import copy


# Generate a matrix of cos similarities for all pairs of embedding values
# between a given column and stim_embedding column, if column is not specified,
# the stim_embedding column will compare with itself to produce the
# similarity matrix
def generate_similarity_matrix(data, column=None):
    # Get the stim_embedding column
    column1_embeddings = np.stack(data['stim_embedding'].values)
    # If no 2nd column is specified in input, compare stim_embedding against
    # itself
    if column is None:
        similarity_matrix = cosine_similarity(column1_embeddings,
                                                column1_embeddings)
    else:
        # If 2nd column is specified, compare stim_embedding against it
        column2_embeddings = np.stack(data[column].values)
        similarity_matrix = cosine_similarity(column1_embeddings,
                                                column2_embeddings)
    return similarity_matrix


# For the given data (in dataframe), calculate the rank and mrr between the
# given combination column and the stim_embedding column
def calculate_ranks_and_mrr(data, combination_col,
                            similarity_matrix, closest_compounds):
    ranks = []  # Stores each rows' rank
    mrr_sum = 0  # sum of reciprocal rank is 0 at start
    # Deep copy closet_compounds since we need to perform the same function
    # on other columns as well
    closest = copy.deepcopy(closest_compounds)

    # Loop through each row in the data, idx is the row index
    for idx, row in data.iterrows():

        current_combination_similarity = cosine_similarity(
                np.array(row[combination_col]).reshape(1, -1),
                np.array(row['stim_embedding']).reshape(1, -1))[0][0]

        # Get all similarities of the current stim_embedding to all others
        # stim_embeddings from the similarity matrix
        all_stim_similarities = similarity_matrix[idx]

        # Find the rank of the current combination embedding's similarity
        rank = (all_stim_similarities >= current_combination_similarity).sum()+1
        ranks.append(rank)

        # If the rank is 1, then it means the predicted embedding is the closest
        # change the value of closets_compounds at index idx
        if rank == 1:
            closest.iloc[idx] = "Predicted Embedding"
        # Get the reciprocal ranks
        mrr_sum += 1 / rank

    # Calculate MRR
    mrr = mrr_sum / len(data)
    # Align ranks with proper row index, otherwise null values might appear
    ranks = pd.Series(ranks, index=data.index)

    return ranks, mrr, closest


# Find the n-closest(default=1) compound-word embedding(stim_embedding) for the
# embeddings used in the cos similarity matrix (stim by default), return a panda
# series with index matching the row index in the dataframe.
def find_closest_embeddings(data, similarity_matrix, top_n=1):
    # Making sure that the row index is correct
    data.reset_index(drop=True, inplace=True)

    # Ignore the diagonal to exclude self comparison
    np.fill_diagonal(similarity_matrix, 0)

    if top_n == 1:
        # Find the index of the max similarity for each row
        closest_indices = np.argmax(similarity_matrix, axis=1)
        # Map the indices to the specified column's values for corresponding
        # compound-words(stim)
        closest_compounds = (data.iloc[closest_indices]['stim'].
                             reset_index(drop=True))
    else:
        # Find the indices of the top N max similarities for each row
        closest_indices = np.argsort(similarity_matrix, axis=1)[:, -top_n:]
        # Map the indices to the specified column's values for corresponding
        # compound-words(stim)
        closest_compounds = pd.Series(
            [data.iloc[indices]['stim'].values.tolist() for indices in
             closest_indices], index=data.index)

    return closest_compounds


# From https://stackoverflow.com/questions/42021972/truncating-decimal-digits-numpy-array-of-floats 
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def create_alpha_column(data, alpha):
    df = data["c1_embedding"]*alpha + data["c2_embedding"]*(1-alpha)
    df.rename("alpha_combination_"+str(alpha), inplace=True)
    return df


def get_best_alpha_combination(path, closest_compounds):
    # Load the dataframe from the pickle file
    data = pd.read_pickle(path)
    # Reset the index for proper alignment, otherwise might have null values
    data.reset_index(drop=True, inplace=True)

    # Create a new column for each alpha combination from 0 to 1 in steps of 0.01
    alphas = (create_alpha_column(data, alpha) for alpha in np.arange(0, 1.01, 0.01))
    alphas_df = pd.concat(alphas, axis=1)
    data = pd.concat([data, alphas_df], axis=1)
    
    alphas = []
    # Perform 5-fold cross validation
    for _ in range(5):
        # Split data 80/20 train/test
        train, test = train_test_split(data, test_size=0.2)

        # Reset the index for proper alignment, otherwise might have null values
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        best_alpha = 0
        best_mrr = 0
        train_similarity_matrix = generate_similarity_matrix(train)

        # Try all alphas on the training set to get best alpha
        for step in range(101):
            alpha = 0.01 * step
            _, mrr, _ = calculate_ranks_and_mrr(
            train, "alpha_combination_"+str(alpha),
            train_similarity_matrix, closest_compounds
            )

            if (mrr > best_mrr):
                best_alpha = alpha
                best_mrr = mrr

        alphas.append(best_alpha)

    # Take average of alphas from all 5 folds as the best alpha
    best_alpha = np.average(alphas)
    return trunc(best_alpha, 2)
    

if __name__ == "__main__":
    paths = [
        'ladec_glove6B50d_combinations.pkl',
        'ladec_glove6B100d_combinations.pkl',
        'ladec_glove6B200d_combinations.pkl',
        'ladec_glove6B300d_combinations.pkl',
        'ladec_glove42B300d_combinations.pkl',
        'ladec_glove840B300d_combinations.pkl',
        'ladec_gloveTwitter27B25d_combinations.pkl',
        'ladec_gloveTwitter27B50d_combinations.pkl',
        'ladec_gloveTwitter27B100d_combinations.pkl',
        'ladec_gloveTwitter27B200d_combinations.pkl',
    ]

    # Update all pickle and csv files at once.
    # This will take a few minutes to run.
    for path in paths:
        # Load the dataframe from the pickle file
        data = pd.read_pickle(path)
        # Reset the index for proper alignment, otherwise might have null values
        data.reset_index(drop=True, inplace=True)
        # Get similarity matrix for all pairs of stim_embeddings
        sim_matrix = generate_similarity_matrix(data)
        closest_compounds = find_closest_embeddings(data, sim_matrix)
        # Find ranks and MRR for each combination type
        ranks_avg, mrr_avg, closest_avg = calculate_ranks_and_mrr(
            data, 'average_combination',
            sim_matrix, closest_compounds)
        ranks_lsa, mrr_lsa, closest_lsa = calculate_ranks_and_mrr(
            data, 'LSA_combination',
            sim_matrix, closest_compounds)
        ranks_cos, mrr_cos, closest_cos = calculate_ranks_and_mrr(
            data, 'COS_combination',
            sim_matrix, closest_compounds)
        # Compute and add alpha combination type
        best_alpha = get_best_alpha_combination(path, closest_compounds)
        data["alpha"] = best_alpha
        data["alpha_combination"] = best_alpha * data["c1_embedding"] + (1-best_alpha) * data["c2_embedding"]
        ranks_alpha, mrr_alpha, closest_alpha = calculate_ranks_and_mrr(
            data, "alpha_combination", 
            sim_matrix, closest_compounds)

        # Assign them to data as new columns
        data['Rank_avg'] = ranks_avg
        data['Rank_LSA'] = ranks_lsa
        data['Rank_COS'] = ranks_cos
        data['Rank_alpha'] = ranks_alpha
        data['MRR_avg'] = mrr_avg
        data['MRR_LSA'] = mrr_lsa
        data['MRR_COS'] = mrr_cos
        data['MRR_alpha'] = mrr_alpha
        data['Rank_1_avg'] = closest_avg
        data['Rank_1_LSA'] = closest_lsa
        data['Rank_1_COS'] = closest_cos
        data['Rank_1_alpha'] = closest_alpha

        # find top3 closest compound words for each method
        top3_closest_avg = find_closest_embeddings(data,
                    generate_similarity_matrix(data,'average_combination'),
                                                    top_n=3)
        top3_closest_lsa = find_closest_embeddings(data,
                    generate_similarity_matrix( data, 'LSA_combination'),
                                                    top_n=3)
        top3_closest_cos = find_closest_embeddings(data,
                    generate_similarity_matrix( data, 'COS_combination'),
                                                    top_n=3)
        top3_closest_alpha = find_closest_embeddings(data,
                    generate_similarity_matrix( data, 'alpha_combination'),
                                                    top_n=3)
        data['Top3_Closest_avg'] = top3_closest_avg
        data['Top3_Closest_LSA'] = top3_closest_lsa
        data['Top3_Closest_COS'] = top3_closest_cos
        data['Top3_Closest_alpha'] = top3_closest_alpha

        # Save to pickle
        data.to_pickle(path)

        # Save to csv
        csv_path = path.replace('.pkl', '.csv')
        data.to_csv(csv_path)
