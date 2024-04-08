import pandas as pd
import numpy as np


# helper function to get all embeddings from the GloVe dataset,
# if the embedding word also appears in LaDEC
# Does not check for the associated constituent word or compound word. Returns
# a dictionary with words as keys and embeddings(an array of floats) as values
def get_embeddings(words, glove_path):
    processed_glove = {}
    # Important: GloVe uses utf-8,
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            split = line.strip().split(' ')
            word = split[0]
            if word in words:
                # Convert to numpy array of floats instead of strings
                vector = np.asarray(split[1:], dtype=float)
                processed_glove[word] = vector
    return processed_glove


# Further filter the LaDec, keep only the 2 constitutents (c1,c2) and the
# compound word (stim) are all present in the GloVe dataset.
# Then add associated word embedding columns (arrays with floats)
def filter_and_merge_embeddings(ladec, glove_embeddings):
    # Keep only rows where c1, c2, stim are all in glove_embeddings
    filtered_ladec = ladec[
        ladec['c1'].isin(glove_embeddings) &
        ladec['c2'].isin(glove_embeddings) &
        ladec['stim'].isin(glove_embeddings)
        ].copy()  # copy needed to avoid setting with copy warning

    # Add GloVe embeddings to the filtered data
    filtered_ladec['c1_embedding'] = filtered_ladec['c1'].apply(
        lambda x: glove_embeddings.get(x))
    filtered_ladec['c2_embedding'] = filtered_ladec['c2'].apply(
        lambda x: glove_embeddings.get(x))
    filtered_ladec['stim_embedding'] = filtered_ladec['stim'].apply(
        lambda x: glove_embeddings.get(x))
    return filtered_ladec


# Guess the compound word embedding with 3 methods as weights:
# 1) Average combination: weights = 0.5
# 2) LSA combination: normalize cosine similarity score base on LSA for c1, c2
#     against their sum so that they sum to 1,then use these normalized scores
#     as weights
# 3) COS combination: same as LSA combinations except we get cos similarity by
#    using (1-cos distance).
#    Note: The LSA and COS data did not use the same cos similarity values.
def add_combinations(data):
    # Average combination
    data['average_combination'] = data.apply(
        lambda row: (row['c1_embedding'] + row['c2_embedding']) / 2, axis=1)

    # LSA combination, empty row if division by zero
    data['LSA_combination'] = data.apply(
        lambda row: np.nan if (row['LSAc1stim'] + row['LSAc2stim']) == 0 else
        (row['LSAc1stim'] / (row['LSAc1stim'] + row['LSAc2stim']) * row['c1_embedding']) +
        (row['LSAc2stim'] / (row['LSAc1stim'] + row['LSAc2stim']) * row['c2_embedding']), axis=1)

    # COS combination, empty row if division by zero
    data['COS_combination'] = data.apply(
        lambda row: np.nan if (row['c1stim_snautCos'] + row['c2stim_snautCos']) == 0 else
        ((1 - row['c1stim_snautCos']) / ((1 - row['c1stim_snautCos']) + (1 - row['c2stim_snautCos'])) * row['c1_embedding']) +
        ((1 - row['c2stim_snautCos']) / ((1 - row['c1stim_snautCos']) + (1 - row['c2stim_snautCos'])) * row['c2_embedding']), axis=1)

    # Drop empty rows (division by zero)
    data.dropna(subset=['LSA_combination', 'COS_combination'], inplace=True)
    return data


if __name__ == "__main__":
    ladec_df = pd.read_csv('LADECv1-2019.csv')

    # Path to the GloVe files
    glove_paths = [
        'glove.6B.50d.txt',
        'glove.6B.100d.txt',
        'glove.6B.200d.txt',
        'glove.6B.300d.txt',
        'glove.42B.300d.txt',
        'glove.840B.300d.txt',
        'glove.twitter.27B.25d.txt',
        'glove.twitter.27B.50d.txt',
        'glove.twitter.27B.100d.txt',
        'glove.twitter.27B.200d.txt'
    ]

    # Select needed columns
    select_columns = ['c1', 'c2', 'stim', 'LSAc1c2', 'LSAc1stim', 'LSAc2stim',
                          'c1c2_snautCos', 'c1stim_snautCos', 'c2stim_snautCos']

    # Drop rows with any NaN values in the selected columns
    selected_ladec_cols = ladec_df[select_columns].dropna()

    # Find all unique words from both LSA and COS
    unique_words = set(selected_ladec_cols[['c1', 'c2', 'stim']].
                       stack().unique())

    # Process all GloVe datasets with the LaDEC dataset to get new datasets with
    # embeddings for c1,c2, stim on each row of the LaDEC dataset, if that row's
    # c1,c2 & stim embeddings all exists in the selected GloVe dataset
    for path in glove_paths:
        # Extract part of the file name to use in the new file name
        name_parts = path.split('.')
        # Twitter file names are slightly different from the rest
        if 'twitter' in name_parts[1]:
            name = 'twitter' + name_parts[2] + name_parts[3]
        else:
            name = name_parts[1] + name_parts[2]

        # Perform the operation to get predictions of the compound word
        # embedding from constituents; and save the result to a CSV file
        (add_combinations(
            filter_and_merge_embeddings(selected_ladec_cols,
                                        get_embeddings(unique_words, path))
        )).to_csv(f'ladec_glove{name}_combinations.csv', index=False)

        # Similarly, save the result as a pickle file for each GloVe dataset
        (add_combinations(
            filter_and_merge_embeddings(selected_ladec_cols,
                                        get_embeddings(unique_words, path))
        )).to_pickle(f'ladec_glove{name}_combinations.pkl')
