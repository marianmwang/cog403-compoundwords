import pandas as pd
import numpy as np
import Validate


def bootstrap_mrr(df, combination_cols, similarity_matrix,
                  closest_compounds, n_iterations=1000):
    # Dictionary to store MRR scores for each combination method
    mrr_results = {col: [] for col in combination_cols}
    for i in range(n_iterations):
        print(f"Iteration {i}")
        # Bootstrap sample with replacement
        bootstrap_sample = df.sample(n=len(df), replace=True)
        # Get mrr score for each combination column, then add it to the
        # dictionary of mrr scores
        for combination_col in combination_cols:
            _, mrr_score, _ = Validate.calculate_ranks_and_mrr(
                bootstrap_sample, combination_col,
                similarity_matrix, closest_compounds)
            mrr_results[combination_col].append(mrr_score)

    mrr_summary = {}
    for combination_col, scores in mrr_results.items():
        # Mean of all mrr scores
        mean_mrr = np.mean(scores)
        # Standard deviation
        s = np.std(scores, ddof=1)
        # Lower and upper bounds of the 95% CI
        lower_ci = mean_mrr - 1.96 * (s / np.sqrt(n_iterations))
        upper_ci = mean_mrr + 1.96 * (s / np.sqrt(n_iterations))
        mrr_summary[combination_col] = (mean_mrr, lower_ci, upper_ci)
    return mrr_summary


if __name__ == '__main__':
    paths = [
        'ladec_glove6B50d_combinations.pkl',
        'ladec_glove6B100d_combinations.pkl',
        'ladec_glove6B200d_combinations.pkl',
        'ladec_glove6B300d_combinations.pkl',
        'ladec_glove42B300d_combinations.pkl',
        'ladec_glove840B300d_combinations.pkl',
        'ladec_glovetwitter27B25d_combinations.pkl',
        'ladec_glovetwitter27B50d_combinations.pkl',
        'ladec_glovetwitter27B100d_combinations.pkl',
        'ladec_glovetwitter27B200d_combinations.pkl',
    ]

# methods of combining embeddings
combinations = ['average_combination', 'LSA_combination', 'COS_combination']

# Process each pickle file, and write the 95% CI results to a txt file.
# Default for bootstrapping is 100 iterations. This will take a long time.
# especially when iterating through all paths.
with (open('ConfidenceInterval.txt', 'w') as file):
    for path in paths:
        data = pd.read_pickle(path)
        # Get file names from path
        name = path.split('_')[1]
        # Get similarity matrix
        sim_matrix = Validate.generate_similarity_matrix(data)
        # Closest_compounds required as function parameter, does not matter here
        closest_compounds = Validate.find_closest_embeddings(data, sim_matrix)
        print(f"=============================================================="
              f"{name}"
              f"==============================================================")
        # Calculate 95%CI for the MRR of each combination
        mrr_summary = bootstrap_mrr(data, combinations, sim_matrix,
                                    closest_compounds)
        for key, value in mrr_summary.items():
            mean_mrr, lower_ci, upper_ci = value
            print(f"{key} - mean MRR: {mean_mrr}")
            print(f"{key} - 95% CI: [{lower_ci}, {upper_ci}]")
        # Print and save results
        for combination, (mean_mrr, lower_ci, upper_ci) in mrr_summary.items():
            print(f"{name} - {combination} bootstrap mean MRR: {mean_mrr}",
                  file=file)

            # The following if blocks puts the corresponding MRR value to check
            # for the confidence interval
            if combination == 'average_combination':
                # note: this column in data have the same value for all rows
                if lower_ci <= data["MRR_avg"][0] <= upper_ci:
                    print(
                        f"{name} - {combination} MRR is within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)
                else:
                    print(
                        f"{name} - {combination} MRR is NOT within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)

            elif combination == 'LSA_combination':
                # note: this column in data have the same value for all rows
                if lower_ci <= data["MRR_LSA"][0] <= upper_ci:
                    print(
                        f"{name} - {combination} MRR is within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)
                else:
                    print(
                        f"{name} - {combination} MRR is NOT within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)

            elif combination == 'COS_combination':
                # note: this column in data have the same value for all rows
                if lower_ci <= data["MRR_COS"][0] <= upper_ci:
                    print(
                        f"{name} - {combination} MRR is within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)
                else:
                    print(
                        f"{name} - {combination} MRR is NOT within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)
            elif combination == 'alpha_combination':
                # note: this column in data have the same value for all rows
                if lower_ci <= data["MRR_alpha"][0] <= upper_ci:
                    print(
                        f"{name} - {combination} MRR is within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)
                else:
                    print(
                        f"{name} - {combination} MRR is NOT within the 95% CI "
                        f"[{lower_ci}, {upper_ci}]", file=file)