# COG403 - Vectorized Semantics: Exploring Conceptual Combination with Compound Word Embeddings
Liyu Feng, Marian Wang

This repository contains all the code needed to reproduce our final paper results.

# How to reproduce results
Our results from running the simulation are saved in the data folder. 

To fully reproduce, first download the [GloVe datasets](https://nlp.stanford.edu/projects/glove/), the GloVE dataset is too big to upload here. Then, run the following files in order. To partially reproduc without the GloVe data, run step 2-4 with the files in data.
1. Combination.py - Cleans data and adds columns for Average, LSA, and Cosine combinations
2. Validate.py - Adds a column for best alpha combination. Computes rank and MRR for Average, LSA, and Cosine combinations.
3. Bootstrap.py - Bootstrapping to determine confidence intervals for MRR
4. PCARankAnalysis.py - Visualizes top 5 and lowest 5 ranked words for Average, LSA, and Cosine combinations


# How to read the data
====Extracted from LaDEC=====
c1:constituent 1
c2:constituent 2
stim: the compound word formed by c1,c2
LSAc1c2: cosince similarity from LSA between c1 and c2
LSAc1stim: cosine similarity from LSA between c1 and stim
LSAc2stim: cosine similarity from LSA between c2 and stim
c1c2_snautCOS: cosine distance between c1 and c2 from SNAUT database
c1stim_snautCOS: cosine distance between c1 and stim from SNAUT database
c2stim_snautCOS: cosine distance between c2 and stim from SNAUT database
====Extracted from GloVe=====
c1_embedding: embedding for c1
c2_embedding: embedding for c2
stim_embedding: embedding for stim
====New data from this study=====
average_combination: new embedding formed from c1,c2 constituents using avg combination method
LSA_combination: new embedding formed from c1,c2 constituents using LSA combination method
COS_combination: new embedding formed from c1,c2 constituents using COS combination method
alpha: the alpha used in alpha combination
alpha_combination: new embedding formed from c1,c2 constituents using alpha combination method
Rank_avg: rank of the new embedding from avg combination method
Rank_LSA: rank of the new embedding from LSA combination method
Rank_COS: rank of the new embedding from COS combination method
Rank_alpha: rank of the new embedding from alpha combination method
MRR_avg: MRR of the new embedding from avg combination method
MRR_LSA: MRR of the new embedding from LSA combination method
MRR_COS: MRR of the new embedding from COS combination method
MRR_alpha: MRR of the new embedding from alpha combination method
Rank1_avg: Closest compound word to the actual compoound word for the avg combination method, if the prediction embedding is rank 1, it     will say predicted embedding
Rank1_LSA: Closest compound word to the actual compoound word for the LSA combination method, if the prediction embedding is rank 1, it     will say predicted embedding
Rank1_COS: Closest compound word to the actual compoound word for the COS combination method, if the prediction embedding is rank 1, it     will say predicted embedding
Rank1_alpha: Closest compound word to the actual compoound word for the alpha combination method, if the prediction embedding is rank 1, it will say predicted embedding
Top3_avg: Top 3 closest compound word to the prediction embedding for the avg combination method
Top3_LSA: Top 3 closest compound word to the prediction embedding for the LSA combination method
Top3_COS: Top 3 closest compound word to the prediction embedding for the COS combination method
Top3_alpha: Top 3 closest compound word to the prediction embedding for the alpha combination method
