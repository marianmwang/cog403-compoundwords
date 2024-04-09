# COG403 - Vectorized Semantics: Exploring Conceptual Combination with Compound Word Embeddings
Liyu Feng, Marian Wang

This repository contains all the code needed to reproduce our final paper results.

# How to reproduce results
Our results from running the simulation are saved in the data folder. 

To reproduce, first download the [Glove datasets](https://nlp.stanford.edu/projects/glove/). Then, run the following files in order:
1. Combination.py - Cleans data and adds columns for Average, LSA, and Cosine combinations
3. Validate.py - Adds a column for best alpha combination. Computes rank and MRR for Average, LSA, and Cosine combinations.
4. Bootstrap.py - Bootstrapping to determine confidence intervals for MRR
5. PCARankAnalysis.py - Visualizes top 5 and lowest 5 ranked words for Average, LSA, and Cosine combinations


