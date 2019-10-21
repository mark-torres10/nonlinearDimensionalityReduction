# Nonlinear Dimensionality Reduction 

A nonlinear approach to dimensionality reduction, via diffusion maps (performed for "Machine Learning for Biology" course). 

### Why Dimensionality Reduction?

Analyzing datasets with a large number of variables becomes computationally complex. Moreover, many variables often are correlated with each other. The combination of these two factors means that we can often reduce the number of variables in our dataset and still retain much of the information. To reduce the size of the dataset, we perform dimensionality reduction. 

### Typical approach to dimensionality reduction: Principal Components Analysis (PCA)

The most straightforward approach to dimensionality reduction is Principal Components Analysis, or PCA. PCA takes the data and finds the directions of maximum variance (by looking at the eigenvectors and eigenvalues), and then projects the data on these newly formed axes. In doing so, the dimensions of the data are greatly reduced while much of the variance is (ideally) preserved.

### When does PCA fail?

PCA assumes that the data has a linear projection on a new set of axes. However, it's possible that the data cannot be projected on these axes. One clear example of this would be a 3-D set of data where, when the data is plotted, it forms a parabolic U-shape. When PCA is formed on this type of data, the underlying shape of the data (the U-shape) isn't captured, so the projection of the data on the principal components won't be accurate. 

An ideal alternative method would be to use the principles of PCA (e.g., using eigenvectors and eigenvalues to determine the directions of maximum variance) while accounting for the "shape" of the data. 

### Alternative to PCA: Diffusion Maps

A nonlinear alternative to PCA is a Diffusion Map. This approach implements a method similar to PCA, but accounts for the underlying "shape" of the data. Diffusion maps look at the distances between points in a dataset to create a visualization of the underlying "shape", or manifold, of the data, and project the data on a new set of axes that follow along the underlying direction of the manifold. After doing so, the diffusion map can implement an approach similar to PCA, but essentially "unfolds" the underlying shape of the data and uses the eigenvectors to find the directions of maximal variance. 

### Implementation 

This repository contains the following files:

• Data: contains the data used for the project. Two datasets were used for this project: (1) a "Swiss roll" dataset, which demosntrates an example of a dataset that would require the use of a diffusion map, and (2) a mass cytometry dataset that measures protein activity for a certain class of fibroblasts. 

• ps1_functions.py: this file contains the helper functions required to implement the steps of the diffusion map (contains 2 pre-provided helper functions, with appropriate credit given in the .py file. The rest of the skeleton code is my own)

• ps1.ipynb: implements the diffusion map algorithm (code is my own)
