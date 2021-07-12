use trained model to apply pca matching

key function in visualize_similarity_pytorch.py: pca_optimization()

1. compute embedding features for each image,
2. combine SVD and get heat map
3. resize to actural image size
4. pca template matching
5. output fusion images
