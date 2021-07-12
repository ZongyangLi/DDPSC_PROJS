from datetime import datetime
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class ConfidenceIntervalNormalize:
    def __init__(self, z=3):
        self.z = z

    def __call__(self, x):
        x = np.array(x).astype(np.float32)
        x_mean = x.mean()
        x_std = x.std()
        if x_std == 0:
            return x
        x_min = x_mean - self.z * x_std
        x_max = x_mean + self.z * x_std
        x[x>x_max] = x_max
        x[x<x_min] = x_min
        return (x - x_min)/(x_max - x_min) 

class NonzeroNormalize:
    """Normalize with the ignorance of zero value."""
    def __init__(self):
        pass

    def __call__(self, x):
        x = np.array(x).astype(np.float32)
        if len(x[x.nonzero()]) == 0:
            return x
        else:
            x_min = x[x.nonzero()].min()
            x_max = x[x.nonzero()].max()
        if x_min == x_max:
             return x/x_max
        x[x.nonzero()] = (x[x.nonzero()] - x_min)/(x_max - x_min) 
        return x

class NChannels():
    def __init__(self, n):
        self.n = n
        pass

    def __call__(self, x):
        return np.repeat(x[:, :, np.newaxis], self.n, axis=2)

def parse_subfolder_datetime(subfolder_name: str) -> datetime:
    subfolder_dt = datetime.strptime(subfolder_name[:-4], '%Y-%m-%d__%H-%M-%S')
    return subfolder_dt


def visualize_enhance(image, copy=False, avg_factor=4):
    if copy:
        image = image.copy()
    avg = np.mean(image)
    image[image < avg/avg_factor] = avg/avg_factor
    image = image - avg/avg_factor
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def dataset_postproc():
    pass


def torch_dataset_to_filename_list(dataset):
    return [sample[0] for sample in dataset.samples]


def compute_tsne(vectors, labels, select_func=None, colored_by=0, plot=False, cmap='hsv', tsne_model=None, **kwargs):
    """
    Compute T-SNE
    :param vectors:
    :param labels:
    :param select_func:
    :param colored_by:
    :param plot:
    :param cmap:
    :param kwargs:
    :param tsne_model
    :return:
    """

    if select_func is not None:
        tsne_idx = select_func(vectors, labels, **kwargs)
        tsne_vectors = vectors[tsne_idx, :]
        tsne_labels = labels[tsne_idx, :]
    else:
        tsne_vectors = vectors
        tsne_labels = labels[:, :]
    if tsne_model is not None:
        tsne_model = tsne_model
    else:
        tsne_model = TSNE(n_components=2, perplexity=40, learning_rate=200, verbose=1)
    tsne_Y = tsne_model.fit_transform(tsne_vectors)
    if plot:
        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_Y[:, 0], tsne_Y[:, 1], c=tsne_labels[:, colored_by], cmap=cmap)
    return tsne_Y, tsne_labels
    
def retrieve_by_knn(distance_matrix, k):
    nearest_neighbor = NearestNeighbors(n_neighbors=k+1, 
                                        algorithm='ball_tree').fit(distance_matrix)
    X_distances, X_neighbor_indices = nearest_neighbor.kneighbors(distance_matrix)
    return X_neighbor_indices[:, 1:]

def compute_nn_recall_at_k(retrieved_indices, Y, k):
    assert retrieved_indices.shape[1] >= k
    assert retrieved_indices.shape[0] == Y.shape[0]
    classes, classes_count = np.unique(Y, return_counts=True)
    class_count_dict = dict(zip(classes, classes_count))
    score_sum = 0
    for indices, y in zip(retrieved_indices, Y):
        if np.isnan(y):
            continue
        indices = indices[:k]
        retrived_labels = np.array([Y[i] for i in indices])
        tp_count = np.count_nonzero(retrived_labels == y)
        score_sum += tp_count / (class_count_dict[y] - 1)
    score = score_sum / len(Y)
    return score
        
        
def compute_nn_percision_at_k(retrieved_indices, Y, k):
    assert retrieved_indices.shape[1] >= k
    assert retrieved_indices.shape[0] == Y.shape[0]
    classes, classes_count = np.unique(Y, return_counts=True)
    class_count_dict = dict(zip(classes, classes_count))
    score_sum = 0
    for indices, y in zip(retrieved_indices, Y):
        if np.isnan(y):
            continue
        indices = indices[:k]
        retrived_labels = np.array([Y[i] for i in indices])
        tp_count = np.count_nonzero(retrived_labels == y)
        score_sum += tp_count / k
    score = score_sum / len(Y)
    return score
        
        
        
        




