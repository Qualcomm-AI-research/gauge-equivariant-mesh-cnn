from gem_cnn.utils.matrix_features import matrix_features


class MatrixFeaturesTransform:
    """
    Compute matrix features using direct neighbours, weights=1.
    """

    def __call__(self, data):
        if hasattr(data, "edge_mask"):
            edges = data.edge_index[:, (data.edge_mask & 2) != 0]  # First convolutional layer
        else:
            edges = data.edge_index
        data.matrix_features = matrix_features(edges, data.pos, data.frame)
        return data
