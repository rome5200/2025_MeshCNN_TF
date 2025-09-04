"""TensorFlow implementation of a minimal MeshCNN model.

This module provides a basic port of the MeshCNN architecture from
https://github.com/ranahanocka/MeshCNN to TensorFlow/Keras. The original
project is built with PyTorch and focuses on edge-based convolutions over
triangle meshes. The implementation below captures the core idea of using
EdgeConv layers that operate on edge neighborhoods and aggregates the
features for classification tasks.

The code is intentionally lightweight to serve as a starting point for
further development. Real-world applications may require additional
features such as dynamic pooling, data augmentation, and handling batches
of meshes with varying sizes.
"""

from __future__ import annotations

import tensorflow as tf


class EdgeConv(tf.keras.layers.Layer):
    """Edge convolution layer used in MeshCNN.

    The layer applies an MLP to the concatenation of the central edge
    features and the difference to its neighbouring edge features. The
    result is aggregated via max pooling over the neighbourhood.

    Args:
        output_channels: Number of output feature channels.
    """

    def __init__(self, output_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(output_channels, activation="relu"),
            tf.keras.layers.Dense(output_channels),
        ])

    def call(self, x: tf.Tensor, neighbors: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        """Apply edge convolution.

        Args:
            x: Tensor of shape ``[num_edges, in_channels]`` containing edge
                features.
            neighbors: Tensor of shape ``[num_edges, k]`` containing the
                indices of neighbouring edges for each edge.

        Returns:
            Tensor of shape ``[num_edges, output_channels]`` with the updated
            edge features.
        """

        # Gather neighbour features ``[num_edges, k, in_channels]``
        neighbor_feat = tf.gather(x, neighbors)
        central_feat = tf.expand_dims(x, axis=1)
        # Combine central and relative features ``[num_edges, k, 2*in_channels]``
        h = tf.concat([central_feat, neighbor_feat - central_feat], axis=-1)
        h = self.mlp(h)
        # Aggregate over neighbours
        h = tf.reduce_max(h, axis=1)
        return h


class MeshCNNModel(tf.keras.Model):
    """Minimal MeshCNN model for edge-based classification.

    Args:
        num_classes: Number of classes to predict.
        hidden_channels: Number of hidden channels for the EdgeConv layers.
    """

    def __init__(self, num_classes: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = EdgeConv(hidden_channels)
        self.conv2 = EdgeConv(hidden_channels)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:  # type: ignore[override]
        x, neighbors = inputs
        # Apply two EdgeConv layers
        x = self.conv1(x, neighbors)
        x = self.conv2(x, neighbors)
        # Add batch dimension for pooling
        x = tf.expand_dims(x, axis=0)
        x = self.global_pool(x)
        return self.classifier(x)


def build_model(num_classes: int, hidden_channels: int = 64) -> MeshCNNModel:
    """Factory method to build a ``MeshCNNModel``.

    This helper is convenient when using ``tf.keras.Model.save`` or ``load_model``.
    """

    return MeshCNNModel(num_classes=num_classes, hidden_channels=hidden_channels)


__all__ = ["EdgeConv", "MeshCNNModel", "build_model"]
