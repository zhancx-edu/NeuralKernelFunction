import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, constraints, backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BackgroundFunction(layers.Layer):
    """Background function layer for modeling the baseline event rate."""
    def __init__(self):
        super().__init__()
        self.mu = tf.Variable(
            initial_value=[[[1.0]]],
            constraint=lambda x: tf.maximum(x, 1e-9),
            trainable=True,
            name='mu'
        )

    def call(self, obj_area, d_time):
        return self.mu * obj_area * d_time

class ProductionFunction(layers.Layer):
    """Production function layer for modeling magnitude-dependent event productivity."""
    def __init__(self, input_m0: float):
        super().__init__()
        self.mag_A = tf.Variable(
            initial_value=3.5,
            constraint=lambda x: tf.maximum(x, 1e-5),
            trainable=True,
            name='mag_A'
        )
        self.mag_alpha = tf.Variable(
            initial_value=0.35,
            constraint=lambda x: tf.maximum(x, 1e-5),
            trainable=True,
            name='mag_alpha'
        )
        self.mag_m0 = tf.constant(input_m0, name='mag_m0')

    def call(self, ht_mag):
        return self.mag_A * tf.exp(self.mag_alpha * (ht_mag - self.mag_m0))

class OmoriFunction(layers.Layer):
    """Omori function layer for modeling temporal decay of aftershocks."""
    def __init__(self):
        super().__init__()
        self.omori_c = tf.Variable(
            initial_value=0.01,
            constraint=lambda x: tf.maximum(x, 1e-5),
            trainable=True,
            name='omori_c'
        )
        self.omori_p = tf.Variable(
            initial_value=1.02,
            constraint=lambda x: tf.maximum(x, 1 + 1e-5),
            trainable=True,
            name='omori_p'
        )

    def call(self, interval_time):
        return 1 - tf.pow(1 + interval_time / self.omori_c, -self.omori_p + 1)

class SpatialFunction(layers.Layer):
    """Spatial function layer for modeling spatial distribution of aftershocks."""
    def __init__(self, input_m0: float):
        super().__init__()
        self.spatial_q = tf.Variable(
            initial_value=1.2,
            constraint=lambda x: tf.maximum(x, 1 + 1e-5),
            trainable=True,
            name='spatial_q'
        )
        self.spatial_D2 = tf.Variable(
            initial_value=0.001,
            constraint=lambda x: tf.maximum(x, 1e-5),
            trainable=True,
            name='spatial_D2'
        )
        self.spatial_gamma = tf.Variable(
            initial_value=0.01,
            constraint=lambda x: tf.maximum(x, 1e-5),
            trainable=True,
            name='spatial_gamma'
        )
        self.spatial_m0 = tf.constant(input_m0, name='spatial_m0')

    def call(self, dist_2, hist_m):
        return (self.spatial_q - 1) / (np.pi * self.spatial_D2 * tf.exp(self.spatial_gamma * (hist_m - self.spatial_m0))) * \
               (1 + dist_2 / (self.spatial_D2 * tf.exp(self.spatial_gamma * (hist_m - self.spatial_m0)))) ** (-self.spatial_q)

class TemporalKernelNetwork(layers.Layer):
    """Neural network layer for modeling the temporal kernel."""
    def __init__(self, size_nn: int, size_layer: int):
        super().__init__()
        self.size_nn = size_nn
        self.size_layer = size_layer
        self.layers = []

        def abs_glorot_uniform(shape, dtype=None, partition_info=None):
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))

        # First hidden layer
        self.layers.append(
            layers.Dense(
                self.size_nn, activation='tanh', use_bias=False,
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='temporal_dense_0'
            )
        )
        # Additional hidden layers
        for i in range(self.size_layer - 1):
            self.layers.append(
                layers.Dense(
                    self.size_nn, activation='tanh', use_bias=False,
                    kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                    name=f'temporal_dense_{i+1}'
                )
            )
        # Output layer
        self.layers.append(
            layers.Dense(
                1, activation='tanh', use_bias=False,
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='temporal_output'
            )
        )

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class SpatialKernelNetwork(layers.Layer):
    """Neural network layer for modeling the spatial kernel."""
    def __init__(self, size_nn: int, size_layer: int):
        super().__init__()
        self.size_nn = size_nn
        self.size_layer = size_layer
        self.layers = []

        def abs_glorot_uniform(shape, dtype=None, partition_info=None):
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))

        # First hidden layer
        self.layers.append(
            layers.Dense(
                self.size_nn, activation='tanh', use_bias=False,
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='spatial_dense_0'
            )
        )
        # Additional hidden layers
        for i in range(self.size_layer - 1):
            self.layers.append(
                layers.Dense(
                    self.size_nn, activation='tanh', use_bias=False,
                    kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                    name=f'spatial_dense_{i+1}'
                )
            )
        # Output layer
        self.layers.append(
            layers.Dense(
                1, activation='tanh', use_bias=False,
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='spatial_output'
            )
        )

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class KappaNetwork(layers.Layer):
    """Neural network layer for modeling the kappa function."""
    def __init__(self, size_nn: int, size_layer: int):
        super().__init__()
        self.size_nn = size_nn
        self.size_layer = size_layer
        self.layers = []

        def abs_glorot_uniform(shape, dtype=None, partition_info=None):
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))

        # First hidden layer
        self.layers.append(
            layers.Dense(
                self.size_nn, activation='tanh',
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='kappa_dense_0'
            )
        )
        # Additional hidden layers
        for i in range(self.size_layer - 1):
            self.layers.append(
                layers.Dense(
                    self.size_nn, activation='tanh',
                    kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                    name=f'kappa_dense_{i+1}'
                )
            )
        # Output layer
        self.layers.append(
            layers.Dense(
                1, activation='softplus',
                kernel_initializer=abs_glorot_uniform, kernel_constraint=constraints.NonNeg(),
                name='kappa_output'
            )
        )

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class UpdateMuCallback(Callback):
    """Callback to update the mu parameter after each epoch."""
    def __init__(self, seismic_model, session):
        super().__init__()
        self.seismic_model = seismic_model
        self.session = session

    def on_epoch_end(self, epoch, logs=None):

        predictions = self.seismic_model.model.predict(
            [self.seismic_model.dis_t1_train, self.seismic_model.dis_t2_train,
             self.seismic_model.hist_m_train, self.seismic_model.dis_xy_train],
            batch_size=512
        )
        lam_ts = predictions[0]
        mu_value = self.session.run(self.seismic_model.bac_func.mu)[0, 0, 0]
        ind_p = mu_value / lam_ts
        background_events = np.sum(ind_p)
        total_time = np.sum(self.seismic_model.elapsed_t_train)
        area = self.seismic_model.area_scalar
        mu_new = max(background_events / (total_time * area), 1e-9)
        # logger.info(f"Epoch {epoch + 1}: area = {area}, mu_new = {mu_new}")

        with tf.compat.v1.get_default_graph().as_default():
            assign_op = self.seismic_model.bac_func.mu.assign([[[mu_new]]])
            self.session.run(assign_op)
            tf.keras.backend.set_value(self.seismic_model.bac_func.mu, [[[mu_new]]])