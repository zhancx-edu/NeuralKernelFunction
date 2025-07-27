import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
import numpy as np
from typing import Tuple
from layers import (
    BackgroundFunction,
    ProductionFunction,
    OmoriFunction,
    SpatialFunction,
    TemporalKernelNetwork,
    SpatialKernelNetwork,
    KappaNetwork,
    UpdateMuCallback
)
from utils import process_data
import logging

logger = logging.getLogger(__name__)

class KernelPointProcess:
    """Point process model for seismic event forecasting, supporting ETAS and neural network components."""
    def __init__(
        self,
        time_step_train: int,
        time_step_val: int,
        time_step_test: int,
        temporal_id: str,
        spatial_id: str,
        kappa_id: str,
        global_m0: float,
        area: float,
        size_layer: int,
        size_nn: int
    ):
        self.time_step_train = time_step_train
        self.time_step_val = time_step_val
        self.time_step_test = time_step_test
        self.area = tf.constant([[[area]]], name='area')
        self.area_scalar = area
        self.size_layer = size_layer
        self.size_nn = size_nn
        self.temporal_id = temporal_id
        self.spatial_id = spatial_id
        self.kappa_id = kappa_id
        self.bac_func = BackgroundFunction()
        self.produc_func = ProductionFunction(global_m0)
        self.omori_func = OmoriFunction()
        self.spatial_func = SpatialFunction(global_m0)
        self.temporal_net = TemporalKernelNetwork(size_nn, size_layer)
        self.spatial_net = SpatialKernelNetwork(size_nn, size_layer)
        self.kappa_net = KappaNetwork(size_nn, size_layer)

    def set_train_data(self, times: np.ndarray, mags: np.ndarray, locs_x: np.ndarray, locs_y: np.ndarray) -> 'SeismicPointProcess':
        """Set training data."""
        self.hist_t_train, self.hist_m_train, self.hist_x_train, self.hist_y_train, \
        self.cur_t_train, self.cur_x_train, self.cur_y_train, self.dis_xy_train, \
        self.dis_t1_train, self.dis_t2_train, self.elapsed_t_train = process_data(
            self.time_step_train, times, mags, locs_x, locs_y
        )
        return self

    def set_val_data(self, times: np.ndarray, mags: np.ndarray, locs_x: np.ndarray, locs_y: np.ndarray) -> 'SeismicPointProcess':
        """Set validation data."""
        self.hist_t_val, self.hist_m_val, self.hist_x_val, self.hist_y_val, \
        self.cur_t_val, self.cur_x_val, self.cur_y_val, self.dis_xy_val, \
        self.dis_t1_val, self.dis_t2_val, self.elapsed_t_val = process_data(
            self.time_step_val, times, mags, locs_x, locs_y
        )
        return self

    def set_test_data(self, times: np.ndarray, mags: np.ndarray, locs_x: np.ndarray, locs_y: np.ndarray) -> 'SeismicPointProcess':
        """Set test data."""
        self.hist_t_test, self.hist_m_test, self.hist_x_test, self.hist_y_test, \
        self.cur_t_test, self.cur_x_test, self.cur_y_test, self.dis_xy_test, \
        self.dis_t1_test, self.dis_t2_test, self.elapsed_t_test = process_data(
            self.time_step_test, times, mags, locs_x, locs_y
        )
        return self

    def set_model(self) -> 'SeismicPointProcess':
        """Define and initialize the neural network model."""
        max_time = np.max(self.dis_t1_train)
        max_mag = np.max(self.hist_m_train)
        max_dis = np.max(self.dis_xy_train)

        # Inputs
        time_in_1 = layers.Input(shape=(None, 1), name='time_in_1')
        time_in_2 = layers.Input(shape=(None, 1), name='time_in_2')
        mags_in_1 = layers.Input(shape=(None, 1), name='mags_in_1')
        mags_in_2 = layers.Lambda(lambda x: x[:, :-1, :], name='mags_in_2')(mags_in_1)
        dis_in = layers.Input(shape=(None, 1), name='dis_in')

        time_in_1_nmlz = layers.Lambda(lambda x: x / max_time, name='time_in_1_nmlz')(time_in_1)
        time_in_2_nmlz = layers.Lambda(lambda x: x / max_time, name='time_in_2_nmlz')(time_in_2)
        mags_in_1_nmlz = layers.Lambda(lambda x: x / max_mag, name='mags_in_1_nmlz')(mags_in_1)
        mags_in_2_nmlz = layers.Lambda(lambda x: x / max_mag, name='mags_in_2_nmlz')(mags_in_2)
        dis_in_nmlz = layers.Lambda(lambda x: x / max_dis, name='dis_in_nmlz')(dis_in)

        if self.spatial_id == "neural":
            cdf_spatial = self.spatial_net(dis_in_nmlz)
            d_v_r = K.gradients(cdf_spatial, dis_in)[0]
            pdf_spatial = layers.Multiply()([d_v_r, 1.0 / (2 * np.pi * dis_in)])
        elif self.spatial_id == "empirical":
            dis_in_square = layers.Lambda(lambda x: x ** 2)(dis_in)
            pdf_spatial = self.spatial_func(dis_in_square, mags_in_1)
        else:
            raise ValueError(f"Invalid spatial_id: {self.spatial_id}")

        if self.temporal_id == "neural":
            cdf_temporal_1 = self.temporal_net(time_in_1_nmlz)
            cdf_temporal_2 = self.temporal_net(time_in_2_nmlz)
        elif self.temporal_id == "empirical":
            cdf_temporal_1 = self.omori_func(time_in_1)
            cdf_temporal_2 = self.omori_func(time_in_2)
        else:
            raise ValueError(f"Invalid temporal_id: {self.temporal_id}")

        if self.kappa_id == "neural":
            kappa_1 = self.kappa_net(mags_in_1_nmlz)
            kappa_2 = self.kappa_net(mags_in_2_nmlz)
        elif self.kappa_id == "empirical":
            kappa_1 = self.produc_func(mags_in_1)
            kappa_2 = self.produc_func(mags_in_2)
        else:
            raise ValueError(f"Invalid kappa_id: {self.kappa_id}")

        Int_tri_1 = layers.Multiply()([cdf_temporal_1, kappa_1])
        Int_tri_2 = layers.Multiply()([cdf_temporal_2, kappa_2])
        Int_tri_1_sum = K.sum(Int_tri_1, axis=1, keepdims=True)
        Int_tri_2_sum = K.sum(Int_tri_2, axis=1, keepdims=True)
        Int_tri = layers.Subtract()([Int_tri_1_sum, Int_tri_2_sum])
        Int_l = Int_tri + self.bac_func.mu * self.area * time_in_1[:, -1:, :]

        pdf_temporal_1 = K.gradients(cdf_temporal_1, time_in_1)[0]
        lam_tri_ts = layers.Multiply()([kappa_1, pdf_temporal_1, pdf_spatial])
        lam_tri_t = layers.Multiply()([kappa_1, pdf_temporal_1])
        lam_tri_ts_sum = K.sum(lam_tri_ts, axis=1, keepdims=True)
        lam_tri_t_sum = K.sum(lam_tri_t, axis=1, keepdims=True)

        l_ts = lam_tri_ts_sum + self.bac_func.mu
        l_t = lam_tri_t_sum + self.bac_func.mu * self.area

        self.model = Model(
            inputs=[time_in_1, time_in_2, mags_in_1, dis_in],
            outputs=[l_ts, l_t, Int_l, pdf_temporal_1, cdf_temporal_1, pdf_spatial, kappa_1]
        )
        self.model.add_loss(-K.mean(K.log(l_ts) - Int_l))

        self.session = tf.compat.v1.Session()
        with tf.compat.v1.get_default_graph().as_default():
            self.session.run(tf.compat.v1.global_variables_initializer())
        logger.info("Model and session initialized")
        return self

    def compile(self, lr: float = 0.005) -> 'SeismicPointProcess':
        """Compile the model with Adam optimizer."""
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        with tf.compat.v1.get_default_graph().as_default():
            self.model.compile(optimizer=optimizer)
        return self

    def save_weights(self, path: str) -> 'SeismicPointProcess':
        """Save model weights to the specified path."""
        try:
            self.model.save_weights(path)
            logger.info(f"Weights saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
        return self

    def load_weights(self, path: str) -> 'SeismicPointProcess':
        """Load model weights from the specified path."""
        try:
            self.model.load_weights(path)
            logger.info(f"Weights loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
        return self

    def summary(self):
        """Print model summary."""
        return self.model.summary()

    def fit_eval(self, epochs: int = 1000, batch_size: int = 256) -> 'SeismicPointProcess':
        """Train and evaluate the model."""
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        mu_callback = UpdateMuCallback(self, self.session)
        history = self.model.fit(
            [self.dis_t1_train, self.dis_t2_train, self.hist_m_train, self.dis_xy_train],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([self.dis_t1_val, self.dis_t2_val, self.hist_m_val, self.dis_xy_val], []),
            callbacks=[mu_callback, es],
            shuffle=True,
            verbose=1
        )
        return self

    def eval_train(self, batch_size: int = 128) -> 'SeismicPointProcess':
        """Evaluate the model on training data."""
        outputs = self.model.predict(
            [self.dis_t1_train, self.dis_t2_train, self.hist_m_train, self.dis_xy_train],
            batch_size=batch_size
        )
        self.lam_ts_train, self.lam_t_train, self.Int_lam_train, self.pdf_temporal_train, \
        self.cdf_temporal_train, self.pdf_spatial_train, self.kappa_train = outputs
        self.LL_ts_train = np.log(self.lam_ts_train) - self.Int_lam_train
        self.LL_t_train = np.log(self.lam_t_train) - self.Int_lam_train
        self.LL_s_train = self.LL_ts_train - self.LL_t_train
        self.LL_ts_average_train = np.mean(self.LL_ts_train)
        self.LL_t_average_train = np.mean(self.LL_t_train)
        self.LL_s_average_train = np.mean(self.LL_s_train)
        return self

    def eval_val(self, batch_size: int = 128) -> 'SeismicPointProcess':
        """Evaluate the model on validation data."""
        outputs = self.model.predict(
            [self.dis_t1_val, self.dis_t2_val, self.hist_m_val, self.dis_xy_val],
            batch_size=batch_size
        )
        self.lam_ts_val, self.lam_t_val, self.Int_lam_val, self.pdf_temporal_val, \
        self.cdf_temporal_val, self.pdf_spatial_val, self.kappa_val = outputs
        self.LL_ts_val = np.log(self.lam_ts_val) - self.Int_lam_val
        self.LL_t_val = np.log(self.lam_t_val) - self.Int_lam_val
        self.LL_s_val = self.LL_ts_val - self.LL_t_val
        self.LL_ts_average_val = np.mean(self.LL_ts_val)
        self.LL_t_average_val = np.mean(self.LL_t_val)
        self.LL_s_average_val = np.mean(self.LL_s_val)
        return self

    def eval_test(self, batch_size: int = 128) -> 'SeismicPointProcess':
        """Evaluate the model on test data."""
        outputs = self.model.predict(
            [self.dis_t1_test, self.dis_t2_test, self.hist_m_test, self.dis_xy_test],
            batch_size=batch_size
        )
        self.lam_ts_test, self.lam_t_test, self.Int_lam_test, self.pdf_temporal_test, \
        self.cdf_temporal_test, self.pdf_spatial_test, self.kappa_test = outputs
        self.LL_ts_test = np.log(self.lam_ts_test) - self.Int_lam_test
        self.LL_t_test = np.log(self.lam_t_test) - self.Int_lam_test
        self.LL_s_test = self.LL_ts_test - self.LL_t_test
        self.LL_ts_average_test = np.mean(self.LL_ts_test)
        self.LL_t_average_test = np.mean(self.LL_t_test)
        self.LL_s_average_test = np.mean(self.LL_s_test)
        return self