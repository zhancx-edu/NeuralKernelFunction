# Spatio-Temporal Neural Kernel Function (ST-NKF) for seismicity modeling.

# Code adapted from:
# Samuel Stockman, Daniel J. Lawson, and Maximilian J. Werner.
# Forecasting the 2016–2017 Central Apennines earthquake sequence with a neural point process,
# Earth’s Future, 11(9): e2023EF003777, 2023.
# https://github.com/ss15859/Neural-Point-Process

# Datasets obtained from:
# Samuel Stockman, Daniel J. Lawson, and Maximilian J. Werner.
# Earthquakenpp: Benchmark datasets for earthquake forecasting with neural point processes,
# arXiv preprint arXiv:2410.08226, 2024.
# https://github.com/ss15859/EarthquakeNPP

import sys
import os

SEED = int(sys.argv[1])

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
os.environ["TF_DETERMINISTIC_OPS"] = "1"


import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.compat.v1.set_random_seed(SEED)



import logging

import pandas as pd

import warnings
from config import DATASET_CONFIGS
from utils import azimuthal_equidistant_projection, quadrilateral_area
from models import KernelPointProcess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")




# datasets_id should be in ["ComCat", "SaltonSea", "SanJac", "WHITE", "SCEDC_20", "SCEDC_25", "SCEDC_30"]

datasets_id = sys.argv[3]

temporal_id = sys.argv[4]
spatial_id = sys.argv[5]
kappa_id = sys.argv[6]


######## ETAS ########
# temporal_id = "empirical"
# spatial_id = "empirical"
# kappa_id = "empirical"
######## ETAS ########

######## NKF ########
# temporal_id = "neural"
# spatial_id = "neural"
# kappa_id = "neural"
######## NKF ########

# You can also try different combinations.
# such as:
# temporal_id = "neural"
# spatial_id = "neural"
# kappa_id = "empirical"


if datasets_id not in DATASET_CONFIGS:
    logger.error(f"Unknown dataset: {datasets_id}")
    sys.exit(1)

config = DATASET_CONFIGS[datasets_id]

try:
    raw_catalog = pd.read_csv(config["catalog_path"])
    cat_shape = np.load(config["shape_path"])
except FileNotFoundError as e:
    logger.error(f"Failed to load dataset files: {e}")
    sys.exit(1)

# Filter catalog for SCEDC datasets
if datasets_id in ["SCEDC_20", "SCEDC_25", "SCEDC_30"]:
    min_magnitude = float(datasets_id.split("_")[-1]) / 10
    raw_catalog = raw_catalog[raw_catalog['magnitude'] >= min_magnitude].reset_index(drop=True)

# Split data by time
auxiliary_num = len(raw_catalog[
    (raw_catalog['time_days'] >= config["auxiliary_start"]) & 
    (raw_catalog['time_days'] < config["training_start"])
])
training_num = len(raw_catalog[
    (raw_catalog['time_days'] >= config["training_start"]) & 
    (raw_catalog['time_days'] < config["validation_start"])
])
validation_num = len(raw_catalog[
    (raw_catalog['time_days'] >= config["validation_start"]) & 
    (raw_catalog['time_days'] < config["testing_start"])
])
testing_num = len(raw_catalog[
    (raw_catalog['time_days'] >= config["testing_start"]) & 
    (raw_catalog['time_days'] <= config["testing_end"])
])

logger.info(f"{datasets_id}: auxiliary={auxiliary_num}, training={training_num}, "
            f"validation={validation_num}, testing={testing_num}")

input_dim_train = auxiliary_num
data_t_train = raw_catalog['time_days'].values[auxiliary_num - input_dim_train: auxiliary_num + training_num]
data_m_train = raw_catalog['magnitude'].values[auxiliary_num - input_dim_train: auxiliary_num + training_num]
data_x_train = raw_catalog['x'].values[auxiliary_num - input_dim_train: auxiliary_num + training_num]
data_y_train = raw_catalog['y'].values[auxiliary_num - input_dim_train: auxiliary_num + training_num]

input_dim_val = auxiliary_num
data_t_val = raw_catalog['time_days'].values[
    auxiliary_num + training_num - input_dim_val: auxiliary_num + training_num + validation_num
]
data_m_val = raw_catalog['magnitude'].values[
    auxiliary_num + training_num - input_dim_val: auxiliary_num + training_num + validation_num
]
data_x_val = raw_catalog['x'].values[
    auxiliary_num + training_num - input_dim_val: auxiliary_num + training_num + validation_num
]
data_y_val = raw_catalog['y'].values[
    auxiliary_num + training_num - input_dim_val: auxiliary_num + training_num + validation_num
]

input_dim_test = auxiliary_num
data_t_test = raw_catalog['time_days'].values[
    auxiliary_num + training_num + validation_num - input_dim_test: 
    auxiliary_num + training_num + validation_num + testing_num
]
data_m_test = raw_catalog['magnitude'].values[
    auxiliary_num + training_num + validation_num - input_dim_test: 
    auxiliary_num + training_num + validation_num + testing_num
]
data_x_test = raw_catalog['x'].values[
    auxiliary_num + training_num + validation_num - input_dim_test: 
    auxiliary_num + training_num + validation_num + testing_num
]
data_y_test = raw_catalog['y'].values[
    auxiliary_num + training_num + validation_num - input_dim_test: 
    auxiliary_num + training_num + validation_num + testing_num
]

center_latitude = raw_catalog['latitude'].mean()
center_longitude = raw_catalog['longitude'].mean()
cat_shape_x, cat_shape_y = azimuthal_equidistant_projection(
    cat_shape[:, 0], cat_shape[:, 1], center_latitude, center_longitude
)
cat_shape_xy = np.stack((cat_shape_x, cat_shape_y), axis=1)
obj_area = quadrilateral_area(cat_shape_xy)

model = KernelPointProcess(
    time_step_train=input_dim_train,
    time_step_val=input_dim_val,
    time_step_test=input_dim_test,
    temporal_id=temporal_id,
    spatial_id=spatial_id,
    kappa_id=kappa_id,
    global_m0=config["global_m0"],
    area=obj_area,
    size_layer=5,
    size_nn=32,
    seed = SEED
).set_train_data(
    data_t_train, data_m_train, data_x_train, data_y_train
).set_val_data(
    data_t_val, data_m_val, data_x_val, data_y_val
).set_test_data(
    data_t_test, data_m_test, data_x_test, data_y_test
).set_model().compile().fit_eval(
    epochs=1000, batch_size=128
).eval_train().eval_val().eval_test().save_weights(
    f"weights/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}"
)

logger.info(f"Test results: LL_ts={model.LL_ts_average_test}, "
            f"LL_t={model.LL_t_average_test}, LL_s={model.LL_s_average_test}")

try:
    with open("ll_seismic_model.log", "a") as log_file:
        log_file.write(
            f"{datasets_id}, {temporal_id}, {spatial_id}, {kappa_id}, "
            f"{model.LL_ts_average_test}, {model.LL_t_average_test}, {model.LL_s_average_test}\n"
        )
except IOError as e:
    logger.error(f"Failed to write to log file: {e}")

results_catalog = raw_catalog.copy()
results_catalog.loc[auxiliary_num: auxiliary_num + training_num - 1, 'step'] = 'train'
results_catalog.loc[auxiliary_num + training_num: auxiliary_num + training_num + validation_num - 1, 'step'] = 'val'
results_catalog.loc[
    auxiliary_num + training_num + validation_num: 
    auxiliary_num + training_num + validation_num + testing_num - 1, 'step'
] = 'test'

for dataset, logli_t, logli_s, logli_ts, loglam_t, loglam_ts, intlam in [
    ('train', model.LL_t_train, model.LL_s_train, model.LL_ts_train, np.log(model.lam_t_train), np.log(model.lam_ts_train), model.Int_lam_train),
    ('val', model.LL_t_val, model.LL_s_val, model.LL_ts_val, np.log(model.lam_t_val), np.log(model.lam_ts_val), model.Int_lam_val),
    ('test', model.LL_t_test, model.LL_s_test, model.LL_ts_test, np.log(model.lam_t_test), np.log(model.lam_ts_test), model.Int_lam_test)
]:
    start_idx = {'train': auxiliary_num, 'val': auxiliary_num + training_num, 'test': auxiliary_num + training_num + validation_num}[dataset]
    end_idx = start_idx + {'train': training_num, 'val': validation_num, 'test': testing_num}[dataset] - 1
    results_catalog.loc[start_idx:end_idx, 'logli_t'] = logli_t.flatten()
    results_catalog.loc[start_idx:end_idx, 'logli_s'] = logli_s.flatten()
    results_catalog.loc[start_idx:end_idx, 'logli_ts'] = logli_ts.flatten()
    results_catalog.loc[start_idx:end_idx, 'loglam_t'] = loglam_t.flatten()
    results_catalog.loc[start_idx:end_idx, 'loglam_ts'] = loglam_ts.flatten()
    results_catalog.loc[start_idx:end_idx, 'intlam'] = intlam.flatten()

try:
    results_catalog.to_csv(
        f"csv/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}.csv",
        index=False,
        encoding='utf-8'
    )
    logger.info(f"Results saved to csv/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}.csv")
except IOError as e:
    logger.error(f"Failed to save results to CSV: {e}")


try:
    sorted_indices = np.argsort(model.dis_xy_train[:-1].flatten())
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_dis_xy_train.npy", model.dis_xy_train[:-1].flatten()[sorted_indices])
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_pdf_spatial_train.npy", model.pdf_spatial_train[:-1].flatten()[sorted_indices])
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_cdf_spatial_train.npy", model.cdf_spatial_train[:-1].flatten()[sorted_indices])
    
    sorted_indices = np.argsort(model.dis_t1_train[:-1].flatten())
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_dis_t_train.npy", model.dis_t1_train[:-1].flatten()[sorted_indices])
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_pdf_temporal_train.npy", model.pdf_temporal_train[:-1].flatten()[sorted_indices])
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_cdf_temporal_train.npy", model.cdf_temporal_train[:-1].flatten()[sorted_indices])

    sorted_indices = np.argsort(model.hist_m_train[:-1].flatten())
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_hist_m_train.npy", model.hist_m_train[:-1].flatten()[sorted_indices])
    np.save(f"results/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}_kappa_train.npy", model.kappa_train[:-1].flatten()[sorted_indices])


except IOError as e:
    logger.error(f"Failed to save pattern: {e}")    


if temporal_id == "neural" and spatial_id == "neural" and kappa_id == "neural": 

    try:
        updated_mu = tf.keras.backend.get_value(model.bac_func.mu)[0,0,0]

        print("Trainable vector after training:", updated_mu)
        with open("etas_parameters.log", "a") as log_file:
            log_file.write(
                f"{datasets_id}, {updated_mu}\n"
            )
            
    except IOError as e:
        logger.error(f"Failed to save parameters: {e}")    

elif temporal_id == "empiricl" and spatial_id == "empiricl" and kappa_id == "empiricl": 

    try:
        updated_mu = tf.keras.backend.get_value(model.bac_func.mu)[0,0,0]
        updated_A = tf.keras.backend.get_value(model.produc_func.mag_A)
        updated_alpha = tf.keras.backend.get_value(model.produc_func.mag_alpha)
        updated_c = tf.keras.backend.get_value(model.omori_func.omori_c)
        updated_p = tf.keras.backend.get_value(model.omori_func.omori_p)
        updated_q = tf.keras.backend.get_value(model.spatial_func.spatial_q)
        updated_d2 = tf.keras.backend.get_value(model.spatial_func.spatial_D2)
        updated_gamma = tf.keras.backend.get_value(model.spatial_func.spatial_gamma)
        
        
        print("Trainable vector after training:", updated_mu, updated_A, updated_alpha, updated_c, updated_p, updated_q, updated_d2, updated_gamma)
        with open("etas_parameters.log", "a") as log_file:
            log_file.write(
                f"{datasets_id}, {updated_mu}, {updated_A}, {updated_alpha}, {updated_c}, {updated_p}, {updated_q}, {updated_d2}, {updated_gamma}\n"
            )
            
    except IOError as e:
        logger.error(f"Failed to save parameters: {e}")    