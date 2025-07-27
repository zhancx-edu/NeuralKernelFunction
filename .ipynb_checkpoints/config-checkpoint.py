import pandas as pd

DATASET_CONFIGS = {
    "ComCat": {
        "catalog_path": "./Datasets/ComCat/ComCat_catalog.csv",
        "shape_path": "./Datasets/ComCat/california_shape.npy",
        "auxiliary_start": pd.Timestamp('1971-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('1981-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('1998-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2007-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2020-01-17 00:00:00').timestamp() / 86400,
        "global_m0": 2.5
    },
    "SaltonSea": {
        "catalog_path": "./Datasets/QTM/SaltonSea_catalog.csv",
        "shape_path": "./Datasets/QTM/SaltonSea_shape.npy",
        "auxiliary_start": pd.Timestamp('2008-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('2009-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2016-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2018-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 1.0
    },
    "SanJac": {
        "catalog_path": "./Datasets/QTM/SanJac_catalog.csv",
        "shape_path": "./Datasets/QTM/SanJac_shape.npy",
        "auxiliary_start": pd.Timestamp('2008-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('2009-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2016-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2018-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 1.0
    },
    "WHITE": {
        "catalog_path": "./Datasets/WHITE/WHITE_catalog.csv",
        "shape_path": "./Datasets/WHITE/WHITE_shape.npy",
        "auxiliary_start": pd.Timestamp('2008-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('2009-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2017-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2021-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 0.6
    },
    "SCEDC_20": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": pd.Timestamp('1981-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('1985-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2005-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2020-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 2.0
    },
    "SCEDC_25": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": pd.Timestamp('1981-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('1985-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2005-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2020-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 2.5
    },
    "SCEDC_30": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": pd.Timestamp('1981-01-01 00:00:00').timestamp() / 86400,
        "training_start": pd.Timestamp('1985-01-01 00:00:00').timestamp() / 86400,
        "validation_start": pd.Timestamp('2005-01-01 00:00:00').timestamp() / 86400,
        "testing_start": pd.Timestamp('2014-01-01 00:00:00').timestamp() / 86400,
        "testing_end": pd.Timestamp('2020-01-01 00:00:00').timestamp() / 86400,
        "global_m0": 3.0
    }
}