import functools
from typing import Callable

import numpy as np

from .settings import DatasetSettings, ModelSettings


def get_dataset_settings(dataset: str) -> DatasetSettings:
    if dataset == "fmow_rgb":
        return DatasetSettings(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
        )
    elif dataset == "fmow_full":
        return DatasetSettings(
            mean=np.array(
                [349.23, 339.76, 378.58, 418.42, 275.86, 431.82, 495.65, 435.05]
            ),
            std=np.array(
                [78.67, 105.54, 142.05, 177.00, 132.29, 151.65, 194.00, 166.27]
            ),
        )
    elif dataset == "xbd_mexico":
        return DatasetSettings(
            mean=np.array(
                [91.49104789859993, 89.81207679717009, 70.44423856814045]
            ),
            std=np.array(
                [39.13155801901116, 34.71767087905296, 33.91283981227421]
            ),
        )
    elif dataset == "xbd_full":
        return DatasetSettings(
            mean=np.array(
                [87.9481, 89.2187, 67.5857]
            ),
            std=np.array(
                [45.2051, 36.5164, 33.8481]
            ),
        )
    else:
        raise ValueError("Invalid dataset type.")


def get_model_settings(model: str) -> ModelSettings:
    if model == "unet":
        return ModelSettings(name="unet")
    else:
        raise ValueError("Invalid dataset type.")


def _preprocess_fn(
    x: np.ndarray, model: ModelSettings, dataset: DatasetSettings
) -> np.ndarray:
    if dataset.mean is not None:
        x = x - dataset.mean

    if dataset.std is not None:
        x = x / dataset.std

    return x


def get_preprocessing_fn(model: str, dataset: str) -> Callable:
    """Get a preprocessing function for a given model and dataset. For a full
    list of preprocessing functions available, see the documentation.

    Args:
        model: Which type of model to preprocess for, e.g. unet.
        dataset: Which dataset to expect for preprocessing, e.g. fmow_rgb.

    Returns:
        fn: A function that can be applied to an input array.
    """
    model_settings = get_model_settings(model)
    dataset_settings = get_dataset_settings(dataset)

    return functools.partial(
        _preprocess_fn, model=model_settings, dataset=dataset_settings
    )
