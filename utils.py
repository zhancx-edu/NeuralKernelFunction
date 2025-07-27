import numpy as np
from typing import List, Tuple, Union

def quadrilateral_area(coords: List[Tuple[float, float]]) -> float:
    """Calculate the area of a quadrilateral given its vertices in clockwise or counterclockwise order.

    Args:
        coords: List of 4 tuples, each containing (x, y) coordinates of the quadrilateral vertices.

    Returns:
        float: The area of the quadrilateral.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    area = abs(x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)) / 2
    return area

def azimuthal_equidistant_projection(
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    center_latitude: float,
    center_longitude: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert geographic coordinates to azimuthal equidistant projection.

    Args:
        latitude: Latitude(s) of the point(s).
        longitude: Longitude(s) of the point(s).
        center_latitude: Latitude of the projection center.
        center_longitude: Longitude of the projection center.

    Returns:
        Tuple of (x, y) coordinates in the projected plane.
    """
    R = 6371  # Earth's radius in kilometers
    phi1 = np.radians(center_latitude)
    phi2 = np.radians(latitude)
    delta_lambda = np.radians(longitude - center_longitude)

    delta_sigma = np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    azimuth = np.arctan2(np.sin(delta_lambda), np.cos(phi1) * np.tan(phi2) - np.sin(phi1) * np.cos(delta_lambda))

    x = R * delta_sigma * np.cos(azimuth)
    y = R * delta_sigma * np.sin(azimuth)
    return x, y

def process_data(
    time_step: int,
    times: np.ndarray,
    mags: np.ndarray,
    locs_x: np.ndarray,
    locs_y: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Process input data for training, validation, or testing.

    Args:
        time_step: Number of time steps for historical data.
        times: Array of event times.
        mags: Array of event magnitudes.
        locs_x: Array of x-coordinates.
        locs_y: Array of y-coordinates.

    Returns:
        Tuple of processed data arrays.
    """
    n = mags.shape[0]
    hist_t = np.array([times[i:i+time_step] for i in range(n-time_step)]).reshape(n-time_step, time_step, 1)
    hist_m = np.array([mags[i:i+time_step] for i in range(n-time_step)]).reshape(n-time_step, time_step, 1)
    hist_x = np.array([locs_x[i:i+time_step] for i in range(n-time_step)]).reshape(n-time_step, time_step, 1)
    hist_y = np.array([locs_y[i:i+time_step] for i in range(n-time_step)]).reshape(n-time_step, time_step, 1)
    
    cur_t = times[-n+time_step:].reshape(n-time_step, 1, 1)
    cur_x = locs_x[-n+time_step:].reshape(n-time_step, 1, 1)
    cur_y = locs_y[-n+time_step:].reshape(n-time_step, 1, 1)
    
    dis_xy = ((cur_x - hist_x) ** 2 + (cur_y - hist_y) ** 2) ** 0.5
    dis_t1 = cur_t - hist_t
    dis_t2 = hist_t[:, -1:, :] - hist_t[:, :-1, :]
    elapsed_time = np.ediff1d(times[-n+time_step-1:]).reshape(n-time_step, 1, 1)
    
    return hist_t, hist_m, hist_x, hist_y, cur_t, cur_x, cur_y, dis_xy, dis_t1, dis_t2, elapsed_time