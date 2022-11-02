import numpy as np


def horizon_shift_vector(
    num_directions,
    radius_pixels,
    min_radius,
    n_points,
    jitter_factor,
):
    """
    Calculates Sky-View determination movements.
    Parameters
    ----------
    num_directions : int
        Number of directions as input.
    radius_pixels : int
        Radius to consider in pixels (not in meters).
    min_radius : int
        Radius to start searching for horizon in pixels (not in meters).
    Returns
    -------
    shift : dict
        Dict with keys corresponding to the directions of search azimuths rounded to 1 decimal number
            - for each key, a subdict contains a key "shift":
                values for this key is a list of tuples prepared for np.roll - shift along lines and columns
            - the second key is "distance":
                values for this key is a list of search radius used for the computation of the elevation angle
    """

    # Generate angles and corresponding normal shifts in X (columns)
    # and Y (lines) direction
    angle_increment = 2 * np.pi / num_directions

    jitter_factor *= angle_increment
    angle_jitter = np.random.random(n_points) * jitter_factor - (jitter_factor / 2)

    # Generate a range of radius values in pixels.
    radii = np.linspace(min_radius**0.5, radius_pixels**0.5, num=n_points) ** 2

    # For each direction compute all possible horizont point position
    # and round them to integers
    for i in range(num_directions):
        angle = angle_increment * i + angle_jitter
        ij = radii[np.newaxis, :] * [np.cos(angle), np.sin(angle)]
        shift_pairs = np.unique(ij.astype(int), axis=1)
        distance = np.sqrt(np.sum(shift_pairs**2, axis=0))
        yield angle, shift_pairs[:, distance > 0], distance[distance > 0]


def sky_view_factor_compute(
    height_arr,
    radius_max,
    radius_min,
    num_directions,
    n_points,
    jitter_factor,
    no_data,
):
    """
    Calculates horizon based visualizations: Sky-view factor, Anisotopic SVF and Openess.
    Parameters
    ----------
    height_arr : numpy.ndarray
        Elevation (DEM) as 2D numpy array.
    radius_max : int
        Maximal search radius in pixels/cells (not in meters).
    radius_min : int
        Minimal search radius in pixels/cells (not in meters), for noise reduction.
    num_directions : int
        Number of directions as input.
    no_data : int or float
        Value that represents no_data, all pixels with this value are changed to np.nan .
    Returns
    -------
    dict_out : dictionary
        Return {"svf": svf_out, "asvf": asvf_out, "opns": opns_out};
        svf_out, skyview factor : 2D numpy array (numpy.ndarray) of skyview factor;
        asvf_out, anisotropic skyview factor : 2D numpy array (numpy.ndarray) of anisotropic skyview factor;
        opns_out, openness : 2D numpy array (numpy.ndarray) openness (elevation angle of horizon).
    """
    # change no_data to np.nan
    if no_data is not None:
        height_arr[height_arr == no_data] = -9999.0

    # determine the subsection for which we can compute the skyview factor
    i, j = np.meshgrid(
        *[np.arange(radius_max, s - radius_max) for s in height_arr.shape],
        indexing="ij"
    )
    out_shape = tuple([s - 2 * radius_max for s in height_arr.shape])
    roi = height_arr[
        tuple([slice(radius_max, s - radius_max) for s in height_arr.shape])
    ]
    assert not any(x < 1 for x in out_shape)

    # compute the vector of movement and corresponding distances
    move_generator = horizon_shift_vector(
        num_directions=num_directions,
        radius_pixels=radius_max,
        min_radius=radius_min,
        n_points=n_points,
        jitter_factor=jitter_factor,
    )

    svf_out = np.zeros(out_shape, dtype=np.float32)

    for _, shift, distance in move_generator:
        dh = (
            height_arr[i[..., np.newaxis] + shift[0], j[..., np.newaxis] + shift[1]]
            - roi[..., np.newaxis]
        )
        max_slope = (dh / distance).max(axis=2)
        svf_out += 1 - np.sin(np.maximum(np.arctan(max_slope), 0))

    svf_out /= num_directions
    return {"svf": svf_out}


def sky_view_factor(
    dem,
    resolution,
    svf_n_dir=16,
    svf_n_points=50,
    svf_r_max=10,
    svf_noise=0,
    svf_jitter=0.0,
    no_data=None,
):
    """
    Prepare the data, call sky_view_factor_compute, reformat and return back 2D arrays.
    Parameters
    ----------
    dem : numpy.ndarray
        Input digital elevation model as 2D numpy array.
    resolution : float
        Pixel resolution.
    svf_n_dir : int
        Number of directions.
    svf_r_max : int
        Maximal search radius in pixels.
    svf_noise : int
        The level of noise remove (0-don't remove, 1-low, 2-med, 3-high).
    no_data : int or float
        Value that represents no_data, all pixels with this value are changed to np.nan .

    """
    if dem.ndim != 2:
        raise Exception("rvt.visualization.sky_view_factor: dem has to be 2D np.array!")
    if svf_noise != 0 and svf_noise != 1 and svf_noise != 2 and svf_noise != 3:
        raise Exception(
            "rvt.visualization.sky_view_factor: svf_noise must be one of the following values (0-don't remove, 1-low,"
            " 2-med, 3-high)!"
        )
    if resolution < 0:
        raise Exception(
            "rvt.visualization.sky_view_factor: resolution must be a positive number!"
        )

    # TODO: proper check of input data: DEM 2D nummeric array, resolution, max_radius....

    dem = dem.astype(np.float32)

    # the portion (percent) of the maximal search radius to ignore in horizon estimation; for each noise level,
    # selected with in_svf_noise (0-3)
    sc_svf_r_min = [0.0, 10.0, 20.0, 40.0]

    # pixel size
    dem = dem / resolution

    # minimal search radious depends on the noise level, it has to be an integer not smaller than 1
    svf_r_min = max(np.round(svf_r_max * sc_svf_r_min[svf_noise] * 0.01, decimals=0), 1)

    dict_svf_asvf_opns = sky_view_factor_compute(
        height_arr=dem,
        radius_max=svf_r_max,
        radius_min=svf_r_min,
        num_directions=svf_n_dir,
        n_points=svf_n_points,
        jitter_factor=svf_jitter,
        no_data=no_data,
    )

    return dict_svf_asvf_opns



def shadow_factor_compute(
    height_arr,
    radius_max,
    radius_min,
    num_directions,
    n_points,
    jitter_factor,
    no_data,
):
    """
    Calculates horizon based visualizations: Sky-view factor, Anisotopic SVF and Openess.
    Parameters
    ----------
    height_arr : numpy.ndarray
        Elevation (DEM) as 2D numpy array.
    radius_max : int
        Maximal search radius in pixels/cells (not in meters).
    radius_min : int
        Minimal search radius in pixels/cells (not in meters), for noise reduction.
    num_directions : int
        Number of directions as input.
    no_data : int or float
        Value that represents no_data, all pixels with this value are changed to np.nan .
    Returns
    -------
    dict_out : dictionary
        Return {"svf": svf_out, "asvf": asvf_out, "opns": opns_out};
        svf_out, skyview factor : 2D numpy array (numpy.ndarray) of skyview factor;
        asvf_out, anisotropic skyview factor : 2D numpy array (numpy.ndarray) of anisotropic skyview factor;
        opns_out, openness : 2D numpy array (numpy.ndarray) openness (elevation angle of horizon).
    """
    # change no_data to np.nan
    if no_data is not None:
        height_arr[height_arr == no_data] = -9999.0

    # determine the subsection for which we can compute the skyview factor
    i, j = np.meshgrid(
        *[np.arange(radius_max, s - radius_max) for s in height_arr.shape],
        indexing="ij"
    )
    out_shape = tuple([s - 2 * radius_max for s in height_arr.shape])
    roi = height_arr[
        tuple([slice(radius_max, s - radius_max) for s in height_arr.shape])
    ]
    assert not any(x < 1 for x in out_shape)

    # compute the vector of movement and corresponding distances
    move_generator = horizon_shift_vector(
        num_directions=num_directions,
        radius_pixels=radius_max,
        min_radius=radius_min,
        n_points=n_points,
        jitter_factor=jitter_factor,
    )

    svf_out = np.zeros(out_shape, dtype=np.float32)

    for angle, shift, distance in move_generator:
        if angle[0] == 0.0:
            sun_angle = 30.0
        else:
            continue
        slope = distance * np.tan(sun_angle / 180 * np.pi)

        dh = (
            height_arr[i[..., np.newaxis] + shift[0], j[..., np.newaxis] + shift[1]]
            - roi[..., np.newaxis]
        )
        shadow = (np.sign(np.diff(np.maximum.accumulate((dh - slope), axis=2), axis=2)) == 0).any(axis=2)
        svf_out += shadow

    svf_out /= num_directions
    return {"svf": svf_out}


def shadow_factor(
    dem,
    resolution,
    svf_n_dir=16,
    svf_n_points=50,
    svf_r_max=10,
    svf_noise=0,
    svf_jitter=0.0,
    no_data=None,
):
    """
    Prepare the data, call sky_view_factor_compute, reformat and return back 2D arrays.
    Parameters
    ----------
    dem : numpy.ndarray
        Input digital elevation model as 2D numpy array.
    resolution : float
        Pixel resolution.
    svf_n_dir : int
        Number of directions.
    svf_r_max : int
        Maximal search radius in pixels.
    svf_noise : int
        The level of noise remove (0-don't remove, 1-low, 2-med, 3-high).
    no_data : int or float
        Value that represents no_data, all pixels with this value are changed to np.nan .

    """
    if dem.ndim != 2:
        raise Exception("rvt.visualization.sky_view_factor: dem has to be 2D np.array!")
    if svf_noise != 0 and svf_noise != 1 and svf_noise != 2 and svf_noise != 3:
        raise Exception(
            "rvt.visualization.sky_view_factor: svf_noise must be one of the following values (0-don't remove, 1-low,"
            " 2-med, 3-high)!"
        )
    if resolution < 0:
        raise Exception(
            "rvt.visualization.sky_view_factor: resolution must be a positive number!"
        )

    dem = dem.astype(np.float32)

    # the portion (percent) of the maximal search radius to ignore in horizon estimation; for each noise level,
    # selected with in_svf_noise (0-3)
    sc_svf_r_min = [0.0, 10.0, 20.0, 40.0]

    # pixel size
    dem = dem / resolution

    # minimal search radious depends on the noise level, it has to be an integer not smaller than 1
    svf_r_min = max(np.round(svf_r_max * sc_svf_r_min[svf_noise] * 0.01, decimals=0), 1)

    dict_svf_asvf_opns = shadow_factor_compute(
        height_arr=dem,
        radius_max=svf_r_max,
        radius_min=svf_r_min,
        num_directions=svf_n_dir,
        n_points=svf_n_points,
        jitter_factor=svf_jitter,
        no_data=no_data,
    )

    return dict_svf_asvf_opns