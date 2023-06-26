import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from coregister.utils import em_nm_to_voxels
from coregister.transform.transform import Transform

import datajoint as dj
nda = dj.create_virtual_module('microns_phase3_nda','microns_phase3_nda')

def get_grid(field_key, desired_res=1):
    """ Get registered grid for this registration. """

    # Get field
    field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')

    # Create grid at desired resolution
    grid = create_grid(field_dims, desired_res=desired_res)  # h x w x 2
    grid = torch.as_tensor(grid, dtype=torch.double)

    # Apply required transform
    params = (nda.Registration & field_key).fetch1(
        'a11', 'a21', 'a31', 
        'a12', 'a22', 'a32', 
        'reg_x', 'reg_y', 'reg_z'
        )
    a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z = params
    linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]], dtype=torch.double)
    translation = torch.tensor([delta_x, delta_y, delta_z], dtype=torch.double)

    return affine_product(grid, linear, translation).numpy()


def create_grid(um_sizes, desired_res=1):
    """ Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
     um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).
    In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
    centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
    are NOT at -2 and 2 which is the assumption in some libraries.
    :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
    :param float or tuple desired_res: Desired resolution (um/px) for the grid.
    :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
    each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
    coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
    p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.double)
                for s, res in zip(out_sizes, desired_res)] # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid


def affine_product(X, A, b):
    """ Special case of affine transformation that receives coordinates X in 2-d (x, y)
    and affine matrix A and translation vector b in 3-d (x, y, z). Y = AX + b
    :param torch.Tensor X: A matrix of 2-d coordinates (d1 x d2 x 2).
    :param torch.Tensor A: The first two columns of the affine matrix (3 x 2).
    :param torch.Tensor b: A 3-d translation vector.
    :return: A (d1 x d2 x 3) torch.Tensor corresponding to the transformed coordinates.
    """
    return torch.einsum('ij,klj->kli', (A, X)) + b


def format_coords(coords_xyz, return_dim=1):
    # format coordinates 
    coords_xyz = np.array(coords_xyz)
    
    assert return_dim==1 or return_dim==2, 'return_dim must be 1 or 2'
    assert coords_xyz.ndim == 1 or coords_xyz.ndim == 2, 'Coordinate(s) must be 1D or 2D'
    assert coords_xyz.shape[-1] == 3, 'Coordinate(s) must have exactly x, y, and z'
    
    coords_xyz = coords_xyz if coords_xyz.ndim == return_dim else np.expand_dims(coords_xyz, 0)
        
    return coords_xyz


def fetch_coreg(transform_id=None, transform_version=None, transform_direction=None, transform_type=None, as_dict=True):
    """
    Fetch and initialize coregistration transform from nda

    :param transform_id: ID of desired transformation
    :param transform_version: "phase2" or "phase3"
    :param transform_direction: 
        "EM2P" --> provided coordinate is EM and output is 2P
        "2PEM" --> provided coordinate is 2P and output is EM
    :param transform_type:
        "linear" --> more nonrigid transform
        "spline" --> more rigid transform
    :param as_dict (bool):
        True - returns dictionary
        False - returns tuple
    
    Returns:
        dict or tuple of transform_id, transform_version, transform_direction, transform_type, transform_obj to pass to function coreg_transform
    """
    if transform_version is not None:
        assert transform_version=='phase2' or transform_version=='phase3', "transform_version must be 'phase2' or 'phase3'"
    if transform_direction is not None:
        assert transform_direction=="EM2P" or transform_direction=="2PEM", "transform_direction must be 'EM2P' or '2PEM"
    if transform_type is not None:
        assert transform_type=='linear' or transform_type=='spline', "transform_version must be 'linear' or 'spline'"

    
    # fetch desired transform solution
    tid_restr = [{'transform_id': transform_id} if transform_id is not None else {}]
    ver_restr = [{'version':transform_version} if transform_version is not None else {}]
    dir_restr = [{'direction': transform_direction} if transform_direction is not None else {}]
    type_restr = [{'transform_type': transform_type} if transform_type is not None else {}]
        
    try:
        transform_id, transform_version, transform_direction, transform_type, transform_solution = (nda.Coregistration & tid_restr & ver_restr & dir_restr & type_restr).fetch1('transform_id','version', 'direction', 'transform_type', 'transform_solution')
    except dj.DataJointError:
        raise ValueError('Specified parameters fail to restrict to a single transformation')
    
    # generate transformation object
    transform_obj = Transform(json=transform_solution)
    
    if as_dict:
        return {'transform_id':transform_id, 'transform_version':transform_version,
                'transform_direction': transform_direction, 'transform_type': transform_type, 'transform_obj': transform_obj}
    else:
        return transform_id, transform_version, transform_direction, transform_type, transform_obj
    

def coreg_transform(coords, transform_id=None, transform_version=None, transform_direction=None, transform_type=None, transform_obj=None):
    """ Transform provided coordinate according to parameters
    
    :param coordinates: 1D or 2D list or array of coordinates 
        if coordinates are 2P, units should be microns
        if coordates are EM, units should be Neuroglancer voxels of the appropriate version
    :param transform_id: ID of desired transformation
    :param transform_version: "phase2" or "phase3"
    :param transform_direction: 
        "EM2P" --> provided coordinate is EM and output is 2P
        "2PEM" --> provided coordinate is 2P and output is EM
    :param transform_type:
        "linear" --> more nonrigid transform
        "spline" --> more rigid transform
    :param transform_obj: option to provide transform_obj
        If transform_obj is None, then it will be fetched using fetch_coreg function and provided transform paramaters
    """ 
    if transform_version is not None:
        assert transform_version=='phase2' or transform_version=='phase3', "transform_version must be 'phase2' or 'phase3'"
    if transform_direction is not None:
        assert transform_direction=="EM2P" or transform_direction=="2PEM", "transform_direction must be 'EM2P' or '2PEM"
    if transform_type is not None:
        assert transform_type=='linear' or transform_type=='spline', "transform_version must be 'linear' or 'spline'"
    
    # format coord
    coords_xyz = format_coords(coords, return_dim=2)
    
    # fetch transformation object
    if transform_obj is None:
        transform_id, transform_version, transform_direction, transform_type, transform_obj = fetch_coreg(transform_id=transform_id, transform_version=transform_version, \
                                    transform_direction=transform_direction, transform_type=transform_type, as_dict=False)    
    else:
        if transform_version is None or transform_direction is None:
            raise Exception('Using provided transformation object but still need to specify transform_version and transform_direction')
        # if transform_type:
            # warnings.warn('Because transformation object is provided, ignoring argument transform_type.')

    # perform transformations
    if transform_version == 'phase2':
        if transform_direction == '2PEM':
            return (em_nm_to_voxels(transform_obj.tform(coords_xyz / 1000))).squeeze()
        
        elif transform_direction == 'EM2P':
            return (transform_obj.tform(em_nm_to_voxels(coords_xyz, inverse=True))*1000).squeeze()
        
        else:
            raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
        
    elif transform_version == 'phase3':
        if transform_direction == '2PEM':
            coords_xyz[:, 1] = 1322 - coords_xyz[:, 1] # phase 3 does not invert y so have to manually do it
            return transform_obj.tform(coords_xyz/1000).squeeze()
        
        elif transform_direction == 'EM2P':
            new_coords = transform_obj.tform(coords_xyz) * 1000
            new_coords[:, 1] = 1322 - new_coords[:, 1] # phase 3 does not invert y so have to manually do it
            return new_coords.squeeze()
        
        else:
            raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
    else:
        raise Exception('Provide transform_version ("phase2" or "phase3")')


def reshape_masks(mask_pixels, mask_weights, image_height, image_width):
    """ Reshape masks into an image_height x image_width x num_masks array."""
    masks = np.zeros(
        [image_height, image_width, len(mask_pixels)], dtype=np.float32
    )

    # Reshape each mask
    for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
        mask_as_vector = np.zeros(image_height * image_width)
        mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
        masks[:, :, i] = mask_as_vector.reshape(
            image_height, image_width, order="F"
        )

    return masks

def get_all_masks(field_key, mask_type=None, plot=False):
    """Returns an image_height x image_width x num_masks matrix with all masks and plots the masks (optional).
    Args:
        field_key      (dict):        dictionary to uniquely identify a field (must contain the keys: "session", "scan_idx", "field")
        mask_type      (str):         options: "soma" or "artifact". Specifies whether to restrict masks by classification. 
                                        soma: restricts to masks classified as soma
                                        artifact: restricts masks classified as artifacts
        plot           (bool):        specify whether to plot masks
        
    Returns:
        masks           (array):      array containing masks of dimensions image_height x image_width x num_masks  
        
        if plot=True:
            matplotlib image    (array):        array of oracle responses interpolated to scan frequency: 10 repeats x 6 oracle clips x f response frames
    """
    mask_rel = nda.Segmentation * nda.MaskClassification & field_key & [{'mask_type': mask_type} if mask_type is not None else {}]

    # Get masks
    image_height, image_width = (nda.Field & field_key).fetch1(
        "px_height", "px_width"
    )
    mask_pixels, mask_weights = mask_rel.fetch(
        "pixels", "weights", order_by="mask_id"
    )

    # Reshape masks
    masks = reshape_masks(
        mask_pixels, mask_weights, image_height, image_width
    )

    if plot:
        corr, avg = (nda.SummaryImages & field_key).fetch1('correlation', 'average')
        image_height, image_width, num_masks = masks.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        plt.imshow(corr*avg)

        cumsum_mask = np.empty([image_height, image_width])
        for i in range(num_masks):
            mask = masks[:, :, i]

            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(
                np.flip(np.argsort(mask, axis=None), axis=0), mask.shape
            )  # max to min value in mask
            cumsum_mask[indices] = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)

            ## Plot contour at desired threshold (with random color)
            random_color = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.contour(cumsum_mask, [0.97], linewidths=0.8, colors=[random_color])

    return masks


def fetch_oracle_raster(unit_key):
    """Fetches the responses of the provided unit to the oracle trials
    Args:
        unit_key      (dict):        dictionary to uniquely identify a functional unit (must contain the keys: "session", "scan_idx", "unit_id") 
        
    Returns:
        oracle_score (float):        
        responses    (array):        array of oracle responses interpolated to scan frequency: 10 repeats x 6 oracle clips x f response frames
    """
    fps = (nda.Scan & unit_key).fetch1('fps') # get frame rate of scan

    oracle_rel = (dj.U('condition_hash').aggr(nda.Trial & unit_key,n='count(*)',m='min(trial_idx)') & 'n=10') # get oracle clips
    oracle_hashes = oracle_rel.fetch('KEY',order_by='m ASC') # get oracle clip hashes sorted temporally

    frame_times_set = []
    # iterate over oracle repeats (10 repeats)
    for first_clip in (nda.Trial & oracle_hashes[0] & unit_key).fetch('trial_idx'): 
        trial_block_rel = (nda.Trial & unit_key & f'trial_idx >= {first_clip} and trial_idx < {first_clip+6}') # uses the trial_idx of the first clip to grab subsequent 5 clips (trial_block) 
        start_times, end_times = trial_block_rel.fetch('start_frame_time', 'end_frame_time', order_by='condition_hash DESC') # grabs start time and end time of each clip in trial_block and orders by condition_hash to maintain order across scans
        frame_times = [np.linspace(s, e , np.round(fps * (e - s)).astype(int)) for s, e in zip(start_times, end_times)] # generate time vector between start and end times according to frame rate of scan
        frame_times_set.append(frame_times)

    trace, fts, delay = ((nda.Activity & unit_key) * nda.ScanTimes * nda.ScanUnit).fetch1('trace', 'frame_times', 'ms_delay') # fetch trace delay and frame times for interpolation
    f2a = interp1d(fts + delay/1000, trace) # create trace interpolator with unit specific time delay
    oracle_traces = np.array([f2a(ft) for ft in frame_times_set]) # interpolate oracle times to match the activity trace
    oracle_traces -= np.min(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
    oracle_traces /= np.max(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
    oracle_score = (nda.Oracle & unit_key).fetch1('pearson') # fetch oracle score
    return oracle_traces, oracle_score

# def em_nm_to_voxels_phase3(xyz, x_offset=31000, y_offset=500, z_offset=3150, inverse=False):
#     """convert EM nanometers to neuroglancer voxels
#     Parameters
#     ----------
#     xyz : :class:`numpy.ndarray`
#         N x 3, the inut array in nm
#     inverse : bool
#         go from voxels to nm
#     Returns
#     -------
#     vxyz : :class:`numpy.ndarray`
#         N x 3, the output array in voxels
#     """
#     if inverse: 
#         vxyz = np.zeros_like(xyz).astype(float)
#         vxyz[:, 0] = (xyz[:, 0] - x_offset) * 4.0
#         vxyz[:, 1] = (xyz[:, 1] - y_offset) * 4.0
#         vxyz[:, 2] = (xyz[:, 2] + z_offset) * 40.0
        
#     else: 
#         vxyz = np.zeros_like(xyz).astype(float)
#         vxyz[:, 0] = ((xyz[:, 0] / 4) + x_offset)
#         vxyz[:, 1] = ((xyz[:, 1] / 4) + y_offset)
#         vxyz[:, 2] = ((xyz[:, 2]/40.0) - z_offset)

#     return vxyz