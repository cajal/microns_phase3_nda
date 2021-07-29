from . import utils
from . import nda

import numpy as np
import torch

import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import iv
from scipy import stats

import datajoint as dj

def em_nm_to_voxels_phase3(xyz, x_offset=31000, y_offset=500, z_offset=3150, inverse=False):
    """convert EM nanometers to neuroglancer voxels
    Parameters
    ----------
    xyz : :class:`numpy.ndarray`
        N x 3, the inut array in nm
    inverse : bool
        go from voxels to nm
    Returns
    -------
    vxyz : :class:`numpy.ndarray`
        N x 3, the output array in voxels
    """
    if inverse: 
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = (xyz[:, 0] - x_offset) * 4.0
        vxyz[:, 1] = (xyz[:, 1] - y_offset) * 4.0
        vxyz[:, 2] = (xyz[:, 2] + z_offset) * 40.0
        
    else: 
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = ((xyz[:, 0] / 4) + x_offset)
        vxyz[:, 1] = ((xyz[:, 1] / 4) + y_offset)
        vxyz[:, 2] = ((xyz[:, 2]/40.0) - z_offset)

    return vxyz


def get_grid(field_key, desired_res=1):
    """ Get registered grid for this registration. """

    # Get field
    field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')

    # Create grid at desired resolution
    grid = utils.create_grid(field_dims, desired_res=desired_res)  # h x w x 2
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

    return utils.affine_product(grid, linear, translation).numpy()


def fetch_coreg(transform_id=None, transform_version=None, transform_direction=None, transform_type=None, as_dict=True):
    if transform_version is not None:
        assert np.logical_or(transform_version=='phase2', transform_version=='phase3'), "transform_version must be 'phase2' or 'phase3'"
    if transform_direction is not None:
        assert np.logical_or(transform_direction=="EM2P", transform_direction=="2PEM"), "transform_direction must be 'EM2P' or '2PEM"
    if transform_type is not None:
        assert np.logical_or(transform_type=='linear', transform_type=='spline'), "transform_version must be 'linear' or 'spline'"

    
    # fetch desired transform solution
    tid_restr = [{'transform_id': transform_id} if transform_id is not None else {}]
    ver_restr = [{'version':transform_version} if transform_version is not None else {}]
    dir_restr = [{'direction': transform_direction} if transform_direction is not None else {}]
    type_restr = [{'transform_type': transform_type} if transform_type is not None else {}]
        
    try:
        transform_id, transform_version, transform_direction, transform_type, transform_solution = (nda.Coregistration & tid_restr & ver_restr & dir_restr & type_restr).fetch1('transform_id','version', 'direction', 'transform_type', 'transform_solution')
    except:
        raise Exception('Specified parameters fail to restrict to a single transformation')
    
    # generate transformation object
    transform_obj = Transform(json=transform_solution)
    
    if as_dict:
        return {'transform_id':transform_id, 'transform_version':transform_version, \
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
        assert np.logical_or(transform_version=='phase2', transform_version=='phase3'), "transform_version must be 'phase2' or 'phase3'"
    if transform_direction is not None:
        assert np.logical_or(transform_direction=="EM2P", transform_direction=="2PEM"), "transform_direction must be 'EM2P' or '2PEM"
    if transform_type is not None:
        assert np.logical_or(transform_type=='linear', transform_type=='spline'), "transform_version must be 'linear' or 'spline'"
    
    # format coord
    coords_xyz = utils.format_coords(coords, return_dim=2)
    
    # fetch transformation object
    if transform_obj is None:
        transform_id, transform_version, transform_direction, transform_type, transform_obj = fetch_coreg(transform_id=transform_id, transform_version=transform_version, \
                                    transform_direction=transform_direction, transform_type=transform_type, as_dict=False)    
    else:
        if np.logical_or(transform_version==None, transform_direction==None):
            raise Exception('Using provided transformation object but still need to specify transform_version and transform_direction')
        # if transform_type:
            # warnings.warn('Because transformation object is provided, ignoring argument transform_type.')

    # perform transformations
    if transform_version == 'phase2':
        if transform_direction == '2PEM':
            return (em_nm_to_voxels(transform_obj.tform(coords_xyz/ 1000))).squeeze()
        
        elif transform_direction == 'EM2P':
            return (transform_obj.tform(em_nm_to_voxels(coords_xyz, inverse=True))*1000).squeeze()
        
        else:
            raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
        
    elif transform_version == 'phase3':
        if transform_direction == '2PEM':
            coords_xyz[:,1] = 1322 - coords_xyz[:,1] # phase 3 does not invert y so have to manually do it
            return transform_obj.tform(coords_xyz/1000).squeeze()
        
        elif transform_direction == 'EM2P':
            new_coords = transform_obj.tform(coords_xyz)*1000
            new_coords[:,1] = 1322 - new_coords[:,1] # phase 3 does not invert y so have to manually do it
            return new_coords.squeeze()
        
        else:
            raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
    else:
        raise Exception('Provide transform_version ("phase2" or "phase3")')


def field_to_EM_grid(field_key, transform_id=None, transform_version=None, transform_direction="2PEM", transform_type=None, transform_obj=None):
    assert transform_direction=='2PEM', "transform_direction must be '2PEM'"

    grid = get_grid(field_key, desired_res=1)
    # convert grid from motor coordinates to numpy coordinates
    center_x, center_y, center_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
    length_x, length_y, length_z = nda.Stack.fetch1('um_width', 'um_height', 'um_depth')
    np_grid = grid - np.array([center_x, center_y, center_z]) + np.array([length_x, length_y, length_z]) / 2
    transformed_coordinates = coreg_transform(utils.coordinate(np_grid), transform_id=transform_id, transform_direction=transform_direction, transform_version=transform_version, transform_type=transform_type, transform_obj=transform_obj)
    return utils.uncoordinate(transformed_coordinates,*np_grid.shape[:2])  


def get_field_image(field_key, summary_image='average*correlation', enhance=True, desired_res=1):
    """ Get resized field image specified by field key. Option to perform operations on images and enhance. 
    
    :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
    :param summary_image: the type of summary image to fetch from nda.SummaryImages. operations on images can be specified. 
        The default is average image multiplied by correlation image.
    :param enhance: If enhance: the image will undergo local contrast normalization and sharpening
    :param desired_res: The desired resolution of output image
    """
    # fetch images and field dimensions   
    correlation, average, l6norm = (nda.SummaryImages & field_key).fetch1('correlation', 'average', 'l6norm')
    field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')    
    
    # perform operation on images
    image = eval(summary_image)
    
    if enhance:
        image = utils.lcn(image, 2.5)
        image = utils.sharpen_2pimage(image)
    
    return utils.resize(image, field_dims, desired_res=desired_res)


def get_stack_field_image(field_key, stack, desired_res=1):
    """ Sample stack with field grid 
    
    :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
    :param stack: the stack to be sampled. 
    :param desired_res: The desired resolution of output image
    """
    
    stack_x, stack_y, stack_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
    grid = get_grid(field_key, desired_res=1)
    grid = grid - np.array([stack_x, stack_y, stack_z])
    return utils.sample_grid(stack, grid).numpy()


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

def get_all_masks(field_key):
    """Returns an image_height x image_width x num_masks matrix with all masks."""
    mask_rel = nda.Segmentation & field_key

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

    trace, fts, delay = ((nda.Activity & unit_key) * nda.FrameTimes * nda.ScanUnit).fetch1('trace', 'frame_times', 'ms_delay') # fetch trace delay and frame times for interpolation
    f2a = interp1d(fts + delay/1000, trace) # create trace interpolator with unit specific time delay
    oracle_traces = np.array([f2a(ft) for ft in frame_times_set]) # interpolate oracle times to match the activity trace
    oracle_traces -= np.min(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
    oracle_traces /= np.max(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
    oracle_score = (nda.Oracle & unit_key).fetch1('pearson') # fetch oracle score
    return oracle_traces, oracle_score