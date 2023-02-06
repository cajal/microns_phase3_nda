from . import nda

import numpy as np
import torch

from torch.nn import functional as F
from scipy import ndimage
import re
import urllib
import json


import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import iv
from scipy import stats, signal
from scipy.ndimage import convolve1d
import cv2
import tqdm as tqdm
from pipeline import experiment, stack, meso, fuse
from pipeline.utils import performance
import scanreader
from stimline import tune
# import mpy 
# import io 
# import imageio 

import datajoint as dj
import matplotlib.pyplot as plt

from . import nda


# def clock2math(t): 
#     """shifts convention from 0 degrees at 12 o'clock to 0 degrees at 3 o'clock""" 
#     return ((np.pi / 2) - t) % (2 * np.pi)


# def create_grid(um_sizes, desired_res=1):
#     """ Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
#      um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).
#     In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
#     centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
#     are NOT at -2 and 2 which is the assumption in some libraries.
#     :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
#     :param float or tuple desired_res: Desired resolution (um/px) for the grid.
#     :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
#     each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
#     coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
#     p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
#     """
#     # Make sure desired_res is a tuple with the same size as um_sizes
#     if np.isscalar(desired_res):
#         desired_res = (desired_res,) * len(um_sizes)

#     # Create grid
#     out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
#     um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.double)
#                 for s, res in zip(out_sizes, desired_res)] # *
#     full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
#     # * this preserves the desired resolution by slightly changing the size of the FOV to
#     # out_sizes rather than um_sizes / desired_res.

#     return full_grid


# def affine_product(X, A, b):
#     """ Special case of affine transformation that receives coordinates X in 2-d (x, y)
#     and affine matrix A and translation vector b in 3-d (x, y, z). Y = AX + b
#     :param torch.Tensor X: A matrix of 2-d coordinates (d1 x d2 x 2).
#     :param torch.Tensor A: The first two columns of the affine matrix (3 x 2).
#     :param torch.Tensor b: A 3-d translation vector.
#     :return: A (d1 x d2 x 3) torch.Tensor corresponding to the transformed coordinates.
#     """
#     return torch.einsum('ij,klj->kli', (A, X)) + b


# def sample_grid(volume, grid):
#     """ 
#     Sample grid in volume.

#     Assumes center of volume is at (0, 0, 0) and grid and volume have the same resolution.

#     :param torch.Tensor volume: A d x h x w tensor. The stack.
#     :param torch.Tensor grid: A d1 x d2 x 3 (x, y, z) tensor. The coordinates to sample.

#     :return: A d1 x d2 tensor. The grid sampled in the stack.
#     """
#     # Make sure input is tensor
#     volume = torch.as_tensor(volume, dtype=torch.double)
#     grid = torch.as_tensor(grid, dtype=torch.double)

#     # Rescale grid so it ranges from -1 to 1 (as expected by F.grid_sample)
#     norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
#     norm_grid = grid / norm_factor

#     # Resample
#     resampled = F.grid_sample(volume[None, None, ...], norm_grid[None, None, ...], padding_mode='zeros', align_corners=True)
#     resampled = resampled.squeeze() # drop batch and channel dimension

#     return resampled


# def format_coords(coords_xyz, return_dim=1):
#     # format coordinates 
#     coords_xyz = np.array(coords_xyz)
    
#     assert np.logical_or(return_dim==1, return_dim==2), '"ndim" must be 1 or 2'
#     assert np.logical_or(coords_xyz.ndim == 1, coords_xyz.ndim == 2), 'Coordinate(s) must be 1D or 2D'
#     assert coords_xyz.shape[-1] == 3, 'Coordinate(s) must have exactly x, y, and z'
#     assert np.logical_or(coords_xyz.dtype==np.int, coords_xyz.dtype==np.float), 'Datatype must be int or float'
    
#     coords_xyz = coords_xyz if coords_xyz.ndim == return_dim else np.expand_dims(coords_xyz, 0)
        
#     return coords_xyz


# def coordinate(grid_to_transform):
#     x = grid_to_transform.shape[0]
#     y = grid_to_transform.shape[1]
#     return grid_to_transform.reshape(x*y,-1)


# def uncoordinate(transformed_coordinates,x,y):
#     return transformed_coordinates.reshape(x,y,-1)


# def lcn(image, sigmas=(12, 12)):
#     """ Local contrast normalization.
#     Normalize each pixel using mean and stddev computed on a local neighborhood.
#     We use gaussian filters rather than uniform filters to compute the local mean and std
#     to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
#     Equivalent using a hard defintion of neighborhood will be:
#         local_mean = ndimage.uniform_filter(image, size=(32, 32))
#     :param np.array image: Array with raw two-photon images.
#     :param tuple sigmas: List with sigmas (one per axis) to use for the gaussian filter.
#         Smaller values result in more local neighborhoods. 15-30 microns should work fine
#     """
#     local_mean = ndimage.gaussian_filter(image, sigmas)
#     local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
#     local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
#     norm = (image - local_mean) / (local_std + 1e-7)

#     return norm


# def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
#     """ Apply a laplacian filter, clip pixel range and normalize.
#     :param np.array image: Array with raw two-photon images.
#     :param float laplace_sigma: Sigma of the gaussian used in the laplace filter.
#     :param float low_percentile, high_percentile: Percentiles at which to clip.
#     :returns: Array of same shape as input. Sharpened image.
#     """
#     sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
#     clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
#     norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
#     return norm


# def resize(original, um_sizes, desired_res, mode='bilinear'):
#     """ Resize array originally of um_sizes size to have desired_res resolution.
#     We preserve the center of original and resized arrays exactly in the middle. We also
#     make sure resolution is exactly the desired resolution. Given these two constraints,
#     we cannot hold FOV of original and resized arrays to be exactly the same.
#     :param np.array original: Array to resize.
#     :param tuple um_sizes: Size in microns of the array (one per axis).
#     :param int or tuple desired_res: Desired resolution (um/px) for the output array.
#     :return: Output array (np.float32) resampled to the desired resolution. Size in pixels
#         is round(um_sizes / desired_res).
#     """

#     # Create grid to sample in microns
#     grid = create_grid(um_sizes, desired_res) # d x h x w x 3

#     # Re-express as a torch grid [-1, 1]
#     um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
#     torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original
#     grid = grid / torch_ones[::-1].astype(np.double)

#     # Resample
#     input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(
#         np.double))
#     grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
#     resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border', mode=mode, align_corners=True)
#     resized = resized_tensor.numpy().squeeze()

#     return resized


# def html_to_json(url_string, return_parsed_url=False, fragment_prefix='!'):
#     # Parse neuromancer url to logically separate the json state dict from the rest of it.
#     full_url_parsed = urllib.parse.urlparse(url_string)
#     # Decode percent-encoding in url, and skip "!" from beginning of string.
#     decoded_fragment = urllib.parse.unquote(full_url_parsed.fragment)
#     if decoded_fragment.startswith(fragment_prefix):
#         decoded_fragment = decoded_fragment[1:]
#     # Load the json state dict string into a python dictionary.
#     json_state_dict = json.loads(decoded_fragment)

#     if return_parsed_url:
#         return json_state_dict, full_url_parsed
#     else:
#         return json_state_dict


# def json_to_url(json_dict, fragment_prefix='https://neuromancer-seung-import.appspot.com/#!'):
#     loaded_state = json.loads(json_dict)
#     return f"{fragment_prefix}{json.dumps(loaded_state)}"


# def add_point_annotations(provided_link, ano_name, centroids, voxelsize=[4,4,40], descriptions=None, color='#f1ff00', overwrite=True):
#     # format annotation list
#     if centroids.ndim<2:
#         centroids = np.expand_dims(centroids,0)
#     if centroids.ndim>2:
#         print('The annotation list must be 1D or 2D')
#         return
    
#     ano_list_dict = []
#     if descriptions is not None:
#         for i, (cent, desc) in enumerate(zip(centroids.tolist(), descriptions)):
#             ano_list_dict.append({'point':cent, 'type':'point', 'id':str(i+1), 'description':f'{desc}', 'tagIds':[]})
# #         print(ano_list_dict)
#     else: 
#         for i, cent in enumerate(centroids.tolist()):
#                 ano_list_dict.append({'point':cent, 'type':'point', 'id':str(i+1), 'tagIds':[]})
# #         print('No descriptions added')

#     json_data, parsed_url = html_to_json(provided_link, return_parsed_url=True)
    
#     # extract layer and index if already present
#     index = [i for i, L in enumerate(json_data['layers']) if L['name']==ano_name]
    
#     # ensure layer not duplicated
#     if len(index) > 1:
#         print('Layer duplicated, please remove duplicates')
#         return
    
#     # if annotation layer doesn't exist, create it
#     elif len(index) == 0:
#         json_data['layers'].append({'tool': 'annotatePoint',
#                                'type': 'annotation',
#                                'annotations': [],
#                                'annotationColor': color,
#                                'annotationTags': [],
#                                'voxelSize': voxelsize,
#                                'name': ano_name})
#         index = [i for i, L in enumerate(json_data['layers']) if L['name']==ano_name][0]
# #         print(index)
# #         print('annotation layer does not exist... creating it')
    
#     else:
#         index = index[0]
# #         print('annotation layer exists')
    
#     # test if voxel size of annotation matches provided voxel size
#     if json_data['layers'][index]['voxelSize']!=voxelsize:
#         print('The annotation layer already exists but does not match your provided voxelsize')
#         return
    
#     # add annotations
#     if overwrite:
#         json_data['layers'][index]['annotations'] = ano_list_dict
#     else:
#         json_data['layers'][index]['annotations'].extend(ano_list_dict)

#     return urllib.parse.urlunparse([parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, '!'+ urllib.parse.quote(json.dumps(json_data))])























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
    coords_xyz = format_coords(coords, return_dim=2)
    
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


# def field_to_EM_grid(field_key, transform_id=None, transform_version=None, transform_direction="2PEM", transform_type=None, transform_obj=None):
#     assert transform_direction=='2PEM', "transform_direction must be '2PEM'"

#     grid = get_grid(field_key, desired_res=1)
#     # convert grid from motor coordinates to numpy coordinates
#     center_x, center_y, center_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
#     length_x, length_y, length_z = nda.Stack.fetch1('um_width', 'um_height', 'um_depth')
#     np_grid = grid - np.array([center_x, center_y, center_z]) + np.array([length_x, length_y, length_z]) / 2
#     transformed_coordinates = coreg_transform(coordinate(np_grid), transform_id=transform_id, transform_direction=transform_direction, transform_version=transform_version, transform_type=transform_type, transform_obj=transform_obj)
#     return uncoordinate(transformed_coordinates,*np_grid.shape[:2])  


# def get_field_image(field_key, summary_image='average*correlation', enhance=True, desired_res=1):
#     """ Get resized field image specified by field key. Option to perform operations on images and enhance. 
    
#     :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
#     :param summary_image: the type of summary image to fetch from nda.SummaryImages. operations on images can be specified. 
#         The default is average image multiplied by correlation image.
#     :param enhance: If enhance: the image will undergo local contrast normalization and sharpening
#     :param desired_res: The desired resolution of output image
#     """
#     # fetch images and field dimensions   
#     correlation, average, l6norm = (nda.SummaryImages & field_key).fetch1('correlation', 'average', 'l6norm')
#     field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')    
    
#     # perform operation on images
#     image = eval(summary_image)
    
#     if enhance:
#         image = lcn(image, 2.5)
#         image = sharpen_2pimage(image)
    
#     return resize(image, field_dims, desired_res=desired_res)


# def get_stack_field_image(field_key, stack, desired_res=1):
#     """ Sample stack with field grid 
    
#     :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
#     :param stack: the stack to be sampled. 
#     :param desired_res: The desired resolution of output image
#     """
    
#     stack_x, stack_y, stack_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
#     grid = get_grid(field_key, desired_res=1)
#     grid = grid - np.array([stack_x, stack_y, stack_z])
#     return sample_grid(stack, grid).numpy()


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

# def slice_array(array, axis, idx=None, start=None, end=None):
#     """
#     Takes array and retrieves items between start and end indices at axis 
#     Args:
#         array:      numpy array 
#         axis:int    axis to get values from 
#         start:int   start index 
#         end:int     end index
    
    
#     """
#     idx_slice = [slice(None)] * array.ndim
#     if (start is None) or (end is None):
#         idx_slice[axis] = idx
#     else:
#         idx_slice[axis] = slice(start,end)
#     return array[tuple(idx_slice)]


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
    
    assert np.logical_or(return_dim==1, return_dim==2), '"ndim" must be 1 or 2'
    assert np.logical_or(coords_xyz.ndim == 1, coords_xyz.ndim == 2), 'Coordinate(s) must be 1D or 2D'
    assert coords_xyz.shape[-1] == 3, 'Coordinate(s) must have exactly x, y, and z'
    assert np.logical_or(coords_xyz.dtype==np.int, coords_xyz.dtype==np.float), 'Datatype must be int or float'
    
    coords_xyz = coords_xyz if coords_xyz.ndim == return_dim else np.expand_dims(coords_xyz, 0)
        
    return coords_xyz
