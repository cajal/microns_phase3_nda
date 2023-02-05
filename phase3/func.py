# from . import plot_utils
# from . import nda

# import numpy as np
# import torch

# import coregister.solve as cs
# from coregister.transform.transform import Transform
# from coregister.utils import em_nm_to_voxels

# from scipy.interpolate import interp1d
# from scipy.optimize import minimize_scalar
# from scipy.special import iv
# from scipy import stats, signal
# from scipy.ndimage import convolve1d
# from stimulus.stimulus import BehaviorSync,Sync
# import cv2
# from stimulus import stimulus
# import tqdm as tqdm
# from pipeline import experiment, stack, meso, fuse
# from pipeline.utils import performance
# import scanreader
# from stimline import tune
# import mpy 
# import io 
# import imageio 

# import datajoint as dj
# import matplotlib.pyplot as plt

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

# def make_movie(t,interpolated_movie,target_hz):
#     frame = interpolated_movie[:,:,int(round(t*target_hz))]
#     return np.repeat(frame[:, :, np.newaxis], 3, axis=2)

# # def get_grid(field_key, desired_res=1):
# #     """ Get registered grid for this registration. """

# #     # Get field
# #     field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')

# #     # Create grid at desired resolution
# #     grid = plot_utils.create_grid(field_dims, desired_res=desired_res)  # h x w x 2
# #     grid = torch.as_tensor(grid, dtype=torch.double)

# #     # Apply required transform
# #     params = (nda.Registration & field_key).fetch1(
# #         'a11', 'a21', 'a31', 
# #         'a12', 'a22', 'a32', 
# #         'reg_x', 'reg_y', 'reg_z'
# #         )
# #     a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z = params
# #     linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]], dtype=torch.double)
# #     translation = torch.tensor([delta_x, delta_y, delta_z], dtype=torch.double)

# #     return plot_utils.affine_product(grid, linear, translation).numpy()


# # def fetch_coreg(transform_id=None, transform_version=None, transform_direction=None, transform_type=None, as_dict=True):
# #     if transform_version is not None:
# #         assert np.logical_or(transform_version=='phase2', transform_version=='phase3'), "transform_version must be 'phase2' or 'phase3'"
# #     if transform_direction is not None:
# #         assert np.logical_or(transform_direction=="EM2P", transform_direction=="2PEM"), "transform_direction must be 'EM2P' or '2PEM"
# #     if transform_type is not None:
# #         assert np.logical_or(transform_type=='linear', transform_type=='spline'), "transform_version must be 'linear' or 'spline'"

    
# #     # fetch desired transform solution
# #     tid_restr = [{'transform_id': transform_id} if transform_id is not None else {}]
# #     ver_restr = [{'version':transform_version} if transform_version is not None else {}]
# #     dir_restr = [{'direction': transform_direction} if transform_direction is not None else {}]
# #     type_restr = [{'transform_type': transform_type} if transform_type is not None else {}]
        
# #     try:
# #         transform_id, transform_version, transform_direction, transform_type, transform_solution = (nda.Coregistration & tid_restr & ver_restr & dir_restr & type_restr).fetch1('transform_id','version', 'direction', 'transform_type', 'transform_solution')
# #     except:
# #         raise Exception('Specified parameters fail to restrict to a single transformation')
    
# #     # generate transformation object
# #     transform_obj = Transform(json=transform_solution)
    
# #     if as_dict:
# #         return {'transform_id':transform_id, 'transform_version':transform_version, \
# #                 'transform_direction': transform_direction, 'transform_type': transform_type, 'transform_obj': transform_obj}
# #     else:
# #         return transform_id, transform_version, transform_direction, transform_type, transform_obj
    

# # def coreg_transform(coords, transform_id=None, transform_version=None, transform_direction=None, transform_type=None, transform_obj=None):
# #     """ Transform provided coordinate according to parameters
    
# #     :param coordinates: 1D or 2D list or array of coordinates 
# #         if coordinates are 2P, units should be microns
# #         if coordates are EM, units should be Neuroglancer voxels of the appropriate version
# #     :param transform_id: ID of desired transformation
# #     :param transform_version: "phase2" or "phase3"
# #     :param transform_direction: 
# #         "EM2P" --> provided coordinate is EM and output is 2P
# #         "2PEM" --> provided coordinate is 2P and output is EM
# #     :param transform_type:
# #         "linear" --> more nonrigid transform
# #         "spline" --> more rigid transform
# #     :param transform_obj: option to provide transform_obj
# #         If transform_obj is None, then it will be fetched using fetch_coreg function and provided transform paramaters
# #     """ 
# #     if transform_version is not None:
# #         assert np.logical_or(transform_version=='phase2', transform_version=='phase3'), "transform_version must be 'phase2' or 'phase3'"
# #     if transform_direction is not None:
# #         assert np.logical_or(transform_direction=="EM2P", transform_direction=="2PEM"), "transform_direction must be 'EM2P' or '2PEM"
# #     if transform_type is not None:
# #         assert np.logical_or(transform_type=='linear', transform_type=='spline'), "transform_version must be 'linear' or 'spline'"
    
# #     # format coord
# #     coords_xyz = plot_utils.format_coords(coords, return_dim=2)
    
# #     # fetch transformation object
# #     if transform_obj is None:
# #         transform_id, transform_version, transform_direction, transform_type, transform_obj = fetch_coreg(transform_id=transform_id, transform_version=transform_version, \
# #                                     transform_direction=transform_direction, transform_type=transform_type, as_dict=False)    
# #     else:
# #         if np.logical_or(transform_version==None, transform_direction==None):
# #             raise Exception('Using provided transformation object but still need to specify transform_version and transform_direction')
# #         # if transform_type:
# #             # warnings.warn('Because transformation object is provided, ignoring argument transform_type.')

# #     # perform transformations
# #     if transform_version == 'phase2':
# #         if transform_direction == '2PEM':
# #             return (em_nm_to_voxels(transform_obj.tform(coords_xyz/ 1000))).squeeze()
        
# #         elif transform_direction == 'EM2P':
# #             return (transform_obj.tform(em_nm_to_voxels(coords_xyz, inverse=True))*1000).squeeze()
        
# #         else:
# #             raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
        
# #     elif transform_version == 'phase3':
# #         if transform_direction == '2PEM':
# #             coords_xyz[:,1] = 1322 - coords_xyz[:,1] # phase 3 does not invert y so have to manually do it
# #             return transform_obj.tform(coords_xyz/1000).squeeze()
        
# #         elif transform_direction == 'EM2P':
# #             new_coords = transform_obj.tform(coords_xyz)*1000
# #             new_coords[:,1] = 1322 - new_coords[:,1] # phase 3 does not invert y so have to manually do it
# #             return new_coords.squeeze()
        
# #         else:
# #             raise Exception('Provide transformation direction ("2PEM" or "EM2P")')
# #     else:
# #         raise Exception('Provide transform_version ("phase2" or "phase3")')


# # def field_to_EM_grid(field_key, transform_id=None, transform_version=None, transform_direction="2PEM", transform_type=None, transform_obj=None):
# #     assert transform_direction=='2PEM', "transform_direction must be '2PEM'"

# #     grid = get_grid(field_key, desired_res=1)
# #     # convert grid from motor coordinates to numpy coordinates
# #     center_x, center_y, center_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
# #     length_x, length_y, length_z = nda.Stack.fetch1('um_width', 'um_height', 'um_depth')
# #     np_grid = grid - np.array([center_x, center_y, center_z]) + np.array([length_x, length_y, length_z]) / 2
# #     transformed_coordinates = coreg_transform(utils.coordinate(np_grid), transform_id=transform_id, transform_direction=transform_direction, transform_version=transform_version, transform_type=transform_type, transform_obj=transform_obj)
# #     return utils.uncoordinate(transformed_coordinates,*np_grid.shape[:2])  


# # def get_field_image(field_key, summary_image='average*correlation', enhance=True, desired_res=1):
# #     """ Get resized field image specified by field key. Option to perform operations on images and enhance. 
    
# #     :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
# #     :param summary_image: the type of summary image to fetch from nda.SummaryImages. operations on images can be specified. 
# #         The default is average image multiplied by correlation image.
# #     :param enhance: If enhance: the image will undergo local contrast normalization and sharpening
# #     :param desired_res: The desired resolution of output image
# #     """
# #     # fetch images and field dimensions   
# #     correlation, average, l6norm = (nda.SummaryImages & field_key).fetch1('correlation', 'average', 'l6norm')
# #     field_dims = (nda.Field & field_key).fetch1('um_height', 'um_width')    
    
# #     # perform operation on images
# #     image = eval(summary_image)
    
# #     if enhance:
# #         image = utils.lcn(image, 2.5)
# #         image = utils.sharpen_2pimage(image)
    
# #     return utils.resize(image, field_dims, desired_res=desired_res)


# # def get_stack_field_image(field_key, stack, desired_res=1):
# #     """ Sample stack with field grid 
    
# #     :param field_key: a dictionary that can restrict into nda.Field to uniquely identify a single field
# #     :param stack: the stack to be sampled. 
# #     :param desired_res: The desired resolution of output image
# #     """
    
# #     stack_x, stack_y, stack_z = nda.Stack.fetch1('motor_x', 'motor_y', 'motor_z')
# #     grid = get_grid(field_key, desired_res=1)
# #     grid = grid - np.array([stack_x, stack_y, stack_z])
# #     return utils.sample_grid(stack, grid).numpy()


# def reshape_masks(mask_pixels, mask_weights, image_height, image_width):
#     """ Reshape masks into an image_height x image_width x num_masks array."""
#     masks = np.zeros(
#         [image_height, image_width, len(mask_pixels)], dtype=np.float32
#     )

#     # Reshape each mask
#     for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
#         mask_as_vector = np.zeros(image_height * image_width)
#         mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
#         masks[:, :, i] = mask_as_vector.reshape(
#             image_height, image_width, order="F"
#         )

#     return masks

# # def get_all_masks(field_key, mask_type=None, plot=False):
# #     """Returns an image_height x image_width x num_masks matrix with all masks and plots the masks (optional).
# #     Args:
# #         field_key      (dict):        dictionary to uniquely identify a field (must contain the keys: "session", "scan_idx", "field")
# #         mask_type      (str):         options: "soma" or "artifact". Specifies whether to restrict masks by classification. 
# #                                         soma: restricts to masks classified as soma
# #                                         artifact: restricts masks classified as artifacts
# #         plot           (bool):        specify whether to plot masks
        
# #     Returns:
# #         masks           (array):      array containing masks of dimensions image_height x image_width x num_masks  
        
# #         if plot=True:
# #             matplotlib image    (array):        array of oracle responses interpolated to scan frequency: 10 repeats x 6 oracle clips x f response frames
# #     """
# #     mask_rel = nda.Segmentation * nda.MaskClassification & field_key & [{'mask_type': mask_type} if mask_type is not None else {}]

# #     # Get masks
# #     image_height, image_width = (nda.Field & field_key).fetch1(
# #         "px_height", "px_width"
# #     )
# #     mask_pixels, mask_weights = mask_rel.fetch(
# #         "pixels", "weights", order_by="mask_id"
# #     )

# #     # Reshape masks
# #     masks = reshape_masks(
# #         mask_pixels, mask_weights, image_height, image_width
# #     )

# #     if plot:
# #         corr, avg = (nda.SummaryImages & field_key).fetch1('correlation', 'average')
# #         image_height, image_width, num_masks = masks.shape
# #         figsize = np.array([image_width, image_height]) / min(image_height, image_width)
# #         fig = plt.figure(figsize=figsize * 7)
# #         plt.imshow(corr*avg)

# #         cumsum_mask = np.empty([image_height, image_width])
# #         for i in range(num_masks):
# #             mask = masks[:, :, i]

# #             ## Compute cumulative mass (similar to caiman)
# #             indices = np.unravel_index(
# #                 np.flip(np.argsort(mask, axis=None), axis=0), mask.shape
# #             )  # max to min value in mask
# #             cumsum_mask[indices] = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)

# #             ## Plot contour at desired threshold (with random color)
# #             random_color = (np.random.rand(), np.random.rand(), np.random.rand())
# #             plt.contour(cumsum_mask, [0.97], linewidths=0.8, colors=[random_color])

# #     return masks



# # def fetch_oracle_raster(unit_key):
# #     """Fetches the responses of the provided unit to the oracle trials
# #     Args:
# #         unit_key      (dict):        dictionary to uniquely identify a functional unit (must contain the keys: "session", "scan_idx", "unit_id") 
        
# #     Returns:
# #         oracle_score (float):        
# #         responses    (array):        array of oracle responses interpolated to scan frequency: 10 repeats x 6 oracle clips x f response frames
# #     """
# #     fps = (nda.Scan & unit_key).fetch1('fps') # get frame rate of scan

# #     oracle_rel = (dj.U('condition_hash').aggr(nda.Trial & unit_key,n='count(*)',m='min(trial_idx)') & 'n=10') # get oracle clips
# #     oracle_hashes = oracle_rel.fetch('KEY',order_by='m ASC') # get oracle clip hashes sorted temporally

# #     frame_times_set = []
# #     # iterate over oracle repeats (10 repeats)
# #     for first_clip in (nda.Trial & oracle_hashes[0] & unit_key).fetch('trial_idx'): 
# #         trial_block_rel = (nda.Trial & unit_key & f'trial_idx >= {first_clip} and trial_idx < {first_clip+6}') # uses the trial_idx of the first clip to grab subsequent 5 clips (trial_block) 
# #         start_times, end_times = trial_block_rel.fetch('start_frame_time', 'end_frame_time', order_by='condition_hash DESC') # grabs start time and end time of each clip in trial_block and orders by condition_hash to maintain order across scans
# #         frame_times = [np.linspace(s, e , np.round(fps * (e - s)).astype(int)) for s, e in zip(start_times, end_times)] # generate time vector between start and end times according to frame rate of scan
# #         frame_times_set.append(frame_times)

# #     trace, fts, delay = ((nda.Activity & unit_key) * nda.ScanTimes * nda.ScanUnit).fetch1('trace', 'frame_times', 'ms_delay') # fetch trace delay and frame times for interpolation
# #     f2a = interp1d(fts + delay/1000, trace) # create trace interpolator with unit specific time delay
# #     oracle_traces = np.array([f2a(ft) for ft in frame_times_set]) # interpolate oracle times to match the activity trace
# #     oracle_traces -= np.min(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
# #     oracle_traces /= np.max(oracle_traces,axis=(1,2),keepdims=True) # normalize the oracle traces
# #     oracle_score = (nda.Oracle & unit_key).fetch1('pearson') # fetch oracle score
# #     return oracle_traces, oracle_score

# def get_timing_offset(key):
#     """
#     Fetches timing offset between behavior and stimulus computers.
#     Args:
#         key: dictionary specifying the scan 
#     Returns:
#         photodiode_zero: float indicating the offset between stimulus and behavior time keeping


    
    
#     """
#     frame_times = (Sync() & key).fetch1('frame_times')
#     frame_times_beh = (BehaviorSync() & key).fetch1('frame_times')
#     photodiode_zero = np.nanmedian(frame_times - frame_times_beh)
#     return photodiode_zero

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


# def resize_movie(movie, target_size, time_axis=2):
#     """
#     Resizes movie to target_size using cv2.resize (inter area interpolation)
    
#     Args:
#         movie:np.array                 movie to resize 
#         target_size:array-like         array-like object representing target size (HxW)
#         time_axis:int                  integer indicated which axis represents time (defaults to 2) 
    
    
#     """
#     new_shape = list(target_size)
#     new_shape.insert(time_axis, movie.shape[time_axis])
#     new_movie = np.zeros(new_shape)
#     # This structure can be used to select slices from a dynamically specified axis
#     # Use by setting idx_slice[dynamic_axis] = idx OR slice(start,end) and then array[tuple(idx_slice)]
#     idx_slice = [slice(None)] * movie.ndim
#     for i in range(movie.shape[time_axis]):
#         idx_slice[time_axis] = i
#         # cv2 uses height x width reversed compared to numpy, target_size is flipped
#         # INTER_AREA interpolation appears best for downsample and upsample
#         # replace with bilinear interpolation
#         new_movie[tuple(idx_slice)] = cv2.resize(movie[tuple(idx_slice)], 
#                                                  target_size[::-1], interpolation=cv2.INTER_AREA)
        
        
#     return new_movie


# def hamming_filter(movie, time_axis, source_times, target_times, filter_size=20):
#     """
#     Hamming filter to change framerate of movie from source_times framerate to target_times framerate at time_axis 
#     Args:
#         movie:np.array              movie to filter 
#         time_axis:int               integer representing time axis of movie 
#         source_times:np.array       numpy array representing timestamps of frames in movie 
#         target_times:np.array       numpy array representing target timestamps of frames in movie 
#         filter_size:int             integer representing size of hamming window
    
#     """
#     source_hz = 1/np.median(np.diff(source_times))
#     target_hz = 1/np.median(np.diff(target_times))
#     scipy_ham = signal.firwin(filter_size, cutoff=target_hz/2, window="hamming", fs=source_hz)
#     filtered_movie = convolve1d(movie, scipy_ham, axis=time_axis)
#     return filtered_movie



# def generate_stimulus_avi(key):
#     """
#     Generates an avi of the stimulus from frames stored in the database 
#     Args:
#         key:dict Scan to generate the avis from 
    
#     Returns:
#         None, writes stimulus AVI locally
    
    
#     """
#     time_axis = 2
#     target_size = (90, 160)
#     full_stimulus = None
#     full_flips = None

#     key['animal_id'] = 17797
#     print(key)
#     scan_times = (stimulus.Sync & key).fetch1('frame_times').squeeze()
#     target_hz = 30
#     trial_data = ((stimulus.Trial & key) * stimulus.Condition).fetch('KEY', 'stimulus_type', order_by='trial_idx ASC')
#     for trial_key,stim_type in zip(tqdm(trial_data[0]), trial_data[1]):
        
#         if stim_type == 'stimulus.Clip':
#             djtable = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
#             flip_times, compressed_clip, skip_time, cut_after,frame_rate = djtable.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
#             flip_times = flip_times[0]
#             # convert to grayscale and stack to movie in width x height x time
#             temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
#             # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
#             temp_vid_length = temp_vid.count_frames()
#             movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)
#             assumed_clip_fps = frame_rate
#             start_idx = int(np.float(skip_time) * assumed_clip_fps)
#             print(trial_key)
#             end_idx = int(start_idx + (np.float(cut_after) * assumed_clip_fps))
#             times = np.linspace(flip_times[0],flip_times[-1], (flip_times[-1] - flip_times[0])*target_hz)
#             movie = movie[:,:,start_idx:end_idx]
#             movie = resize_movie(movie, target_size, time_axis)
#             movie = hamming_filter(movie, time_axis, flip_times, times)
#             full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#             full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
#         elif stim_type == 'stimulus.Monet2':
#             flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('flip_times', 'movie')
#             flip_times = flip_times[0]
#             movie = movie[:,:,0,:]  
#             movie = resize_movie(movie, target_size, time_axis)
#             times = np.linspace(flip_times[0],flip_times[-1],(flip_times[-1] - flip_times[0])*target_hz)
#             movie = hamming_filter(movie, time_axis, flip_times, times)
#             full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#             full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
#         elif stim_type == 'stimulus.Trippy':
#             flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('flip_times', 'movie')
#             flip_times = flip_times[0]
#             movie = resize_movie(movie, target_size, time_axis)
#             times = np.linspace(flip_times[0],flip_times[-1],(flip_times[-1] - flip_times[0])*target_hz)
#             movie = hamming_filter(movie, time_axis, flip_times, times)
#             full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#             full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
#         else:
#             raise Exception(f'Error: stimulus type {stim_type} not understood')

#     pre_blank_length = full_flips[0] - scan_times[0]
#     post_blank_length = scan_times[-1] - full_flips[-1]
#     pre_nframes = np.ceil(pre_blank_length*target_hz)
#     h,w,t = full_stimulus.shape
#     times = np.linspace(full_flips[0],full_flips[-1],int(np.ceil((full_flips[-1] - full_flips[0])*target_hz)))

#     interpolated_movie = np.zeros((h, w, int(np.ceil(scan_times[-1] - scan_times[0])*target_hz)))

#     for t_time,i in zip(tqdm(times), range(len(times))):
#         idx = (full_flips < t_time).sum() - 1
#         if(idx < 0):
#             continue
#         myinterp = interp1d(full_flips[idx:idx+2], full_stimulus[:,:,idx:idx+2], axis=2)
#         interpolated_movie[:,:,int(i+pre_nframes)] = myinterp(t_time)

#         # NOTE : For compressed version, use the default settings associated with mp4 (libx264)
#         #        For the lossless version, use PNG codec and .avi output
    
    
#     overflow = np.where(interpolated_movie > 255)
#     underflow = np.where(interpolated_movie < 0)
#     interpolated_movie[overflow[0],overflow[1],overflow[2]] = 255
#     interpolated_movie[underflow[0],underflow[1],underflow[2]] = 0
#     f = lambda t: make_movie(t,interpolated_movie,target_hz)
#     clip = mpy.VideoClip(f, duration=(interpolated_movie.shape[-1]/target_hz))

#     clip.write_videofile(f"stimulus_17797_{key['session']}_{key['scan_idx']}_v3_compressed.mp4", codec='libx264', fps=target_hz)
#     clip.write_videofile(f"stimulus_17797_{key['session']}_{key['scan_idx']}_v3.avi", fps=target_hz, codec='png')

# def correct_scan(i,key):
#     """
#     Raster and motion-correct field i in scan specified by key
#     Args:
#         i:int:           field to raster and motion correct 
#         key:dictionary   scan to correct
#     Returns:
#         corrected_scan:np.array   field i raster and motion-corrected
    
#     """
#     scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
#     scan = scanreader.read_scan(scan_filename)
#     nframes = scan.num_frames
#     if experiment.Session.PMTFilterSet() & key & {'pmt_filter_set': '3P1 green-THG'}:
#         key['channel'] = 2
#     else:
#         key['channel'] = 1
#     pipe = (fuse.MotionCorrection() & key).module 

#     # Map: Correct scan in parallel
#     f = performance.parallel_correct_scan # function to map
#     raster_phase = (pipe.RasterCorrection() & key).fetch1('raster_phase')
#     fill_fraction = (pipe.ScanInfo() & key).fetch1('fill_fraction')
#     y_shifts, x_shifts = (pipe.MotionCorrection() & key).fetch1('y_shifts', 'x_shifts')
#     kwargs = {'raster_phase': raster_phase, 'fill_fraction': fill_fraction,
#             'y_shifts': y_shifts, 'x_shifts': x_shifts}
#     results = performance.map_frames(f, scan, field_id=i,
#                                     channel=0, kwargs=kwargs)

#     # Reduce: Rescale and save as int16
#     height, width, _ = results[0][1].shape
#     corrected_scan = np.zeros([height, width, nframes], dtype=np.int16)
#     max_abs_intensity = max(np.abs(c).max() for f, c in results)
#     scale = max_abs_intensity / (2 ** 15 - 1)
#     for frames, chunk in results:
#         corrected_scan[:, :, frames] = (chunk / scale).astype(np.int16)
    
#     return corrected_scan


# def generate_functional_scan(key,filename):
#     """
#         Creates composite by correcting each field and interleaving 
#         Args:
#             key        dictionary      Scan to generate 
#             filename   string          Filename to eventually save
    



#     """
#     # Create a composite by interleaving fields
#     # Read scan
#     from skimage.external.tifffile import imsave
#     print('Reading scan...')
#     scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
#     scan = scanreader.read_scan(scan_filename)
#     num_fields = (meso.ScanInfo.Field & key).fetch('field', order_by="field DESC", limit=1)[0]
#     nframes = scan.num_frames
#     height,width = (meso.ScanInfo().Field & key & 'field = 1').fetch1('px_height','px_width')
#     composite = np.zeros([num_fields * nframes, height, width], dtype=np.uint16)

#     for i in range(num_fields):
#         # Get some params
        
        
#         corrected_scan = correct_scan(i,key)
#         over = np.where(corrected_scan > 65535)
#         under = np.where(corrected_scan < 0)
#         corrected_scan[under[0],under[1],under[2]] = 0 
#         corrected_scan[over[0],over[1],over[2]] = 65535
#         composite[i::num_fields] = np.move_axis(corrected_scan,2,0)
    
#     imsave(filename,composite)
    


# def generate_stack(key,filename):
#     """
#     Grabs stack from the database and saves in filename.tif

#     Args:
#         key         dictionary      Stack to generate
#         filename    string          filename to save tiff file to    

    
#     """
#     from skimage.external.tifffile import imsave

#     # Create a composite interleaving channels
#     height, width, depth = (stack.CorrectedStack() & key).fetch1('px_height', 'px_width', 'px_depth')
#     num_channels = (stack.StackInfo() & (stack.CorrectedStack() & key)).fetch1('nchannels')
#     composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
#     for i in range(num_channels):
#         composite[i::num_channels] = (stack.CorrectedStack() & key).get_stack(i + 1)

#     imsave(filename,composite)
