import cv2
import numpy as np
from scipy import stats, signal
from scipy.ndimage import convolve1d
from inhib import inhib_nda as nda
import matplotlib.pyplot as plt
import datajoint as dj
from scipy.interpolate import interp1d

# private BCM schemas
# from stimulus.stimulus import BehaviorSync,Sync


def resize_movie(movie, target_size, time_axis=2):
    """
    Resizes movie to target_size using cv2.resize (inter area interpolation)
    
    Args:
        movie:np.array                 movie to resize 
        target_size:array-like         array-like object representing target size (HxW)
        time_axis:int                  integer indicated which axis represents time (defaults to 2) 
    
    
    """
    new_shape = list(target_size)
    new_shape.insert(time_axis, movie.shape[time_axis])
    new_movie = np.zeros(new_shape)
    # This structure can be used to select slices from a dynamically specified axis
    # Use by setting idx_slice[dynamic_axis] = idx OR slice(start,end) and then array[tuple(idx_slice)]
    idx_slice = [slice(None)] * movie.ndim
    for i in range(movie.shape[time_axis]):
        idx_slice[time_axis] = i
        # cv2 uses height x width reversed compared to numpy, target_size is flipped
        # INTER_AREA interpolation appears best for downsample and upsample
        # replace with bilinear interpolation
        new_movie[tuple(idx_slice)] = cv2.resize(movie[tuple(idx_slice)], 
                                                 target_size[::-1], interpolation=cv2.INTER_AREA)
        
        
    return new_movie


def hamming_filter(movie, time_axis, source_times, target_times, filter_size=20):
    """
    Hamming filter to change framerate of movie from source_times framerate to target_times framerate at time_axis 
    Args:
        movie:np.array              movie to filter 
        time_axis:int               integer representing time axis of movie 
        source_times:np.array       numpy array representing timestamps of frames in movie 
        target_times:np.array       numpy array representing target timestamps of frames in movie 
        filter_size:int             integer representing size of hamming window
    
    """
    source_hz = 1/np.median(np.diff(source_times))
    target_hz = 1/np.median(np.diff(target_times))
    scipy_ham = signal.firwin(filter_size, cutoff=target_hz/2, window="hamming", fs=source_hz)
    filtered_movie = convolve1d(movie, scipy_ham, axis=time_axis)
    return filtered_movie

def get_timing_offset(key):
    """
    Fetches timing offset between behavior and stimulus computers.
    Args:
        key: dictionary specifying the scan 
    Returns:
        photodiode_zero: float indicating the offset between stimulus and behavior time keeping
    
    
    """
    frame_times = (Sync() & key).fetch1('frame_times')
    frame_times_beh = (BehaviorSync() & key).fetch1('frame_times')
    photodiode_zero = np.nanmedian(frame_times - frame_times_beh)
    return photodiode_zero

def make_movie(t,interpolated_movie,target_hz):
    frame = interpolated_movie[:,:,int(round(t*target_hz))]
    return np.repeat(frame[:, :, np.newaxis], 3, axis=2)

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



