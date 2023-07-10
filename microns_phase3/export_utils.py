import os
import cv2
import numpy as np
import multiprocessing as mp
from itertools import repeat
from scipy import signal
from sklearn.cluster import KMeans
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

import datajoint as dj

# private BCM schemas
from commons import lab
from pipeline.utils import h5
from pipeline import experiment, meso, stack
from stimulus import stimulus
pixeltune = dj.create_virtual_module('pipeline_pixeltune','pipeline_pixeltune')


def resize_movie(movie, target_size, resize_method='inter_area',time_axis=2):
    """
    Resizes movie to target_size using cv2.resize (inter area interpolation)
    
    Args:
        movie:np.array                 movie to resize 
        target_size:array-like         array-like object representing target size (HxW)
        resize_method:str              key to lookup table for cv2 resize methods
        time_axis:int                  integer indicated which axis represents time (defaults to 2) 
    
    
    """
    
    method_lookup = {'inter_area':cv2.INTER_AREA,
                     'bilinear':cv2.INTER_LINEAR}
    
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
                                                 target_size[::-1], interpolation=method_lookup[resize_method])
    return new_movie


def hamming_filter(movie, source_times, target_times, time_axis=2):
    """
    Hamming filter to change framerate of movie from source_times framerate to target_times framerate at time_axis 
    Args:
        movie:np.array              movie to filter 
        source_times:np.array       numpy array representing timestamps of frames in movie 
        target_times:np.array       numpy array representing target timestamps of frames in movie 
        filter_size:int             integer representing size of hamming window
        time_axis:int               integer representing time axis of movie 
    """
    source_hz = 1/np.nanmedian(np.diff(source_times))
    target_hz = 1/np.nanmedian(np.diff(target_times))
    filter_size = np.round(2 * source_hz + 1).astype(int)
    scipy_ham = signal.firwin(filter_size, cutoff=target_hz/2, window="hamming", fs=source_hz)
    filtered_movie = convolve1d(movie, scipy_ham, axis=time_axis)
    return filtered_movie


def detect_loading_onset(scan_key, loading_screen_value=128,gamma=1.74):
    """
    Detects the onset of psychtoolbox loading, when screen goes from pixel value zero to pixel value 128, 
    and returns the time from the start of the scan to the loading onset
    Args:
        scan_key    restricts to a single scan to calculate loading onset delay
    Returns:
        loading_onset_delay:float   loading onset time - scan start time in beh clock (seconds)
    """

#     TODO loading screen expected pixel value has some hard coded values 
#     (gamma, loading screen value, etc) that are valid for this release but may not be valid for all scans

    est_refresh_rate = 60

    # load pd trace for experimental scan
    scan_path = (experiment.Scan & scan_key).local_filenames_as_wildcard
    scan_dir = os.path.split(scan_path)[0]

    behavior_file = (experiment.Scan.BehaviorFile() & scan_key).fetch1('filename')
    full_beh_file = os.path.join(scan_dir, behavior_file)

    data = h5.read_behavior_file(full_beh_file)

    # correct from cycling counter to seconds
    photodiode_times = h5.ts2sec(data['ts'], is_packeted=True)
    photodiode = data['syncPd']

    # resample at approx 2x monitor refresh rate
    photodiode_freq = 1/np.nanmedian(np.diff(photodiode_times))
    resample_int = (np.floor(photodiode_freq/est_refresh_rate)/2).astype(int)
    photodiode = photodiode[::resample_int]
    photodiode_times = photodiode_times[::resample_int]

    # linear interpolation from stimulus clock to behavior clock
    stim_scan_times = (stimulus.Sync & scan_key).fetch1('frame_times')
    if not (scan_key['session'] == 4 and scan_key['scan_idx'] == 9):
        stim_scan_times = stim_scan_times[0]
    beh_scan_times = (stimulus.BehaviorSync & scan_key).fetch1('frame_times')
    stim2beh = interp1d(stim_scan_times,beh_scan_times)

    # onset and offset flips of stimulus
    stim_onset = (stimulus.Trial & scan_key).fetch('flip_times',order_by='trial_idx ASC',limit=1)[0][0][0]
    stim_onset_idx = np.argmin(np.abs(photodiode_times-stim2beh(stim_onset)))
    stim_offset = (stimulus.Trial & scan_key).fetch('flip_times',order_by='trial_idx DESC',limit=1)[0][0][-1]
    stim_offset_idx = np.argmin(np.abs(photodiode_times-stim2beh(stim_offset)))

    # find photodiode samples occuring during stimulus presentation
    stim_idx = np.logical_and(photodiode_times>=stim2beh(stim_onset),photodiode_times<=stim2beh(stim_offset))

    # cluster photodiode samples into 3 flip amplitudes
    labels = KMeans(n_clusters=3).fit_predict((photodiode[stim_idx][:,None]))

    # find peaks of each cluster
    peaks = []
    for label in sorted(list(set(labels))):
        counts,bins = np.histogram(photodiode[stim_idx][labels==label],bins=100)
        peaks.append(np.mean(bins[np.argmax(counts):np.argmax(counts)+2]))

    # save flip_max and flip_min
    flip_max,flip_min = np.max(peaks),np.min(peaks)

    # calculate expected pd for flip min/max range for loading_screen_value
    loading_thresh = (flip_min + flip_max * ((loading_screen_value/255)**gamma)) * 0.75

    # detect time in beh clock of first flip to go over that threshold as loading onset relative to scan start
    loading_onset_time = photodiode_times[np.min(np.where(photodiode > loading_thresh)[0])]

    # calculate delay from scan onset to loading onset in behavior clock
    scan_onset_time = beh_scan_times[0]
    loading_onset_delay = loading_onset_time - scan_onset_time

    return loading_onset_delay


def reconstruct_refresh_rate_trial(trial_key,stim_type,intertrial_time,
                                   target_size,time_axis,resize_method,tol,est_refresh_rate,pool_cores):
    """
    Fetches stimulus video, resize h and w to target size, replicate frames at refresh rate between known flip times
    
    Args:
        trial_key                      restricts to single trial in single scan
        stim_type:str                  identifies the trial stimulus type
        intertrial_time                time from last flip of trial to first flip of following trial (s)
        target_size:array-like         array-like object representing target size (HxW)
        time_axis:int                  integer indicated which axis represents time (defaults to 2) 
        resize_method:str              method by which each frame will be resized to target size
        tol:float                      tolerance for flip time deviation from expected frame rate
        est_refresh_rate:int           estimated underlying monitor refresh rate, Hz
        pool_cores:int                 number of multiprocessing cores to use
    
    Returns:
        trial_flip_times:              trial time stamps at refresh rate and added intertrial frames
        trial_movie:np.array           trial movie, resampled at refresh rate and added intertrial frames
    """
    import io
    import imageio

    cond_rel = (stimulus.Trial & trial_key) * stimulus.Condition

    if stim_type == 'stimulus.Clip':
        cond_rel = (cond_rel * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
        flip_times, compressed_clip, skip_time, cut_after,frame_rate = cond_rel.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
        temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
        # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
        temp_vid_length = temp_vid.count_frames()
        # convert to grayscale and stack to movie along time_axis
        movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=time_axis)

        # slice frames between start time and start time + cut after
        start_idx = int(np.float(skip_time) * frame_rate)
        end_idx = int(start_idx + (np.float(cut_after) * frame_rate))
        movie = movie[:,:,start_idx:end_idx]

    elif stim_type == 'stimulus.Monet2':
        cond_rel = (cond_rel * stimulus.Monet2)
        flip_times, movie, bg_sat, frame_rate = cond_rel.fetch1('flip_times', 'movie',
                                                                'blue_green_saturation','fps')
        assert bg_sat == 0.0, 'not configured for dual channel Monet2'
        movie = movie[:,:,0,:]

    elif stim_type == 'stimulus.Trippy':
        cond_rel = (cond_rel * stimulus.Trippy)
        flip_times, movie, frame_rate = cond_rel.fetch1('flip_times', 'movie','fps')

    else:
        raise Exception(f'Error: stimulus type {stim_type} not understood')
        
    if flip_times.shape[-1] != movie.shape[-1]:
        print(f'flip / frame mismatch:\n {flip_times.shape[-1]} flips != {movie.shape[-1]} movie frames')
        crop_length = np.min((flip_times.shape[-1],movie.shape[-1]))
        print(f'cropping to length {crop_length}')
        flip_times = flip_times[...,:crop_length]
        movie = movie[...,:crop_length]

    # resize to target pixel resolution
    movie = resize_movie(movie, target_size, resize_method=resize_method,time_axis=time_axis)

    frame_rate = float(frame_rate)

    assert ((est_refresh_rate / frame_rate) % 1) == 0, 'refresh rate not integer multiple of frame rate'
    upsample_ratio = int(est_refresh_rate / frame_rate)


    # detect deviant flips with abnormal frame rates
    # note: only session 8 scan 9 had intratrial hanging frames, trial 0, total 33 ms intratrial delay
    dev_flips = np.abs(np.diff(flip_times) - (1/frame_rate))>tol

    if np.any(dev_flips):
        # split into blocks of uniform frame rates with hanging final frame
        dev_flip_frames = np.diff(flip_times)[0][np.where(dev_flips)[1]] * est_refresh_rate
        assert np.all((np.abs(np.round(dev_flip_frames) - dev_flip_frames)/est_refresh_rate) < tol),\
                'non-integer dropped frames detected'
        print(f'intratrial dropped frames detected, filling independently at {est_refresh_rate} Hz')
        print(trial_key)

        interblock_times = [*np.diff(flip_times)[0][np.where(dev_flips)[1]],intertrial_time]
        block_flip_sets = [np.asarray(a) for a in np.split(flip_times[0],np.where(dev_flips)[1]+1)]
        block_movies = [np.asarray(a) for a in np.split(movie,np.where(dev_flips)[1]+1,axis=time_axis)]
    else:
        # if not deviant frames format as list for iteration of 1
        interblock_times = [intertrial_time]
        block_movies = [movie]
        block_flip_sets = flip_times

    trial_flip_times,trial_movie = [],[]
    for interblock_time,block_flips,block_movie in zip(interblock_times,block_flip_sets,block_movies):

        assert np.all(np.abs(np.diff(block_flips)-(1/frame_rate)) < tol), 'frame rate deviation > 1 ms detected'

        # last flip and interblock period at refresh rate
        interblock_frames = (np.round(interblock_time * est_refresh_rate)-1).astype(int)

        # linearly interpolated interblock flip times at refresh rate
        interblock_flip_times = np.linspace(block_flips[-1],
                                            block_flips[-1]+interblock_time,
                                            interblock_frames+2)[0:-1]

        # repeat last stim frame for interblock period at refresh rate
        interblock_movie = np.repeat(block_movie[:,:,-1:],interblock_frames+1,axis=time_axis)

        if upsample_ratio > 1:
            # index to flip time linear interpolation
            idx2ft = interp1d(np.arange(0,len(block_flips)*upsample_ratio,upsample_ratio),
                              block_flips,kind='linear') 

            # linearly interpolated intrablock flip times at refresh rate
            intrablock_flip_times = idx2ft(np.arange(0,(len(block_flips)-1)*upsample_ratio))

            # repeat intrablock stim frames to fill refresh rate
            intrablock_movie = np.repeat(block_movie[:,:,:-1],upsample_ratio,axis=time_axis)

        elif upsample_ratio == 1:
            intrablock_flip_times = block_flips[:-1]
            intrablock_movie = block_movie[:,:,:-1]

        # concatenate
        trial_flip_times.append(np.concatenate((intrablock_flip_times,interblock_flip_times)))
        trial_movie.append(np.concatenate((intrablock_movie,interblock_movie),axis=time_axis))

    # concatenate
    trial_flip_times = np.concatenate(trial_flip_times,axis=0)
    trial_movie = np.concatenate(trial_movie,axis=time_axis)

    return trial_flip_times,trial_movie

def resample_stim(scan_key, target_times=None, target_size = (90,160), time_axis=2, 
                  resize_method='inter_area',tol=1e-3, est_refresh_rate = 60, 
                  loading_px = 128, pool_cores = 20):
    """
    Fetches stimulus video, resize h and w to target size, applies hamming filter, concatenates 
    and resamples at target times
    
    Args:
        scan_key                       restricts to all Trials of a single scan
        target_times:np.array          time points at which stimulus will be resampled
        target_size:array-like         array-like object representing target size (HxW)
        time_axis:int                  integer indicated which axis represents time (defaults to 2) 
        resize_method:str              method by which each frame will be resized to target size
        tol:float                      tolerance for flip time deviation from expected frame rate
        est_refresh_rate:int           estimated underlying monitor refresh rate, Hz
        loading_px:int                 pixel value of loading screen 
        pool_cores:int                 number of multiprocessing cores to use
    
    Returns:
        interpolated_movie:np.array    resampled stimulus movie
    """
    
    
    assert len(experiment.Scan & scan_key) == 1, 'scan_key does not restrict to a single scan'

    # get all trial keys and corresponding stimulus types
    trial_rel = ((stimulus.Trial & scan_key) * stimulus.Condition)
    trial_keys,stim_types,flip_times = trial_rel.fetch('KEY','stimulus_type','flip_times',order_by='trial_idx ASC')
    intertrial_times = np.array([flip_times[i+1][0][0] - flip_times[i][0][-1] for i in range(len(flip_times)-1)])
    
    # empirical frame rate in stimulus clock
    emp_refresh_rate = 1/np.mean(np.hstack([np.diff(ft) for ft in flip_times if (1/np.mean(np.diff(ft))) > (.75*est_refresh_rate)]))


    # assume median intertrial_period approximates final intertrial period following last trial
    intertrial_times = np.append(intertrial_times,np.median(intertrial_times))

    # check that intertrial time is integer multiple of refresh frame rate
    assert np.all(np.abs(np.round(intertrial_times * est_refresh_rate) - (intertrial_times*est_refresh_rate)) < tol), \
                f'intertrial frame rate deviation > {np.round(tol*1000).astype(int)} ms detected'
    
    # multiprocess trials, starmap returns results in same order as iterable
    with mp.Pool(pool_cores) as pool:
        trial_iter = zip(trial_keys, stim_types,intertrial_times,
                         repeat(target_size),repeat(time_axis),repeat(resize_method),
                         repeat(tol),repeat(est_refresh_rate),repeat(pool_cores))
        results = pool.starmap(reconstruct_refresh_rate_trial,trial_iter)
    
    # concatenate multiprocessing results across time
    full_flips, full_stimulus = np.array(results).T
    full_flips = np.concatenate(full_flips)
    full_stimulus = np.concatenate(full_stimulus,axis=time_axis)

    # get scan times in stimulus clock
    scan_times = (stimulus.Sync & scan_key).fetch1('frame_times')
    if not (scan_key['session'] == 4 and scan_key['scan_idx'] == 9):
        scan_times = scan_times[0]
    scan_onset_time, scan_offset_time = scan_times[0],scan_times[-1]

    # detect scan duration preceding stimulus onset, create prepad movie at refresh rate to encompass
    prepad_frames = np.ceil((full_flips[0]-scan_onset_time)*emp_refresh_rate).astype(int)
    prepad_flips = np.linspace(full_flips[0] - (prepad_frames/emp_refresh_rate), 
                               full_flips[0], prepad_frames, endpoint=False)
    prepad_movie = np.zeros((*target_size,prepad_frames))

    # detect loading period and set prepad movie pixel value
    # loading_onset_delay = detect_loading_onset(scan_key,loading_screen_value=loading_px)
    loading_onset_delay = detect_loading_onset(scan_key)

    loading_onset_idx = np.ceil((scan_onset_time - prepad_flips[0] + loading_onset_delay)*emp_refresh_rate).astype(int)
    prepad_movie[...,loading_onset_idx:] = loading_px

    # detect scan duration following stimulus offset, create postpad movie at refresh rate to encompass
    postpad_frames = np.ceil((scan_offset_time - full_flips[-1])*emp_refresh_rate).astype(int)
    postpad_flips = np.linspace(full_flips[-1],full_flips[-1]+postpad_frames/emp_refresh_rate,
                                postpad_frames+1,endpoint=True)[1:]
    postpad_movie = np.zeros((*target_size,postpad_frames))

    # concatenate prepad, stimulus, and postpad
    full_stimulus = np.concatenate((prepad_movie,full_stimulus,postpad_movie),axis=time_axis)
    full_flips = np.concatenate((prepad_flips,full_flips,postpad_flips))

    if target_times is not None:

        # check if frequency of target times differs from refresh rate before applying filter
        target_freq = 1/np.median(np.diff(target_times))
        if (emp_refresh_rate - target_freq) > tol:
            # apply hamming filter before temporal resample
            full_stimulus = hamming_filter(full_stimulus, full_flips, target_times, time_axis=time_axis)

        # time to stimulus linear interpolation object
        t2s = interp1d(full_flips,full_stimulus,axis=2,kind='linear')

        # linearly interpolate frames for duration of stimulus
        interpolated_movie = t2s(target_times)

    else:
        interpolated_movie = full_stimulus

    # clip overflow/underflow pixels introduced by hamming filter before recast to uint8
    interpolated_movie = np.clip(interpolated_movie,0,255).astype(np.uint8)

    return interpolated_movie, emp_refresh_rate


def reconstruct_stimulus(path, key, version, est_refresh_rate=60):
    
    """
    Generates an avi/mp4 of the stimulus from frames stored in the database 
    Args:
        path:str               path to save directory
        key:dict               scan for which to generate the avi/mp4
        version:int            version of save file
        est_refresh_rate:int   estimated underlying monitor refresh rate, Hz

    Returns:
        None, writes stimulus avi/mp4 at path directory
    """
    import moviepy.editor as mpy
    
    key_str = '_'.join([str(s) for s in (experiment.Scan.proj() & key).fetch1('KEY').values()])
    assert len(experiment.Scan & key) == 1, 'key does not restrict to a single scan'
    
    interpolated_movie,emp_refresh_rate = resample_stim(key, 
                                                        est_refresh_rate=est_refresh_rate,
                                                        target_size = (144,256),
                                                        resize_method='bilinear',
                                                        tol = 2e-3)
    
    def make_movie(t,interpolated_movie,est_refresh_rate):
        frame = interpolated_movie[:,:,int(round(t*est_refresh_rate))]
        return np.repeat(frame[:, :, np.newaxis], 3, axis=2)

    f = lambda t: make_movie(t,interpolated_movie,est_refresh_rate)
    clip = mpy.VideoClip(f, duration=(interpolated_movie.shape[-1]/est_refresh_rate))

    filename = path + 'stimulus_' + key_str + f'_v{str(version)}.avi'
    clip.write_videofile(filename, codec='png', fps=est_refresh_rate)

    filename = path + 'stimulus_' + key_str + f'_v{str(version)}_compressed.mp4'
    clip.write_videofile(filename, codec='libx264', fps=est_refresh_rate)
    
    
def reconstruct_stack(path, key, version):
    """
    Generates and saves stack tif from database
    Args:
        path:str         path to save directory
        key:dictionary   stack to reconstruct
        version:int      version of save file
       
    Returns:
        None, writes original/resized stack tifs at path directory
    
    """
    from skimage.external.tifffile import imsave
    
    key_str = '_'.join([str(s) for s in (experiment.Stack.proj() & key).fetch1('KEY').values()])

    num_channels = (stack.StackInfo() & (stack.CorrectedStack() & key)).fetch1('nchannels')
    
    # Create a composite interleaving channels at original pixel resolution
    height, width, depth = (stack.CorrectedStack() & key).fetch1('px_height', 'px_width', 'px_depth')
    composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
    for i in range(num_channels):
        composite[i::num_channels] = (stack.CorrectedStack() & key).get_stack(i + 1)
        
    # save original stack to directory
    filename = path + 'two_photon_stack_' + key_str + f'_v{str(version)}.tif'
    imsave(filename,composite)
    
    # Create a composite interleaving channels at resized 1px/um resolution
    height, width, depth = (stack.CorrectedStack() & key).fetch1('um_height', 'um_width', 'um_depth')
    height, width, depth = [int(np.round(c)) for c in (height,width,depth)]
    composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
    for i in range(num_channels):
        composite[i::num_channels] = (stack.PreprocessedStack & key & f'channel = {i+1}').fetch1('resized')

    # save resized stack to directory
    filename = path + 'two_photon_stack_' + key_str + f'_resized_v{str(version)}.tif'
    imsave(filename,composite)    
    
    
def reconstruct_scan(path,scan_key,version):
    """
    Generates and saves corrected scan tif from database
    Args:
        path:str         path to save directory
        key:dictionary   scan to reconstruct
        version:int      version of save file
       
    Returns:
        None, writes scan tifs at path directory
    
    """
    from skimage.external.tifffile import imsave
    from stimline import pixeltune

    dim_rel = dj.U('px_height','px_width') & (meso.ScanInfo.Field & scan_key)
    assert len(dim_rel) == 1, 'more than one set of pixel dimensions across fields'
    height,width = dim_rel.fetch1('px_height','px_width')

    nfields,nframes = (meso.ScanInfo & scan_key).fetch1('nfields','nframes')
    assert len(pixeltune.CaMovie & scan_key) == nfields, 'not all fields have corrected calcium movies'
    field_keys = (pixeltune.CaMovie & scan_key).fetch('KEY',order_by='field ASC')

    composite = np.zeros([nframes*nfields,height,width],dtype=np.int16)
    for i,field_key in enumerate(field_keys):
        print('fetching: ', field_key)
        corrected_scan = (pixeltune.CaMovie & field_key).fetch1('corrected_scan')
        print('interleaving')
        composite[i::nfields,:,:] = np.moveaxis(corrected_scan,2,0) #[(frames*fields) x height x width)

        
    print('starting save')
    key_str = '_'.join([str(s) for s in (experiment.Scan.proj() & scan_key).fetch1('KEY').values()])
    filename = path + 'functional_scan_' + key_str + '_v' + str(int(version)) + '.tif'

    imsave(filename,composite)
