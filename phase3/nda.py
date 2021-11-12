"""
Phase3 nda schema classes and methods
"""
import numpy as np
import datajoint as dj

schema = dj.schema('microns_phase3_nda', create_tables=True)
schema.spawn_missing_classes()

import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels
from .func import resize_movie,hamming_filter,get_timing_offset
from scipy.interpolate import interp1d
import imageio
from pipeline import meso
from stimulus import stimulus
from tqdm import tqdm
from pipeline import stack
from stimline import tune
import math
import io 

params = {'ignore_extra_fields':True,'skip_duplicates':True}

@schema
class Scan(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Information on completed scan
    session              : smallint                     # Session ID
    scan_idx             : smallint                     # Scan ID
    ---
    nframes              : int                          # number of frames per scan
    nfields              : tinyint                      # number of fields per scan
    fps                  : float                        # frames per second (Hz)
    """
    
    platinum_scans = [
        {'animal_id': 17797, 'session': 4, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 4, 'scan_idx': 9},
        {'animal_id': 17797, 'session': 4, 'scan_idx': 10},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 3},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 6},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 2},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 6},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 3},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 5},
        {'animal_id': 17797, 'session': 8, 'scan_idx': 5},
        {'animal_id': 17797, 'session': 8, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 8, 'scan_idx': 9},
        {'animal_id': 17797, 'session': 9, 'scan_idx': 3},
        {'animal_id': 17797, 'session': 9, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 9, 'scan_idx': 6}
    ]
        
    @property
    def key_source(self):
        return (meso.experiment.Scan * meso.ScanInfo).proj('filename', 'nfields', 'nframes', 'fps') & self.platinum_scans
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


class ScanInclude(dj.Lookup):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Scans suitable for analysis
    ->Scan
    """

    contents = np.array([
        [4, 7],
        [5, 6],
        [5, 7],
        [6, 2],
        [6, 4],
        [6, 6],
        [6, 7],
        [7, 3],
        [7, 4],
        [7, 5],
        [8, 5],
        [9, 3],
        [9, 4],
        [9, 6]])


@schema
class Field(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Individual fields of scans
    -> Scan
    field                : smallint                     # Field Number
    ---
    px_width             : smallint                     # field pixels per line
    px_height            : smallint                     # lines per field
    um_width             : float                        # field width (microns)
    um_height            : float                        # field height (microns)
    field_x              : float                        # field x motor coordinates (microns)
    field_y              : float                        # field y motor coordinates (microns)
    field_z              : float                        # field z motor coordinates (microns)
    """
    


    @property
    def key_source(cls):
        return meso.ScanInfo.Field.proj('px_width', 'px_height', 'um_width', 'um_height', field_x= 'x', field_y = 'y', field_z = 'z') \
                & {'animal_id': 17797} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class RawManualPupil(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Pupil traces
    -> Scan
    ---
    pupil_min_r          : longblob                     # vector of pupil minor radii  (pixels)
    pupil_maj_r          : longblob                     # vector of pupil major radii  (pixels)
    pupil_x              : longblob                     # vector of pupil x positions  (pixels)
    pupil_y              : longblob                     # vector of pupil y positions  (pixels)
    pupil_times          : longblob                     # vector of times relative to scan start (seconds)
    """
    @property
    def key_source(cls):
        return Scan().platinum_scans

    @classmethod
    def fill(cls):
        for key in cls.key_source:
            pupil = dj.create_virtual_module('pupil',"pipeline_eye")

            pupil_info = (pupil.FittedPupil.Ellipse & key & 'tracking_method = 1').fetch(order_by='frame_id ASC')
            raw_maj_r,raw_min_r = pupil_info['major_radius'],pupil_info['minor_radius']
            raw_pupil_x = [np.nan if entry is None else entry[0] for entry in pupil_info['center']]
            raw_pupil_y = [np.nan if entry is None else entry[1] for entry in pupil_info['center']]
            pupil_times = (pupil.Eye() & key).fetch1('eye_time')
            offset = (stimulus.BehaviorSync() & key).fetch1('frame_times')[0]

            adjusted_pupil_times = pupil_times - offset


            cls.insert1({'session':key['session'],'scan_idx':key['scan_idx'],
                                'pupil_min_r':raw_min_r,
                                'pupil_maj_r':raw_maj_r,
                                'pupil_x':raw_pupil_x,
                                'pupil_y':raw_pupil_y,
                                'pupil_times':adjusted_pupil_times
                                },**params)

@schema
class ManualPupil(dj.Manual):
    definition = """
    # Pupil traces
    -> RawManualPupil
    ---
    pupil_min_r          : longblob                     # vector of pupil minor radii synchronized with field 1 frame times (pixels)
    pupil_maj_r          : longblob                     # vector of pupil major radii synchronized with field 1 frame times (pixels)
    pupil_x              : longblob                     # vector of pupil x positions synchronized with field 1 frame times (pixels)
    pupil_y              : longblob                     # vector of pupil y positions synchronized with field 1 frame times (pixels)
    """

    @property 
    def key_source(cls):
        return RawManualPupil().proj(animal_id='17797') 

    @classmethod
    def fill(cls):
        for key in cls.key_source:
            stored_pupilinfo = (RawManualPupil() & key).fetch1()
            pupil_times = stored_pupilinfo['pupil_times']
            frame_times,ndepths = (FrameTimes() & key).fetch1('frame_times','ndepths')
            top_frame_scan_times_beh_clock = frame_times[::ndepths]

            if((key['session'] == 4) and (key['scan_idx']==9)):
                top_frame_scan_times_beh_clock = top_frame_scan_times_beh_clock[:-1]
            
            raw_pupil_x = [np.nan if entry is None else entry for entry in stored_pupilinfo['pupil_x']]
            raw_pupil_y = [np.nan if entry is None else entry for entry in stored_pupilinfo['pupil_y']]
            pupil_x_interp = interp1d(pupil_times, raw_pupil_x, kind='linear', bounds_error=False, fill_value=np.nan)
            pupil_y_interp = interp1d(pupil_times, raw_pupil_y, kind='linear', bounds_error=False, fill_value=np.nan)
            major_r_interp = interp1d(pupil_times, stored_pupilinfo['pupil_maj_r'], kind='linear', bounds_error=False, fill_value=np.nan)
            minor_r_interp = interp1d(pupil_times,stored_pupilinfo['pupil_min_r'],kind='linear',bounds_error=False,fill_value=np.nan)
            pupil_x = pupil_x_interp(top_frame_scan_times_beh_clock)
            pupil_y = pupil_y_interp(top_frame_scan_times_beh_clock)
            pupil_maj_r = major_r_interp(top_frame_scan_times_beh_clock)
            pupil_min_r = minor_r_interp(top_frame_scan_times_beh_clock)
            cls.insert1({**key,'pupil_min_r':pupil_min_r,'pupil_maj_r':pupil_maj_r,'pupil_x':pupil_x,'pupil_y':pupil_y},**params)


@schema
class RawTreadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->Scan
    ---
    treadmill_velocity      : longblob                     # vector of treadmill velocities (cm/s)
    treadmill_timestamps    : longblob                     # vector of times relative to scan start (seconds)
    """

    @property 
    def key_source(cls):
        return Scan().platinum_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            treadmill = dj.create_virtual_module("treadmill","pipeline_treadmill")
            treadmill_info = (treadmill.Treadmill & key).fetch1()
            frame_times_beh = (stimulus.BehaviorSync() & key).fetch1('frame_times')

            adjusted_treadmill_times = treadmill_info['treadmill_time'] - frame_times_beh[0]
            cls.insert1({**key,'treadmill_timestamps':adjusted_treadmill_times,'treadmill_velocity':treadmill_info['treadmill_vel']},**params)


@schema
class Treadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->RawTreadmill
    ---
    treadmill_speed      : longblob                     # vector of treadmill velocities synchronized with field 1 frame times (cm/s)
    """
    
    @property
    def key_source(cls):
        return RawTreadmill().proj(animal_id='17797')
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            tread_time, tread_vel = (RawTreadmill() & key).fetch1('treadmill_timestamps', 'treadmill_velocity')
            tread_interp = interp1d(tread_time, tread_vel, kind='linear', bounds_error=False, fill_value=np.nan)
            tread_time = tread_time.astype(np.float)
            tread_vel = tread_vel.astype(np.float)
            frame_times,ndepths = (FrameTimes() & key).fetch1('frame_times','ndepths')
            if((key['session'] == 4) and (key['scan_idx'] == 9)):
                top_frame_time = frame_times[::ndepths][:-1]
            else:
                top_frame_time = frame_times[::ndepths]
        
            interp_tread_vel = tread_interp(top_frame_time)
            treadmill_key = {
                'session': key['session'],
                'scan_idx': key['scan_idx'],
                'treadmill_speed': interp_tread_vel
            }
            cls.insert1(treadmill_key,**params)

        
    

@schema
class ScanTimes(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """ 
    # scan times per frame (in seconds, relative to the start of the scan)
    ->Scan
    ---
    frame_times        : longblob            # stimulus frame times for field 1 of each scan, (len = nframes)
    ndepths             : smallint           # number of imaging depths recorded for each scan
    """
    @property 
    def key_source(cls):
        return Scan().platinum_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            frame_times = (stimulus.BehaviorSync() & key).fetch1('frame_times')
            ndepths = len(dj.U('z') &  (meso.ScanInfo().Field() & key))
            cls.insert1({**key,'frame_times':frame_times - frame_times[0],'ndepths':ndepths},**params)

@schema
class Stimulus(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Stimulus presented
    -> Scan
    ---
    movie                : longblob                     # stimulus images synchronized with field 1 frame times (H x W X F matrix)
    """
    @property 
    def key_source(cls):
        return Scan().platinum_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            time_axis = 2
            target_size = (90, 160)
            full_stimulus = None
            full_flips = None

            num_depths = np.unique((meso.ScanInfo.Field & key).fetch('z')).shape[0]
            scan_times = (stimulus.Sync & key).fetch1('frame_times').squeeze()[::num_depths]
            trial_data = ((stimulus.Trial & key) * stimulus.Condition).fetch('KEY', 'stimulus_type', order_by='trial_idx ASC')
            for trial_key,stim_type in zip(tqdm(trial_data[0]), trial_data[1]):
                
                if stim_type == 'stimulus.Clip':
                    djtable = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
                    flip_times, compressed_clip, skip_time, cut_after,frame_rate = djtable.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
                    # convert to grayscale and stack to movie in width x height x time
                    temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
                    # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
                    temp_vid_length = temp_vid.count_frames()
                    movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)
                    assumed_clip_fps = frame_rate
                    start_idx = int(np.float(skip_time) * assumed_clip_fps)
                    print(trial_key)
                    end_idx = int(start_idx + (np.float(cut_after) * assumed_clip_fps))
                
                    movie = movie[:,:,start_idx:end_idx]
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                elif stim_type == 'stimulus.Monet2':
                    flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('flip_times', 'movie')
                    movie = movie[:,:,0,:]
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                elif stim_type == 'stimulus.Trippy':
                    flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('flip_times', 'movie')
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                else:
                    raise Exception(f'Error: stimulus type {stim_type} not understood')

            h,w,t = full_stimulus.shape
            interpolated_movie = np.zeros((h, w, scan_times.shape[0]))
            for t_time,i in zip(tqdm(scan_times), range(len(scan_times))):
                idx = (full_flips < t_time).sum() - 1
                if (idx < 0) or (idx >= full_stimulus.shape[2]-2):
                    interpolated_movie[:,:,i] = np.zeros(full_stimulus.shape[0:2])
                else:
                    myinterp = interp1d(full_flips[idx:idx+2], full_stimulus[:,:,idx:idx+2], axis=2)
                    interpolated_movie[:,:,i] = myinterp(t_time)
            

            overflow = np.where(interpolated_movie > 255)
            underflow = np.where(interpolated_movie < 0)
            interpolated_movie[overflow[0],overflow[1],overflow[2]] = 255
            interpolated_movie[underflow[0],underflow[1],underflow[2]] = 0


            if( (key['session'] == 4) and (key['scan_idx'] == 9)):
                interpolated_movie = interpolated_movie[:,:,:-1]

            cls.insert1({'movie':interpolated_movie.astype(np.uint8),'session':key['session'],'scan_idx':key['scan_idx']},**params)

@schema
class Trial(dj.Manual):
    definition = """
    # Information for each Trial
    ->Stimulus
    trial_idx            : smallint                     # index of trial within stimulus
    ---
    type                 : varchar(16)                  # type of stimulus trial
    start_idx            : int unsigned                 # index of field 1 scan frame at start of trial
    end_idx              : int unsigned                 # index of field 1 scan frame at end of trial
    start_frame_time     : double                       # start time of stimulus frame relative to scan start (seconds)
    end_frame_time       : double                       # end time of stimulus frame relative to scan start (seconds)
    stim_times           : longblob                     # full vector of stimulus frame times relative to scan start (seconds)
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    """
    @property 
    def key_source(cls):
        return Scan().platinum_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            data = ((stimulus.Trial() & key) * stimulus.Condition()).fetch(as_dict=True)
            ndepths = len(dj.U('z') & (meso.ScanInfo().Field() & key))

            offset = get_timing_offset(key)
        
            if((key['scan_idx'] == 9) and (key['session'] == 4)):
                
                frame_times = (stimulus.Sync() & key).fetch1('frame_times')
                field1_pixel0 = frame_times[::ndepths][:-1]
            else:
                frame_times = (stimulus.Sync() & key).fetch1('frame_times')[0]
                field1_pixel0 = frame_times[::ndepths]
            
            
            field1_pixel0 = field1_pixel0 - offset 


            for idx,trial in enumerate(tqdm(data)):
                trial_flip_times = trial['flip_times'].squeeze() - offset
                
                start_index = (field1_pixel0 < trial_flip_times[0]).sum() - 1
                end_index = (field1_pixel0 < trial_flip_times[-1]).sum() - 1
                start_time = trial_flip_times[0]
                end_time = trial_flip_times[-1]
                if(idx > 0):
                    if(data[idx-1]['end_idx'] == start_index):
                        print('ding')
                        t0 = data[idx-1]
                        med = (start_time  +t0['end_frame_time'])/2
                        nearest_frame_start = np.argmin(np.abs(field1_pixel0 - med))
                        data[idx-1]['end_idx'] = nearest_frame_start-1
                        start_index = nearest_frame_start
                

                
                data[idx]['start_frame_time'] = start_time  
                data[idx]['end_frame_time'] = end_time  
                data[idx]['start_idx'] = start_index
                data[idx]['end_idx'] = end_index
                data[idx]['frame_times'] = trial_flip_times
                data[idx]['type'] = data[idx]['stimulus_type']
                        
            cls.insert(data,**params)

@schema
class Clip(dj.Manual):
    definition = """
    # Movie clip condition
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    movie_name           : char(250)                    # full clip source
    duration             : decimal(7,3)                 # duration of clip (seconds)
    clip                 : longblob                     # clip used for stimulus (T x H x W)
    short_movie_name     : char(15)                     # clip type (cinematic, sports1m, rendered)
    fps                  : float                        # original framerate of clip
    """

    @property 
    def key_source(cls):
        return Scan().platinum_scans
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            movie_mapping = {
                'poqatsi':"Cinematic",
                'MadMax':   "Cinematic",
                'naqatsi':  "Cinematic",
                'koyqatsi': "Cinematic",
                'matrixrv': "Cinematic",
                'starwars': "Cinematic",
                'matrixrl': "Cinematic",
                'matrix':   "Cinematic",
            }

            long_movie_mapping = {
                'poqatsi':  "Powaqqatsi: Life in Transformation (1988)",
                'MadMax':   "Mad Max: Fury Road (2015)",
                'naqatsi':  "Naqoyqatsi: Life as War (2002)",
                'koyqatsi': "Koyaanisqatsi: Life Out of Balance (1982)",
                'matrixrv': "The Matrix Revolutions (2003)",
                'starwars': "Star Wars: Episode VII - The Force Awakens (2015)",
                'matrixrl': "The Matrix Reloaded (2003)",
                'matrix':   "The Matrix (1999)"}

            short_mapping = {'bigrun':'Rendered','finalrun':'Rendered','sports1m':'sports1m'}
            short_mapping = {**short_mapping,**movie_mapping}
            trials = stimulus.Trial() & key 
            movie_df = (trials.proj('condition_hash') * stimulus.Movie().Clip() * stimulus.Clip() * stimulus.Movie()).fetch(format='frame')
            movie_df = movie_df.reset_index()
            movie_df['new_movie_name'] = movie_df.apply(lambda x: x['parent_file_name'][:-4],axis=1)
            movie_df['new_movie_name'] = movie_df.apply(lambda x: x['new_movie_name'] if x['movie_name'] not in long_movie_mapping else long_movie_mapping[x['movie_name']],axis=1)
            for entry in movie_df.to_dict('records'): 
                clip, skip_time, cut_after, hz = entry['clip'],entry['skip_time'],entry['cut_after'],entry['frame_rate']
                
                vid = imageio.get_reader(io.BytesIO(clip.tobytes()), 'ffmpeg')
                _hz = 30
                frames = np.stack([frame.mean(axis=-1) for frame in vid], 0)
                total_frames = frames.shape[0]
                skip_time, cut_after = float(skip_time), float(cut_after)
                _start_frame = round(skip_time * _hz)
                _end_frame = _start_frame + round(cut_after * _hz)
                start_frame = min(math.ceil(_start_frame / _hz * hz), total_frames)
                end_frame = min(math.floor(_end_frame / _hz * hz), total_frames)
                cut_clip = frames[start_frame:end_frame,:,:]
                movie_name = movie_df[lambda df: df['condition_hash'] == entry['condition_hash']]['new_movie_name'].iloc[0]
                pack = {'condition_hash':entry['condition_hash'],
                        'movie_name':movie_name,
                        'duration':entry['cut_after'],
                        'clip_number':entry['clip_number'],
                        'clip':cut_clip,
                        'short_movie_name':short_mapping[entry['movie_name']],
                        'fps':entry['frame_rate']}
                cls.insert1(pack,**params)

@schema
class Monet2(dj.Manual):
    definition = """
    # Improved Monet stimulus: pink noise with periods of coherent motion
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    fps                  : decimal(6,3)                 # display refresh rate
    duration             : decimal(6,3)                 # (s) trial duration
    rng_seed             : double                       # random number generator seed
    blue_green_saturation=0.000 : decimal(4,3)          # 0 = grayscale, 1=blue/green
    pattern_width        : smallint                     # width of generated pattern
    pattern_aspect       : float                        # the aspect ratio of generated pattern
    temp_kernel          : varchar(16)                  # temporal kernel type (hamming, half-hamming)
    temp_bandwidth       : decimal(4,2)                 # (Hz) temporal bandwidth of the stimulus
    ori_coherence        : decimal(4,2)                 # 1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
    ori_fraction         : float                        # fraction of stimulus with coherent orientation vs unoriented
    ori_mix              : float                        # mixin-coefficient of orientation biased noise
    n_dirs               : smallint                     # number of directions
    speed                : float                        # (units/s)  where unit is display width
    directions           : longblob                     # computed directions of motion in degrees
    onsets               : blob                         # computed direction onset (seconds)
    movie                : longblob                     # rendered uint8 movie (H X W X 1 X T)
    """
    @property 
    def key_source(cls):
        return (stimulus.Trial().proj('condition_hash') * stimulus.Monet2 & Scan().platinum_scans) 

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source,**params)


@schema
class Trippy(dj.Manual):
    definition = """
    # randomized curvy dynamic gratings
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    fps                  : decimal(6,3)                 # display refresh rate (Hz)
    rng_seed             : double                       # random number generate seed
    packed_phase_movie   : longblob                     # phase movie before spatial and temporal interpolation
    tex_ydim             : smallint                     # (pixels) texture height
    tex_xdim             : smallint                     # (pixels) texture width
    duration             : float                        # (s) trial duration
    xnodes               : tinyint                      # x dimension of low-res phase movie
    ynodes               : tinyint                      # y dimension of low-res phase movie
    up_factor            : tinyint                      # spatial upscale factor
    temp_freq            : float                        # (Hz) temporal frequency if the phase pattern were static
    temp_kernel_length   : smallint                     # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
    spatial_freq         : float                        # (cy/point) approximate max. The actual frequencies may be higher.
    movie                : longblob                     # rendered movie (H X W X T)
    """
    @property 
    def key_source(cls):
        return (stimulus.Trial().proj('condition_hash') * stimulus.Trippy & Scan().platinum_scans) 

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source,**params)


@schema
class MeanIntensity(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # mean intensity of imaging field over time
    ->Field
    ---
    intensities    : longblob                     # mean intensity
    """
    
    @property
    def key_source(self):
        return meso.Quality.MeanIntensity & {'animal_id': 17797} & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
    
@schema
class SummaryImages(dj.Manual):
    definition = """
    ->Field
    ---
    correlation    : longblob                     # average image
    average        : longblob                     # correlation image
    """
    
    @property
    def key_source(self):
        return meso.SummaryImages.Correlation.proj(correlation='correlation_image') * meso.SummaryImages.L6Norm.proj(l6norm='l6norm_image') * meso.SummaryImages.Average.proj(average='average_image') & {'animal_id': 17797} & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class Stack(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # all slices of each stack after corrections.
    stack_session        : smallint                     # session index for the mouse
    stack_idx            : smallint                     # id of the stack
    ---
    motor_z                    : float                  # (um) center of volume in the motor coordinate system (cortex is at 0)
    motor_y                    : float                  # (um) center of volume in the motor coordinate system
    motor_x                    : float                  # (um) center of volume in the motor coordinate system
    px_depth             : smallint                     # number of slices
    px_height            : smallint                     # lines per frame
    px_width             : smallint                     # pixels per line
    um_depth             : float                        # depth in microns
    um_height            : float                        # height in microns
    um_width             : float                        # width in microns
    surf_z               : float                        # (um) depth of first slice - half a z step (cortex is at z=0)
    """
    
    platinum_stack = {'animal_id': 17797, 'stack_session': 9, 'stack_idx': 19}
    
    @property
    def key_source(self):
        return stack.CorrectedStack.proj(..., stack_session='session') & self.platinum_stack
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Registration(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # align a 2-d scan field to a stack with affine matrix learned via gradient ascent
    ->Stack
    ->Field
    ---
    a11                  : float                        # (um) element in row 1, column 1 of the affine matrix
    a21                  : float                        # (um) element in row 2, column 1 of the affine matrix
    a31                  : float                        # (um) element in row 3, column 1 of the affine matrix
    a12                  : float                        # (um) element in row 1, column 2 of the affine matrix
    a22                  : float                        # (um) element in row 2, column 2 of the affine matrix
    a32                  : float                        # (um) element in row 3, column 2 of the affine matrix
    reg_x                : float                        # z translation (microns)
    reg_y                : float                        # y translation (microns)
    reg_z                : float                        # z translation (microns)
    score                : float                        # cross-correlation score (-1 to 1)
    reg_field            : longblob                     # extracted field from the stack in the specified position
    """
    
    @property
    def key_source(self):
        return stack.Registration.Affine.proj(..., session='scan_session') & {'animal_id': 17797} & Stack & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class Coregistration(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # transformation solutions between 2P stack and EM stack and vice versa from the Allen Institute
    ->Stack
    transform_id            : int                          # id of the transform
    ---
    version                 : varchar(256)                 # coordinate framework
    direction               : varchar(16)                  # direction of the transform (EMTP: EM -> 2P, TPEM: 2P -> EM)
    transform_type          : varchar(16)                  # linear (more rigid) or spline (more nonrigid)
    transform_args=null     : longblob                     # parameters of the transform
    transform_solution=null : longblob                     # transform solution
    """
    
    @property
    def key_source(self):
        return m65p3.Coregistration()
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)



@schema
class Segmentation(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Different mask segmentations
    ->Field
    mask_id         :  smallint
    ---
    pixels          : longblob      # indices into the image in column major (Fortran) order
    weights         : longblob      # weights of the mask at the indices above
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.Segmentation.Mask & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Fluorescence(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # fluorescence traces before spike extraction or filtering
    -> Segmentation
    ---
    trace                   : longblob #fluorescence trace 
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.Fluorescence.Trace & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class ScanUnit(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # single unit in the scan
    -> Scan
    unit_id                 : int               # unique per scan
    ---
    -> Fluorescence
    um_x                : smallint      # centroid x motor coordinates (microns)
    um_y                : smallint      # centroid y motor coordinates (microns)
    um_z                : smallint      # centroid z motor coordinates (microns)
    px_x                : smallint      # centroid x pixel coordinate in field (pixels
    px_y                : smallint      # centroid y pixel coordinate in field (pixels
    ms_delay            : smallint      # delay from start of frame (field 1 pixel 1) to recording ot his unit (milliseconds)
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return (meso.ScanSet.Unit * meso.ScanSet.UnitInfo) & self.segmentation_key & Field  
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class Activity(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # activity inferred from fluorescence traces
    -> ScanUnit
    ---
    trace                   : longblob  #spike trace
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return meso.Activity.Trace & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class StackUnit(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # centroids of each unit in stack coordinate system using affine registration
    -> Registration
    -> ScanUnit
    ---
    motor_x            : float    # centroid x stack coordinates with motor offset (microns)
    motor_y            : float    # centroid y stack coordinates with motor offset (microns)
    motor_z            : float    # centroid z stack coordinates with motor offset (microns)
    stack_x            : float    # centroid x stack coordinates (microns)
    stack_y            : float    # centroid y stack coordinates (microns)
    stack_z            : float    # centroid z stack coordinates (microns)
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.StackCoordinates.UnitInfo & self.segmentation_key & Stack & Field
    
    @classmethod
    def fill(cls):
        stack_unit = (cls.key_source*Stack).proj(stack_x = 'round(stack_x - x + um_width/2, 2)', stack_y = 'round(stack_y - y + um_height/2, 2)', stack_z = 'round(stack_z - z + um_depth/2, 2)')
        cls.insert((cls.key_source.proj(motor_x='stack_x', motor_y='stack_y', motor_z='stack_z') * stack_unit), **params)

@schema 
class AreaMembership(dj.Manual):
    definition = """
    -> ScanUnit
    ---
    brain_area          : char(10)    # Visual area membership of unit
    
    """
    @property 
    def key_source(cls):
        return Scan().platinum_scans 
    
    def fill(cls):
        units = (meso.AreaMembership().UnitInfo() & {'ret_hash':'edec10b648420dd1dc8007b607a5046b'}  & 'segmentation_method = 6' & cls.key_source).fetch('session','scan_idx','brain_area','unit_id',as_dict=True)
        cls.insert(units)



@schema
class MaskClassification(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # classification of segmented masks using CaImAn package
    ->Segmentation
    ---
    mask_type                 : varchar(16)                  # classification of mask as soma or artifact
    """

    @property
    def key_source(self):
        return meso.MaskClassification.Type.proj(mask_type='type') & {'animal_id': 17797, 'segmentation_method': 6} & Scan

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Oracle(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Leave-one-out correlation for repeated videos in stimulus.
    -> ScanUnit
    ---
    trials               : int                          # number of trials used
    pearson              : float                        # per unit oracle pearson correlation over all movies
    """
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return tune.MovieOracle.Total & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)



@schema
class ScanHash(dj.Manual):
    definition = """
    
    scan_id     :   bigint       #
    --- 
    -> Scan
    """
    @property
    def key_source(cls):
        return Scan() 
    
    def fill(cls):
        for key in cls.key_source :
            bin_animal_id = bin(key['animal_id'][2:])
            bin_session = bin(key['session'][2:])

            if(len(bin_session) < 4):
                bin_session = '0'*(4-len(bin_session)) + bin_session
            bin_scan = bin(key['session'][2:])

            if(len(bin_scan < 4)):
                bin_scan = '0'*(4-len(bin_scan)) + bin_scan
            
            scan_id = (bin_animal_id << 8) + (bin_session << 4) + bin_scan
            cls.insert1({'scan_id':scan_id,**key},**params)
        
    

    def decode(scan_id):
        animal_id_extract = ((2**15)-1) << 8
        session_extract = ((2**4)-1) << 4
        scan_extract = ((2**4)-1)
        

        animal_id = (scan_id['animal_id'] & animal_id_extract) >> 8
        session = (scan_id['animal_id'] & session_extract) >> 4
        scan_idx = scan_id['animal_id'] & scan_extract

        return {'animal_id':animal_id,'session':session,'scan_idx':scan_idx}


        

        



            



