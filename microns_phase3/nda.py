"""
MICrONS phase3 nda schema classes and methods
"""

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import datajoint as dj

# # BCM specific export schemas
# from stimulus import stimulus
# from stimline import tune
# from pipeline import meso, experiment, stack
# m65p3 = dj.create_virtual_module('microns_minnie65_02','microns_minnie65_02')

schema = dj.schema('microns_phase3_nda', create_tables=True)
# schema.spawn_missing_classes()

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


@schema
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
        return meso.ScanInfo.Field.proj('px_width', 'px_height', 'um_width', 'um_height', 
                                        field_x= 'x', field_y = 'y', field_z = 'z') \
                & {'animal_id': 17797} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        
@schema
class RasterCorrection(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Raster Correction applied to each field
    -> Field
    ---
    raster_phase         : float                        # difference between expected and recorded scan angle
    """
    
    @property
    def key_source(self):
        return meso.RasterCorrection & {'animal_id':17797} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        

@schema
class MotionCorrection(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Motion Correction applied to each field
    -> Field
    ---
    y_shifts             : longblob                     # y motion correction shifts (pixels) 
    x_shifts             : longblob                     # x motion correction shifts (pixels) 
    y_std                : float                        # standard deviation of y shifts (um)
    x_std                : float                        # standard deviation of x shifts (um)
    """
    
    @property
    def key_source(self):
        return meso.MotionCorrection & {'animal_id':17797} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
    

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
            nframes = (Scan & key).fetch1('nframes')
            ndepths = len(dj.U('z') &  (meso.ScanInfo().Field() & key))
            frame_times = frame_times[:nframes*ndepths:ndepths] - frame_times[0]
            cls.insert1({**key,
                         'frame_times':frame_times,
                         'ndepths'    :ndepths},**params)
            



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
    correlation    : longblob                     # correlation image
    average        : longblob                     # average image
    """
    
    @property
    def key_source(self):
        return (meso.SummaryImages.Correlation.proj(correlation='correlation_image') *
                meso.SummaryImages.Average.proj(average='average_image') & {'animal_id': 17797} & Field)
    
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
    mask_id         :  smallint     # mask ID, unique per field
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
    
    @property
    def key_source(self):
        return meso.Fluorescence.Trace & Segmentation.segmentation_key & Field
    
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
    ms_delay            : smallint      # delay from start of frame (field 1 pixel 1) to recording of this unit (milliseconds)
    """
    
    @property
    def key_source(self):
        return (meso.ScanSet.Unit * meso.ScanSet.UnitInfo) & Segmentation.segmentation_key & Field  
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        
@schema 
class UnitHash(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Assign hash to each unique session - scan_idx - unit triplet
    -> ScanUnit
    ---
    hash                : varchar(64)                   # unique hash per unit
    str_key             : varchar(64)                   # session, scan_idx and unit_id as string
    """

    @property 
    def key_source(self):
        return ScanUnit.proj()
    
    @classmethod 
    def fill(self):
        import json
        import hashlib
        
        hash_keys = []
        for key in self.key_source:
            unicode_json = json.dumps(key).encode()
            h = hashlib.sha256(unicode_json)
            str_key = '_'.join([str(s).zfill(z) for s,z in zip(key.values(),(2,2,6))])
            hash_keys.append({**key,
                              'hash':h.hexdigest()[:12],
                              'str_key':str_key})
        self.insert(hash_keys, **params)

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
    
    activity_key = {'spike_method': 5}
    
    @property
    def key_source(self):
        return meso.Activity.Trace & self.activity_key & Segmentation.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


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
        units = (meso.AreaMembership.UnitInfo & {'ret_hash':'edec10b648420dd1dc8007b607a5046b'} & 
                 Segmentation.segmentation_key & cls.key_source).fetch('session','scan_idx',
                                                                       'brain_area','unit_id',as_dict=True)
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
        return meso.MaskClassification.Type.proj(mask_type='type') & Segmentation.segmentation_key & Scan

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
    @property
    def key_source(self):
        return tune.MovieOracle.Total & Segmentation.segmentation_key & Activity.activity_key & Field
    
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
    motor_z                    : float                  # center of volume in the motor coordinate system (microns, cortex is at 0)
    motor_y                    : float                  # center of volume in the motor coordinate system (microns)
    motor_x                    : float                  # center of volume in the motor coordinate system (microns)
    px_depth             : smallint                     # number of slices
    px_height            : smallint                     # lines per frame
    px_width             : smallint                     # pixels per line
    um_depth             : float                        # depth (microns)
    um_height            : float                        # height (microns)
    um_width             : float                        # width (microns)
    surf_z               : float                        # depth of first slice - half a z step (microns,cortex is at z=0)
    """
    
    platinum_stack = {'animal_id': 17797, 'stack_session': 9, 'stack_idx': 19}
    
    @property
    def key_source(self):
        return stack.CorrectedStack.proj(..., stack_session='session',
                          motor_z = 'z', motor_y = 'y', motor_x = 'x') & self.platinum_stack
    
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
    a11                  : float                        # row 1, column 1 of the affine matrix (microns)
    a21                  : float                        # row 2, column 1 of the affine matrix (microns)
    a31                  : float                        # row 3, column 1 of the affine matrix (microns)
    a12                  : float                        # row 1, column 2 of the affine matrix (microns)
    a22                  : float                        # row 2, column 2 of the affine matrix (microns)
    a32                  : float                        # row 3, column 2 of the affine matrix (microns)
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
    
    @property
    def key_source(self):
        return meso.StackCoordinates.UnitInfo & Segmentation.segmentation_key & Stack & Field
    
    @classmethod
    def fill(cls):
        stack_unit = (cls.key_source*Stack).proj(stack_x = 'round(stack_x - motor_x + um_width/2, 2)', 
                                                 stack_y = 'round(stack_y - motor_y + um_height/2, 2)', 
                                                 stack_z = 'round(stack_z - motor_z + um_depth/2, 2)')
        cls.insert((meso.StackCoordinates.UnitInfo.proj(motor_x='stack_x', 
                                                        motor_y='stack_y', 
                                                        motor_z='stack_z') * stack_unit), **params)

        
            
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
        return (experiment.Scan.proj() & Scan().platinum_scans) - Stimulus
    
    @classmethod
    def fill(cls):
    
        from .export_utils import resample_stim
        for key in cls.key_source:
            # fetch field0pixel1 scan times in stimulus clock
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            scan_times = (stimulus.Sync & key).fetch1('frame_times')
            if not (key['session'] == 4 and key['scan_idx'] == 9):
                scan_times = scan_times[0]
            scan_times = scan_times[:nframes*ndepths:ndepths]
            
            est_refresh_rate=60
            
            interpolated_movie,emp_refresh_rate = resample_stim(key, 
                                                                target_times=scan_times,
                                                                est_refresh_rate=est_refresh_rate,
                                                                target_size = (90,160),
                                                                resize_method='inter_area',
                                                                tol = 2e-3)
            
            # insert into table
            cls.insert1({'session' : key['session'],
                         'scan_idx': key['scan_idx'],
                         'movie'   : interpolated_movie}, **params)
    
    

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
            # scan field1_pixel0 frame time
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            scan_times = (stimulus.Sync & key).fetch1('frame_times')
            if not (key['session'] == 4 and key['scan_idx'] == 9):
                scan_times = scan_times[0]
            field1_pixel0 = scan_times[:nframes*ndepths:ndepths]
            
            
            # scan field1_pixel0 frame time relative to start of scan
            scan_start = field1_pixel0[0]
            field1_pixel0 = field1_pixel0-scan_start

            # pull dataframe of all trials, since some trial features depend on following trials
            data = ((stimulus.Trial() & key) * stimulus.Condition()).fetch(as_dict=True,order_by='trial_idx ASC')
            trial_df = pd.DataFrame(data)

            # stimulus frame times relative to start of scan
            frame_times = np.array(trial_df['flip_times'])-scan_start

            # first and last frame time of each trial
            start_times = np.array([ft.squeeze()[0] for ft in frame_times])
            end_times = np.array([ft.squeeze()[-1] for ft in frame_times])

            # median duration between end of preceding trial and start of following trial (slightly longer than one frame)
            median_intertrial_time = np.median(np.diff(np.hstack((end_times[:-1,None],start_times[1:,None])),axis=1))

            # interpolate from time to scan frame index
            t2f = interp1d(field1_pixel0,range(nframes))

            # find nearest scan frame index to first frame time of each trial
            start_frames = np.array([np.round(t2f(ft.squeeze()[0])) for ft in frame_times])

            # since the last frame of each trial persists on monitor until replaced with next trial
            # take scan frame index of following trial - 1 as end frame
            # for last trial, estimate clear time with median intertrial time
            end_frames = np.hstack((start_frames[1:]-1,np.round(t2f(end_times[-1] + median_intertrial_time))-1))

            trial_df['start_frame_time'] = start_times
            trial_df['end_frame_time'] = end_times
            trial_df['start_idx'] = start_frames
            trial_df['end_idx'] = end_frames
            trial_df['stim_times'] = frame_times
            trial_df['type'] = trial_df['stimulus_type']

            cls.insert(trial_df,**params)

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
        import io
        import imageio
        for key in cls.key_source:
            movie_mapping = {
                'poqatsi':"Cinematic",
                'MadMax':   "Cinematic",
                'naqatsi':  "Cinematic",
                'koyqatsi': "Cinematic",
                'matrixrv': "Cinematic",
                'starwars': "Cinematic",
                'matrixrl': "Cinematic",
                'matrix':   "Cinematic"}

            long_movie_mapping = {
                'poqatsi':  "Powaqqatsi: Life in Transformation (1988)",
                'MadMax':   "Mad Max: Fury Road (2015)",
                'naqatsi':  "Naqoyqatsi: Life as War (2002)",
                'koyqatsi': "Koyaanisqatsi: Life Out of Balance (1982)",
                'matrixrv': "The Matrix Revolutions (2003)",
                'starwars': "Star Wars: Episode VII - The Force Awakens (2015)",
                'matrixrl': "The Matrix Reloaded (2003)",
                'matrix':   "The Matrix (1999)"}

            short_mapping = {
                 'bigrun':'Rendered',
                 'finalrun':'Rendered',
                 'sports1m':'sports1m',
                 **movie_mapping}
            
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
    pupil_min_r          : longblob               # vector of pupil minor radii synchronized with field 1 frame times (pixels)
    pupil_maj_r          : longblob               # vector of pupil major radii synchronized with field 1 frame times (pixels)
    pupil_x              : longblob               # vector of pupil x positions synchronized with field 1 frame times (pixels)
    pupil_y              : longblob               # vector of pupil y positions synchronized with field 1 frame times (pixels)
    """

    @property 
    def key_source(cls):
        return RawManualPupil().proj(animal_id='17797') 

    @classmethod
    def fill(cls):
        for key in cls.key_source:
            from .export_utils import hamming_filter
            
            # get frame times for field1pixel0 in behavior clock
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            frame_times = np.array((stimulus.BehaviorSync & key).fetch1('frame_times')[:nframes*ndepths:ndepths])
            offset = (stimulus.BehaviorSync() & key).fetch1('frame_times')[0]

            # normalize to first frame of scan in behavior clock
            frame_times -= offset

            # fetch raw pupil info from RawManualPupil
            stored_pupilinfo = (RawManualPupil() & key).fetch1()

            # already normalized to first frame of scan in behavior clock
            pupil_times = stored_pupilinfo['pupil_times']

            # apply hamming filter to fitted parameter traces
            param_stack = np.vstack((stored_pupilinfo['pupil_x'],
                                     stored_pupilinfo['pupil_y'],
                                     stored_pupilinfo['pupil_maj_r'],
                                     stored_pupilinfo['pupil_min_r']))
            param_stack = hamming_filter(param_stack, pupil_times, frame_times, time_axis=1)

            #linearly interpolate from behavior clock times to fit parameters
            t2params = interp1d(pupil_times, param_stack, kind='linear',bounds_error=False,fill_value=np.nan)
            pupil_x,pupil_y,pupil_maj_r,pupil_min_r = t2params(frame_times)

            cls.insert1({**key,
                         'pupil_min_r':pupil_min_r,
                         'pupil_maj_r':pupil_maj_r,
                         'pupil_x':pupil_x,
                         'pupil_y':pupil_y},**params)

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
            # pull treadmill info from database
            treadmill = dj.create_virtual_module("treadmill","pipeline_treadmill")
            treadmill_info = (treadmill.Treadmill & key).fetch1()
            
            # normalize pupil times to first frame of scan in behavior clock
            frame_times_beh = (stimulus.BehaviorSync() & key).fetch1('frame_times')
            adjusted_treadmill_times = treadmill_info['treadmill_time'] - frame_times_beh[0]
            
            cls.insert1({**key,'treadmill_timestamps':adjusted_treadmill_times,
                               'treadmill_velocity':treadmill_info['treadmill_vel']},**params)


@schema
class Treadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->RawTreadmill
    ---
    treadmill_velocity      : longblob          # vector of treadmill velocities synchronized with field 1 frame times (cm/s)
    """
    
    @property
    def key_source(cls):
        return RawTreadmill().proj(animal_id='17797')
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            from .export_utils import hamming_filter

            # get frame times for field1pixel0 in behavior clock
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            frame_times = np.array((stimulus.BehaviorSync & key).fetch1('frame_times')[:nframes*ndepths:ndepths])
            offset = (stimulus.BehaviorSync() & key).fetch1('frame_times')[0]

            # normalize to first frame of scan in behavior clock
            frame_times -= offset

            # fetch raw treadmill info from RawTreadmill
            tread_time, tread_vel = (RawTreadmill() & key).fetch1('treadmill_timestamps', 'treadmill_velocity')

            # apply hamming filter to treadmill velocities
            tread_vel = hamming_filter(tread_vel, tread_time, frame_times, time_axis=0)

            #linearly interpolate from behavior clock times to fit parameters
            tread_interp = interp1d(tread_time, tread_vel, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_tread_vel = tread_interp(frame_times)
            
            cls.insert1({'session': key['session'],
                         'scan_idx': key['scan_idx'],
                         'treadmill_velocity': interp_tread_vel},**params)
        
        
        
        