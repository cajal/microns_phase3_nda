"""
Phase3 nda schema classes and methods
"""
import numpy as np
import datajoint as dj

# private BCM schemas
# from pipeline import experiment,meso,fuse, stack, treadmill
# from stimline import tune
# pupil = dj.create_virtual_module('pupil',"pipeline_eye")
# import scanreader
# from pipeline.utils import performance
# from stimulus import stimulus

import imageio
import io
import math
from tqdm import tqdm
from inhib.func import resize_movie,hamming_filter,get_timing_offset, make_movie
from scipy.interpolate import interp1d
import moviepy.editor as mpy
import pandas as pd
import hashlib
import json


# schema = dj.schema('21617_release_nda', create_tables=True)
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
    
    inhib_scans  =  [{'animal_id': 21617, 'session': 7, 'scan_idx': 5},
                     {'animal_id': 21617, 'session': 7, 'scan_idx': 8},
                     {'animal_id': 21617, 'session': 7, 'scan_idx': 9},
                     {'animal_id': 21617, 'session': 8, 'scan_idx': 5},
                     {'animal_id': 21617, 'session': 8, 'scan_idx': 8},
                     {'animal_id': 21617, 'session': 8, 'scan_idx': 11},
                     {'animal_id': 21617, 'session': 9, 'scan_idx': 5},
                     {'animal_id': 21617, 'session': 9, 'scan_idx': 7}] #,
#                      {'animal_id': 21617, 'session': 9, 'scan_idx': 8}]  # last scan is frame scan
        
    @property
    def key_source(self):
        return (meso.experiment.Scan * meso.ScanInfo).proj('filename', 'nfields', 'nframes', 'fps') & self.inhib_scans
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)
        
        

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
    def key_source(self):
        return meso.ScanInfo.Field.proj('px_width', 'px_height', 'um_width', 'um_height', 
                                         field_x= 'x', field_y = 'y', field_z = 'z') \
                & {'animal_id': 21617} & Scan
    
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
        return meso.RasterCorrection & {'animal_id':21617} & Scan
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        

@schema
class MotionCorrection(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Raster Correction applied to each field
    -> Field
    ---
    y_shifts             : longblob                     # y motion correction shifts (pixels) 
    x_shifts             : longblob                     # x motion correction shifts (pixels) 
    y_std                : float                        # standard deviation of y shifts (um)
    x_std                : float                        # standard deviation of x shifts (um)
    """
    
    @property
    def key_source(self):
        return meso.MotionCorrection & {'animal_id':21617} & Scan
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
        return Scan().inhib_scans
    
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
        return meso.Quality.MeanIntensity & {'animal_id': 21617} & Field
    
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
        return (meso.SummaryImages.Correlation.proj(correlation='correlation_image') * 
                meso.SummaryImages.Average.proj(average='average_image') & {'animal_id': 21617} & Field)
    
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
    
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6}
    
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
    
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6}
    
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
    ms_delay            : smallint      # delay from start of frame (field 1 pixel 1) to recording of this unit (milliseconds)
    """
    
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return (meso.ScanSet.Unit * meso.ScanSet.UnitInfo) & self.segmentation_key & Field  
    
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
    """

    @property 
    def key_source(self):
        return ScanUnit.proj()
    
    @classmethod 
    def fill(self):
        hash_keys = []
        for key in self.key_source:
            unicode_json = json.dumps(key).encode()
            h = hashlib.sha256(unicode_json)
            hash_keys.append({**key,'hash':h.hexdigest()})
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
    
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return meso.Activity.Trace & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


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
        return meso.MaskClassification.Type.proj(mask_type='type') & {'animal_id': 21617, 'segmentation_method': 6} & Scan

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
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return tune.MovieOracle.Total & self.segmentation_key & Field
    
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
    stack_session        : smallint                     # session ID
    stack_idx            : smallint                     # stack ID
    ---
    motor_z                    : float                  # center of volume in the motor coordinate system (microns, cortex at z=0)
    motor_y                    : float                  # center of volume in the motor coordinate system (microns)
    motor_x                    : float                  # center of volume in the motor coordinate system (microns)
    px_depth             : smallint                     # number of slices
    px_height            : smallint                     # lines per frame
    px_width             : smallint                     # pixels per line
    um_depth             : float                        # depth (microns)
    um_height            : float                        # height (microns)
    um_width             : float                        # width (microns)
    surf_z               : float                        # depth of first slice - half a z step (microns, cortex is at z=0)
    """
    
    inhib_stack = {'animal_id': 21617, 'stack_session': 9, 'stack_idx': 11}
    
    @property
    def key_source(self):
        return stack.CorrectedStack.proj(...,stack_session = 'session',
                          motor_z = 'z', motor_y = 'y', motor_x = 'x') & self.inhib_stack

    
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
        return stack.Registration.Affine.proj(..., session='scan_session') & {'animal_id': 21617} & Stack & Field
    
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
    
    segmentation_key = {'animal_id': 21617, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.StackCoordinates.UnitInfo & self.segmentation_key & Stack & Field
    
    @classmethod
    def fill(cls):
        stack_unit = (cls.key_source * Stack).proj(stack_x = 'round(stack_x - motor_x + um_width/2, 2)', 
                                                   stack_y = 'round(stack_y - motor_y + um_height/2, 2)', 
                                                   stack_z = 'round(stack_z - motor_z + um_depth/2, 2)')

        cls.insert((meso.StackCoordinates.UnitInfo.proj(motor_x='stack_x', 
                                                        motor_y='stack_y', 
                                                        motor_z='stack_z') * stack_unit), **params)
        
# @schema
# class Stimulus(dj.Manual):
#     """
#     Class methods not available outside of BCM pipeline environment
#     """
#     definition = """
#     # Stimulus presented
#     -> Scan
#     ---
#     movie                : longblob                     # stimulus images synchronized with field 1 frame times (H x W X F matrix)
#     """
#     @property 
#     def key_source(cls):
#         return Scan().inhib_scans
    
#     @classmethod
#     def fill(cls):
#         for key in cls.key_source:
#             time_axis = 2
#             target_size = (90, 160)
#             full_stimulus = None
#             full_flips = None

#             num_depths = np.unique((meso.ScanInfo.Field & key).fetch('z')).shape[0]
#             scan_times = (stimulus.Sync & key).fetch1('frame_times').squeeze()[::num_depths]
#             trial_data = ((stimulus.Trial & key) * stimulus.Condition).fetch('KEY', 'stimulus_type', order_by='trial_idx ASC')
#             for trial_key,stim_type in zip(tqdm(trial_data[0]), trial_data[1]):
                
#                 if stim_type == 'stimulus.Clip':
#                     djtable = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
#                     flip_times, compressed_clip, skip_time, cut_after,frame_rate = djtable.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
#                     # convert to grayscale and stack to movie in width x height x time
#                     temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
#                     # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
#                     temp_vid_length = temp_vid.count_frames()
#                     movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)
#                     assumed_clip_fps = frame_rate
#                     start_idx = int(np.float(skip_time) * assumed_clip_fps)
# #                     print(trial_key)
#                     end_idx = int(start_idx + (np.float(cut_after) * assumed_clip_fps))
                
#                     movie = movie[:,:,start_idx:end_idx]
#                     movie = resize_movie(movie, target_size, time_axis)
#                     movie = hamming_filter(movie, time_axis, flip_times, scan_times)
#                     full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#                     full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
#                 elif stim_type == 'stimulus.Monet2':
#                     flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('flip_times', 'movie')
#                     movie = movie[:,:,0,:]
#                     movie = resize_movie(movie, target_size, time_axis)
#                     movie = hamming_filter(movie, time_axis, flip_times, scan_times)
#                     full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#                     full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
#                 elif stim_type == 'stimulus.Trippy':
#                     flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('flip_times', 'movie')
#                     movie = resize_movie(movie, target_size, time_axis)
#                     movie = hamming_filter(movie, time_axis, flip_times, scan_times)
#                     full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
#                     full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
#                 else:
#                     raise Exception(f'Error: stimulus type {stim_type} not understood')

#             h,w,t = full_stimulus.shape
#             interpolated_movie = np.zeros((h, w, scan_times.shape[0]))
#             for t_time,i in zip(tqdm(scan_times), range(len(scan_times))):
#                 idx = (full_flips < t_time).sum() - 1
#                 if (idx < 0) or (idx >= full_stimulus.shape[2]-2):
#                     interpolated_movie[:,:,i] = np.zeros(full_stimulus.shape[0:2])
#                 else:
#                     myinterp = interp1d(full_flips[idx:idx+2], full_stimulus[:,:,idx:idx+2], axis=2)
#                     interpolated_movie[:,:,i] = myinterp(t_time)
            

#             overflow = np.where(interpolated_movie > 255)
#             underflow = np.where(interpolated_movie < 0)
#             interpolated_movie[overflow[0],overflow[1],overflow[2]] = 255
#             interpolated_movie[underflow[0],underflow[1],underflow[2]] = 0
#             cls.insert1({'movie':interpolated_movie.astype(np.uint8),'session':key['session'],'scan_idx':key['scan_idx']},**params)
        

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
        return Scan().inhib_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            time_axis = 2
            target_size = (90, 160)

            # fetch field0pixel1 scan times in stimulus clock
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            scan_times = (stimulus.Sync & key).fetch1('frame_times')[:nframes*ndepths:ndepths]

            # get all trial keys and corresponding stimulus types
            trial_rel = ((stimulus.Trial & key) * stimulus.Condition)
            trial_keys,stim_types = trial_rel.fetch('KEY','stimulus_type',order_by='trial_idx ASC')

            # iterate through all trials in order and append movie and flip_times
            full_stimulus = None
            full_flips = None
            for trial_key,stim_type in zip(tqdm(trial_keys), stim_types):
                cond_rel = (stimulus.Trial & trial_key) * stimulus.Condition

                if stim_type == 'stimulus.Clip':
                    cond_rel = (cond_rel * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
                    flip_times, compressed_clip, skip_time, cut_after,frame_rate = cond_rel.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
                    temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
                    # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
                    temp_vid_length = temp_vid.count_frames()
                    # convert to grayscale and stack to movie in width x height x time
                    movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)

                    # slice frames between start time and start time + cut after
                    start_idx = int(np.float(skip_time) * frame_rate)
                    end_idx = int(start_idx + (np.float(cut_after) * frame_rate))
                    movie = movie[:,:,start_idx:end_idx]

                elif stim_type == 'stimulus.Monet2':
                    cond_rel = (cond_rel * stimulus.Monet2)
                    flip_times, movie, bg_sat = cond_rel.fetch1('flip_times', 'movie','blue_green_saturation')
                    assert bg_sat == 0.0, 'not configured for dual channel Monet2'
                    movie = movie[:,:,0,:]

                elif stim_type == 'stimulus.Trippy':
                    cond_rel = (cond_rel * stimulus.Trippy)
                    flip_times, movie = cond_rel.fetch1('flip_times', 'movie')

                else:
                    raise Exception(f'Error: stimulus type {stim_type} not understood')

                # resize to target pixel resolution
                movie = resize_movie(movie, target_size, time_axis)

                # apply hamming filter before temporal resample
                movie = hamming_filter(movie, time_axis, flip_times, scan_times)

                # append stimulus and flips
                full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()


            # identify scan frames when stimulus was shown
            stim_scan_idx = np.logical_and(scan_times>=full_flips[0],scan_times<=full_flips[-1])

            # time to stimulus linear interpolation object
            t2s = interp1d(full_flips,full_stimulus,axis=2,kind='linear')

            # linearly interpolate frames for duration of stimulus
            interpolated_movie = np.zeros((*target_size, len(scan_times)))
            interpolated_movie[:,:,stim_scan_idx] = t2s(scan_times[stim_scan_idx])

            # clip overflow/underflow pixels introduced by hamming filter before recast to uint8
            interpolated_movie = np.clip(interpolated_movie,0,255).astype(np.uint8)

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
        return Scan().inhib_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
#             data = ((stimulus.Trial() & key) * stimulus.Condition()).fetch(as_dict=True)
#             ndepths = len(dj.U('z') & (meso.ScanInfo().Field() & key))

#             frame_times = (stimulus.Sync() & key).fetch1('frame_times')
#             field1_pixel0 = frame_times[::ndepths]
            
#             offset = get_timing_offset(key)
#             field1_pixel0 = field1_pixel0 - offset 


#             for idx,trial in enumerate(tqdm(data)):
#                 trial_flip_times = trial['flip_times'].squeeze() - offset
                
#                 start_index = (field1_pixel0 < trial_flip_times[0]).sum() - 1
#                 end_index = (field1_pixel0 < trial_flip_times[-1]).sum() - 1
#                 start_time = trial_flip_times[0]
#                 end_time = trial_flip_times[-1]
#                 if(idx > 0):
#                     if(data[idx-1]['end_idx'] == start_index):
#                         print('ding')
#                         t0 = data[idx-1]
#                         med = start_time
#                         nearest_frame_start = np.argmin(np.abs(field1_pixel0 - med))
#                         data[idx-1]['end_idx'] = nearest_frame_start-1
#                         start_index = nearest_frame_start
                
#                 data[idx]['start_frame_time'] = start_time  
#                 data[idx]['end_frame_time'] = end_time  
#                 data[idx]['start_idx'] = start_index
#                 data[idx]['end_idx'] = end_index
#                 data[idx]['stim_times'] = trial_flip_times
#                 data[idx]['type'] = data[idx]['stimulus_type']
                        
#             cls.insert(data,**params)
            
            # scan field1_pixel0 frame time
            ndepths = (ScanTimes & key).fetch1('ndepths')
            nframes = (Scan & key).fetch1('nframes')
            field1_pixel0 = (stimulus.Sync & key).fetch1('frame_times')[:nframes*ndepths:ndepths]

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
    fps                  : float                        # original framerate of clip (Hz)
    """

    @property 
    def key_source(cls):
        return (experiment.Scan.proj() & Scan().inhib_scans) & (stimulus.Trial * stimulus.Clip)
    
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
    fps                  : decimal(6,3)                 # display refresh rate (Hz)
    duration             : decimal(6,3)                 # trial duration (seconds)
    rng_seed             : double                       # random number generator seed
    blue_green_saturation=0.000 : decimal(4,3)          # 0 = grayscale, 1=blue/green
    pattern_width        : smallint                     # width of generated pattern (pixels)
    pattern_aspect       : float                        # the aspect ratio of generated pattern
    temp_kernel          : varchar(16)                  # temporal kernel type (hamming, half-hamming)
    temp_bandwidth       : decimal(4,2)                 # temporal bandwidth of the stimulus (Hz)
    ori_coherence        : decimal(4,2)                 # 1=unoriented noise. pi/ori_coherence = bandwidth of orientation kernel.
    ori_fraction         : float                        # fraction of stimulus with coherent orientation vs unoriented
    ori_mix              : float                        # mixin-coefficient of orientation biased noise
    n_dirs               : smallint                     # number of directions
    speed                : float                        # (units/s) motion component, where unit is display width
    directions           : longblob                     # computed directions of motion (deg)
    onsets               : blob                         # computed direction onset (seconds)
    movie                : longblob                     # rendered uint8 movie (H X W X 1 X T)
    """
    @property 
    def key_source(cls):
        return (stimulus.Trial().proj('condition_hash') * stimulus.Monet2 & Scan().inhib_scans) 

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
    tex_ydim             : smallint                     # texture height (pixels)
    tex_xdim             : smallint                     # texture width (pixels)
    duration             : float                        # trial duration (seconds)
    xnodes               : tinyint                      # x dimension of low-res phase movie
    ynodes               : tinyint                      # y dimension of low-res phase movie
    up_factor            : tinyint                      # spatial upscale factor
    temp_freq            : float                        # temporal frequency if the phase pattern were static (Hz)
    temp_kernel_length   : smallint                     # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
    spatial_freq         : float                        # approximate max. The actual frequencies may be higher. (cy/point)
    movie                : longblob                     # rendered movie (H X W X T)
    """
    @property 
    def key_source(cls):
        return (stimulus.Trial().proj('condition_hash') * stimulus.Trippy & Scan().inhib_scans) 

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source,**params)

# @schema
# class Frame(dj.Manual):
#     definition = """
#     # static image condition
#     condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
#     ---
#     image_id             : int                          # image_id 
#     imagenet_id          : varchar(25)                  # id from imagenet dataset
#     pre_blank_period     : float                        # (s) duration of preblanking period
#     presentation_time    : float                        # (s) image duration
#     frame_width          : int                          # pixels
#     frame_height         : int                          # pixels
#     description          : varchar(255)                 # image content, from imagenet dataset
#     image                : longblob                     # image (h x w)
#     """

#     @property 
#     def key_source(cls):
#         return ((stimulus.Trial.proj('condition_hash') & Scan.inhib_scans) * 
#                             stimulus.Frame * stimulus.StaticImage * 
#                             stimulus.StaticImage.Image * stimulus.StaticImage.ImageNet)
        
#     @classmethod
#     def fill(cls):
#         cls.insert(cls.key_source,**params)
        
    
@schema
class RawDLCPupil(dj.Manual):
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
        return Scan().inhib_scans

    @classmethod
    def fill(cls):
        for key in cls.key_source:

            pupil_info = (pupil.FittedPupil.Ellipse & key & 'tracking_method = 2').fetch(order_by='frame_id ASC')
            raw_maj_r,raw_min_r = pupil_info['major_radius'],pupil_info['minor_radius']
            raw_pupil_x = np.array([np.nan if entry is None else entry[0] for entry in pupil_info['center']])
            raw_pupil_y = np.array([np.nan if entry is None else entry[1] for entry in pupil_info['center']])
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
class DLCPupil(dj.Manual):
    definition = """
    # Pupil traces
    -> RawDLCPupil
    ---
    pupil_min_r          : longblob                     # vector of pupil minor radii synchronized with field 1 frame times (pixels)
    pupil_maj_r          : longblob                     # vector of pupil major radii synchronized with field 1 frame times (pixels)
    pupil_x              : longblob                     # vector of pupil x positions synchronized with field 1 frame times (pixels)
    pupil_y              : longblob                     # vector of pupil y positions synchronized with field 1 frame times (pixels)
    """

    @property 
    def key_source(cls):
        return RawDLCPupil().proj(animal_id='21617') 

    @classmethod
    def fill(cls):
        for key in cls.key_source:
            stored_pupilinfo = (RawDLCPupil() & key).fetch1()
            pupil_times = stored_pupilinfo['pupil_times']
            frame_times,ndepths = (ScanTimes()  & key).fetch1('frame_times','ndepths')
            top_frame_scan_times_beh_clock = frame_times[::ndepths]
            
            ## small note about 4-9: the scan was stopped prematurely, so the length needs to be corrected as a result.
            if((key['session'] == 4) and (key['scan_idx']==9)):
                top_frame_scan_times_beh_clock = top_frame_scan_times_beh_clock[:-1]
            
            raw_pupil_x = np.array([np.nan if entry is None else entry for entry in stored_pupilinfo['pupil_x']])
            raw_pupil_y = np.array([np.nan if entry is None else entry for entry in stored_pupilinfo['pupil_y']])
            pupil_x_interp = interp1d(pupil_times, raw_pupil_x, kind='linear', bounds_error=False, fill_value=np.nan)
            pupil_y_interp = interp1d(pupil_times, raw_pupil_y, kind='linear', bounds_error=False, fill_value=np.nan)
            major_r_interp = interp1d(pupil_times, stored_pupilinfo['pupil_maj_r'], kind='linear', bounds_error=False, fill_value=np.nan)
            minor_r_interp = interp1d(pupil_times,stored_pupilinfo['pupil_min_r'],kind='linear',bounds_error=False,fill_value=np.nan)
            pupil_x = pupil_x_interp(top_frame_scan_times_beh_clock)
            pupil_y = pupil_y_interp(top_frame_scan_times_beh_clock)
            pupil_maj_r = major_r_interp(top_frame_scan_times_beh_clock)
            pupil_min_r = minor_r_interp(top_frame_scan_times_beh_clock)
            cls.insert1({**key,'pupil_min_r':pupil_min_r,'pupil_maj_r':pupil_maj_r,
                               'pupil_x':pupil_x,'pupil_y':pupil_y},**params)


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
        return Scan().inhib_scans
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            treadmill = dj.create_virtual_module("treadmill","pipeline_treadmill")
            treadmill_info = (treadmill.Treadmill & key).fetch1()
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
    treadmill_velocity      : longblob                     # vector of treadmill velocities synchronized with field 1 frame times (cm/s)
    """
    
    @property
    def key_source(cls):
        return RawTreadmill().proj(animal_id='21617')
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            tread_time, tread_vel = (RawTreadmill() & key).fetch1('treadmill_timestamps', 'treadmill_velocity')
            tread_interp = interp1d(tread_time, tread_vel, kind='linear', bounds_error=False, fill_value=np.nan)
            tread_time = tread_time.astype(np.float)
            tread_vel = tread_vel.astype(np.float)
            frame_times,ndepths = (ScanTimes()  & key).fetch1('frame_times','ndepths')
            if((key['session'] == 4) and (key['scan_idx'] == 9)):
                top_frame_time = frame_times[::ndepths][:-1]
            else:
                top_frame_time = frame_times[::ndepths]
        
            interp_tread_vel = tread_interp(top_frame_time)
            treadmill_key = {
                'session': key['session'],
                'scan_idx': key['scan_idx'],
                'treadmill_velocity': interp_tread_vel
            }
            cls.insert1(treadmill_key,**params)

            
def generate_stack(key,filename):
    """
    Grabs stack from the database and saves in filename.tif
    Args:
        key         dictionary      Stack to generate
        filename    string          filename to save tiff file to    
    
    """
    from skimage.external.tifffile import imsave

    # Create a composite interleaving channels
    height, width, depth = (stack.CorrectedStack() & key).fetch1('px_height', 'px_width', 'px_depth')
    num_channels = (stack.StackInfo() & (stack.CorrectedStack() & key)).fetch1('nchannels')
    composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
    for i in range(num_channels):
        composite[i::num_channels] = (stack.CorrectedStack() & key).get_stack(i + 1)

    imsave(filename,composite)
    
def generate_resized_stack(key,filename):
    """
    Grabs stack from the database and saves in filename.tif
    Args:
        key         dictionary      Stack to generate
        filename    string          filename to save tiff file to    
    
    """
    from skimage.external.tifffile import imsave

    # Create a composite interleaving channels
    height, width, depth = (stack.CorrectedStack() & key).fetch1('um_height', 'um_width', 'um_depth')
    height, width, depth = [int(np.round(c)) for c in (height,width,depth)]
    num_channels = (stack.StackInfo() & (stack.CorrectedStack() & key)).fetch1('nchannels')
    composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
    for i in range(num_channels):
        composite[i::num_channels] = (stack.PreprocessedStack & key & f'channel = {i+1}').fetch1('resized')

    imsave(filename,composite)
    
    
def generate_stimulus_avi(path,key):
    """
    Generates an avi of the stimulus from frames stored in the database 
    Args:
        key:dict Scan to generate the avis from 
    
    Returns:
        None, writes stimulus AVI locally
    
    
    """
    time_axis = 2
    target_size = (90, 160)
    full_stimulus = None
    full_flips = None

    key['animal_id'] = 21617
    print(key)
    scan_times = (stimulus.Sync & key).fetch1('frame_times').squeeze()
    target_hz = 30
    trial_data = ((stimulus.Trial & key) * stimulus.Condition).fetch('KEY', 'stimulus_type', order_by='trial_idx ASC')
    for trial_key,stim_type in zip(tqdm(trial_data[0]), trial_data[1]):
        
        if stim_type == 'stimulus.Clip':
            djtable = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
            flip_times, compressed_clip, skip_time, cut_after,frame_rate = djtable.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
            flip_times = flip_times[0]
            # convert to grayscale and stack to movie in width x height x time
            temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
            # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
            temp_vid_length = temp_vid.count_frames()
            movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)
            assumed_clip_fps = frame_rate
            start_idx = int(np.float(skip_time) * assumed_clip_fps)
            
            (trial_key)
            end_idx = int(start_idx + (np.float(cut_after) * assumed_clip_fps))
            times = np.linspace(flip_times[0],flip_times[-1], int((flip_times[-1] - flip_times[0])*target_hz))
            movie = movie[:,:,start_idx:end_idx]
            movie = resize_movie(movie, target_size, time_axis)
            movie = hamming_filter(movie, time_axis, flip_times, times)
            full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
            full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
        elif stim_type == 'stimulus.Monet2':
            flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('flip_times', 'movie')
            flip_times = flip_times[0]
            movie = movie[:,:,0,:]  
            movie = resize_movie(movie, target_size, time_axis)
            times = np.linspace(flip_times[0],flip_times[-1],int((flip_times[-1] - flip_times[0])*target_hz))
            movie = hamming_filter(movie, time_axis, flip_times, times)
            full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
            full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
        elif stim_type == 'stimulus.Trippy':
            flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('flip_times', 'movie')
            flip_times = flip_times[0]
            movie = resize_movie(movie, target_size, time_axis)
            times = np.linspace(flip_times[0],flip_times[-1],int((flip_times[-1] - flip_times[0])*target_hz))
            movie = hamming_filter(movie, time_axis, flip_times, times)
            full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
            full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
        
        else:
            raise Exception(f'Error: stimulus type {stim_type} not understood')

    pre_blank_length = full_flips[0] - scan_times[0]
    post_blank_length = scan_times[-1] - full_flips[-1]
    pre_nframes = np.ceil(pre_blank_length*target_hz)
    h,w,t = full_stimulus.shape
    times = np.linspace(full_flips[0],full_flips[-1],int(np.ceil((full_flips[-1] - full_flips[0])*target_hz)))

    interpolated_movie = np.zeros((h, w, int(np.ceil(scan_times[-1] - scan_times[0])*target_hz)))

    for t_time,i in zip(tqdm(times), range(len(times))):
        idx = (full_flips < t_time).sum() - 1
        if(idx < 0):
            continue
        myinterp = interp1d(full_flips[idx:idx+2], full_stimulus[:,:,idx:idx+2], axis=2)
        interpolated_movie[:,:,int(i+pre_nframes)] = myinterp(t_time)

        # NOTE : For compressed version, use the default settings associated with mp4 (libx264)
        #        For the lossless version, use PNG codec and .avi output
    
    
    overflow = np.where(interpolated_movie > 255)
    underflow = np.where(interpolated_movie < 0)
    interpolated_movie[overflow[0],overflow[1],overflow[2]] = 255
    interpolated_movie[underflow[0],underflow[1],underflow[2]] = 0
    f = lambda t: make_movie(t,interpolated_movie,target_hz)
    clip = mpy.VideoClip(f, duration=(interpolated_movie.shape[-1]/target_hz))

    clip.write_videofile(path + f"stimulus_21617_{key['session']}_{key['scan_idx']}_v1_compressed.mp4", codec='libx264', fps=target_hz)
    clip.write_videofile(path + f"stimulus_21617_{key['session']}_{key['scan_idx']}_v1.avi", fps=target_hz, codec='png')