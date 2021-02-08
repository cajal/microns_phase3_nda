"""
Phase3 nda schema classes and methods
"""
import numpy as np
import datajoint as dj

schema = dj.schema('microns_phase3_nda')

import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels

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
    filename             : varchar(255)                 # Scan base filename uploaded to S3
    nframes              : int                          # frames recorded
    nfields              : tinyint                      # number of fields
    px_width             : smallint                     # field pixels per line
    px_height            : smallint                     # lines per field
    um_width             : float                        # field width (microns)
    um_height            : float                        # field height (microns)
    fps                  : float                        # frames per second (Hz)
    """
    
    platinum_scans = [
        {'animal_id': 17797, 'session': 4, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 4, 'scan_idx': 9},
        {'animal_id': 17797, 'session': 4, 'scan_idx': 10},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 3},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 5},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 6},
        {'animal_id': 17797, 'session': 5, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 2},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 6},
        {'animal_id': 17797, 'session': 6, 'scan_idx': 7},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 3},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 4},
        {'animal_id': 17797, 'session': 7, 'scan_idx': 5},
        {'animal_id': 17797, 'session': 8, 'scan_idx': 4},
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
    field_x              : float                        # field x from registration into stack (microns)
    field_y              : float                        # field y from registration into stack (microns)
    field_z              : float                        # field z from registration into stack (microns)
    """
      
    @property
    def key_source(self):
        return meso.ScanInfo.Field.proj('px_width', 'px_height', 'um_width', 'um_height', field_x= 'x', field_y = 'y', field_z = 'z') \
                & {'animal_id': 17797} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


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
    z                    : float                        # (um) center of volume in the motor coordinate system (cortex is at 0)
    y                    : float                        # (um) center of volume in the motor coordinate system
    x                    : float                        # (um) center of volume in the motor coordinate system
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
        cls.insert(cls.key_source, ignore_extra_fields=True)


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
    reg_x                : float                        # (um) element in row 1, column 4 of the affine matrix
    reg_y                : float                        # (um) element in row 2, column 4 of the affine matrix
    reg_z                : float                        # (um) element in row 3, column 4 of the affine matrix
    score                : float                        # cross-correlation score (-1 to 1)
    reg_field            : longblob                     # extracted field from the stack in the specified position
    """
    
    @property
    def key_source(self):
        return stack.Registration.Affine.proj(..., session='scan_session') & {'animal_id': 17797} & Stack & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


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
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class Fluorescence(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # fluorescence traces before spike extraction or filtering
    -> Segmentation
    ---
    trace                   : longblob
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.Fluorescence.Trace & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


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
    um_x                : smallint      # x-coordinate of centroid in motor coordinate system
    um_y                : smallint      # y-coordinate of centroid in motor coordinate system
    um_z                : smallint      # z-coordinate of mask relative to surface of the cortex
    px_x                : smallint      # x-coordinate of centroid in the frame
    px_y                : smallint      # y-coordinate of centroid in the frame
    ms_delay            : smallint      # (ms) delay from start of frame to recording of this unit
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return (meso.ScanSet.Unit * meso.ScanSet.UnitInfo) & self.segmentation_key & Field  
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class Activity(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # activity inferred from fluorescence traces
    -> ScanUnit
    ---
    trace                   : longblob
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return meso.Activity.Trace & self.segmentation_key & nda3.Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class Oracle(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Leave-one-out correlation for repeated videos in stimulus.
    -> nda.ScanUnit
    ---
    trials               : int                          # number of trials used
    pearson              : float                        # per unit oracle pearson correlation over all movies
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return tune.MovieOracle.Total & self.segmentation_key & nda.Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)

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
    motor_x         : float    # x coordinate of unit_id in motor/ stack coordinates
    motor_y         : float    # y coordinate of unit_id in motor/ stack coordinates
    motor_z         : float    # z coordinate of unit_id in motor/ stack coordinates
    np_x            : float    # x coordinate of unit_id in numpy / stack coordinates
    np_y            : float    # y coordinate of unit_id in numpy / stack coordinates
    np_z            : float    # z coordinate of unit_id in numpy / stack coordinates
    """
    
    segmentation_key = {'animal_id': 17797, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return meso.StackCoordinates.UnitInfo & self.segmentation_key & Stack & Field
    
    @classmethod
    def fill(cls):
        stack_unit_np = (cls.key_source*Stack).proj(np_x = 'round(stack_x - x + um_width/2, 2)', np_y = 'round(stack_y - y + um_height/2, 2)', np_z = 'round(stack_z - z + um_depth/2, 2)')
        cls.insert((cls.key_source.proj(motor_x='stack_x', motor_y='stack_y', motor_z='stack_z') * stack_unit_np), ignore_extra_fields=True)


@schema
class Pupil(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Pupil traces
    -> Scan
    ---
    pupil_min_r          : longblob                     # vector of pupil minor radii synchronized with field 1 frame times (pixels)
    pupil_maj_r          : longblob                     # vector of pupil major radii synchronized with field 1 frame times (pixels)
    pupil_x              : longblob                     # vector of pupil x positions synchronized with field 1 frame times (pixels)
    pupil_y              : longblob                     # vector of pupil y positions synchronized with field 1 frame times (pixels)
    """
    
    @property
    def key_source(self):
        return nda3.Pupil()
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class Treadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->Scan
    ---
    treadmill_speed      : longblob                     # vector of treadmill velocities synchronized with field 1 frame times (cm/s)
    """
    
    @property
    def key_source(self):
        return nda3.Treadmill()
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class Stimulus(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Stimulus presented
    -> Scan
    ---
    movie                : longblob                     # stimulus images synchronized with field 1 frame times (H x W x T matrix)
    """
    
    class Trial(dj.Part):
        definition = """
        # Information for each Trial
        -> master
        trial_idx    :   smallint      # index of trial within stimulus
        ---
        type         :   varchar(16)   # type of stimulus trial
        start_idx    :   int unsigned      # start frame of trial
        end_idx      :   int unsigned     # end frame of trial
        condition_hash    : char(20)   # 120-bit hash (The first 20 chars of MD5 in base64)
        """

@schema
class Clip(dj.Manual):
    definition = """
    # Movie clip condition
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    movie_name           : char(8)                      # short movie title
    clip_number          : int                          # clip index
    skip_time=0.000      : decimal(7,3)                 # (s) skip to this time in the clip
    cut_after            : decimal(7,3)                 # (s) cut clip if it is longer than this duration
    """

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
    pattern_width        : smallint                     # pixel size of the resulting pattern
    pattern_aspect       : float                        # the aspect ratio of the pattern
    temp_kernel          : varchar(16)                  # 
    temp_bandwidth       : decimal(4,2)                 # (Hz) temporal bandwidth of the stimulus
    ori_coherence        : decimal(4,2)                 # 1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
    ori_fraction         : float                        # fraction of time coherent orientation is on
    ori_mix              : float                        # mixin-coefficient of orientation biased noise
    n_dirs               : smallint                     # number of directions
    speed                : float                        # (units/s)  where unit is display width
    directions           : longblob                     # computed directions of motion in degrees
    onsets               : blob                         # (s) computed
    movie                : longblob                     # (computed) uint8 movie
    """

@schema
class Trippy(dj.Manual):
    definition = """
    # randomized curvy dynamic gratings
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    fps                  : decimal(6,3)                 # monitor rate
    rng_seed             : double                       # random number generate seed
    packed_phase_movie   : longblob                     # phase movie before spatial and temporal interpolation
    tex_ydim             : smallint                     # (pixels) texture dimension
    tex_xdim             : smallint                     # (pixels) texture dimension
    duration             : float                        # (s) trial duration
    xnodes               : tinyint                      # x dimension of low-res phase movie
    ynodes               : tinyint                      # y dimension of low-res phase movie
    up_factor            : tinyint                      # spatial upscale factor
    temp_freq            : float                        # (Hz) temporal frequency if the phase pattern were static
    temp_kernel_length   : smallint                     # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
    spatial_freq         : float                        # (cy/point) approximate max. The actual frequencies may be higher.
    movie                : longblob                     # rendered movie
    """

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
        cls.insert(cls.key_source, ignore_extra_fields=True)


@schema
class SummaryImages(dj.Manual):
    definition = """
    ->Field
    ---
    correlation    : longblob                     # 
    l6norm         : longblob                     # 
    average        : longblob                     # 
    """
    
    @property
    def key_source(self):
        return meso.SummaryImages.Correlation.proj(correlation='correlation_image') * meso.SummaryImages.L6Norm.proj(l6norm='l6norm_image') * meso.SummaryImages.Average.proj(average='average_image') & {'animal_id': 17797} & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, ignore_extra_fields=True)