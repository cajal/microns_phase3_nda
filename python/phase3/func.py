from . import utils
from . import nda

import numpy as np

import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels

phase3_ng_link = "https://akhilesh-graphene-sharded-dot-neuromancer-seung-import.appspot.com/#!%7B%22layers%22:%5B%7B%22source%22:%22precomputed://https://seungdata.princeton.edu/minnie65-phase3-em/aligned/v1%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22#uicontrol%20float%20black%20slider%28min=0%2C%20max=1%2C%20default=0.33%29%5Cn#uicontrol%20float%20white%20slider%28min=0%2C%20max=1%2C%20default=0.66%29%5Cnfloat%20rescale%28float%20value%29%20%7B%5Cn%20%20return%20%28value%20-%20black%29%20/%20%28white%20-%20black%29%3B%5Cn%7D%5Cnvoid%20main%28%29%20%7B%5Cn%20%20float%20val%20=%20toNormalized%28getDataValue%28%29%29%3B%5Cn%20%20if%20%28val%20%3C%20black%29%20%7B%5Cn%20%20%20%20emitRGB%28vec3%280%2C0%2C0%29%29%3B%5Cn%20%20%7D%20else%20if%20%28val%20%3E%20white%29%20%7B%5Cn%20%20%20%20emitRGB%28vec3%281.0%2C%201.0%2C%201.0%29%29%3B%5Cn%20%20%7D%20else%20%7B%5Cn%20%20%20%20emitGrayscale%28rescale%28val%29%29%3B%5Cn%20%20%7D%5Cn%7D%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22em-phase3%22%7D%2C%7B%22source%22:%22graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1%22%2C%22type%22:%22segmentation_with_graph%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22graphOperationMarker%22:%5B%7B%22annotations%22:%5B%5D%2C%22tags%22:%5B%5D%7D%2C%7B%22annotations%22:%5B%5D%2C%22tags%22:%5B%5D%7D%5D%2C%22pathFinder%22:%7B%22color%22:%22#ffff00%22%2C%22pathObject%22:%7B%22annotationPath%22:%7B%22annotations%22:%5B%5D%2C%22tags%22:%5B%5D%7D%2C%22hasPath%22:false%7D%7D%2C%22name%22:%22seg-phase3%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://https://s3-hpcrc.rc.princeton.edu/minnie65-phase3-ws/nuclei/v0/seg%22%2C%22type%22:%22segmentation%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22nuclear-seg-phase3%22%2C%22visible%22:false%7D%5D%2C%22navigation%22:%7B%22pose%22:%7B%22position%22:%7B%22voxelSize%22:%5B4%2C4%2C40%5D%2C%22voxelCoordinates%22:%5B227465.734375%2C187007.984375%2C19551.4921875%5D%7D%2C%22orientation%22:%5B0%2C-0.7071067690849304%2C0%2C0.7071067690849304%5D%7D%2C%22zoomFactor%22:248.06867890125633%7D%2C%22perspectiveOrientation%22:%5B-0.0012847303878515959%2C0.9988105297088623%2C0.040208619087934494%2C0.02755257673561573%5D%2C%22perspectiveZoom%22:35418.8697184842%2C%22showSlices%22:false%2C%22gpuMemoryLimit%22:2000000000%2C%22concurrentDownloads%22:128%2C%22jsonStateServer%22:%22https://globalv1.daf-apis.com/nglstate/api/v1/post%22%2C%22layout%22:%223d%22%7D"

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

    grid = utils.get_grid(field_key, desired_res=1)
    # convert grid from motor coordinates to numpy coordinates
    center_x, center_y, center_z = nda.Stack.fetch1('x', 'y', 'z')
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
    
    stack_x, stack_y, stack_z = nda.Stack.fetch1('x', 'y', 'z')
    grid = utils.get_grid(field_key, desired_res=1)
    grid = grid - np.array([stack_x, stack_y, stack_z])
    return utils.sample_grid(stack, grid).numpy()