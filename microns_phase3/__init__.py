from .version import __version__

import datajoint as dj

dj.config['enable_python_native_blobs'] = True
dj.errors._switch_filepath_types(True)
dj.errors._switch_adapted_types(True)

# try:
#     import coregister.solve as cs
# except ModuleNotFoundError:
#     raise ModuleNotFoundError('Coregistration package missing. Run "pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3"') 