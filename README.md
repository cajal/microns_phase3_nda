# microns_phase3_nda

This repository contains the Python functions and utilities required to interact with the functional data for the MICrONS project. The database containing this functional data is called `nda` (short for neural data access).

For more on the MICrONS project please see: [MICrONS Explorer](https://www.microns-explorer.org/)

The current version of this repository and database is v8 (Semantic version: 0.8.0).

## Interactive Environment

Here are some options that provide a great experience:

- Cloud-based Environment (*recommended*)
  - Fork this repository, which will allow you to save your updates to the tutorials and codebase.
  - Launch using [GitHub Codespaces](https://github.com/features/codespaces) on your fork with the default options by selecting the green `Code` button and then the green `Create codespace on main` button. For more control, under the green `Code` button select the `...` button where you may create `New with options...`.
  - Build time for a codespace is **~5m**. This is done infrequently and cached for convenience.
  - Start time for a codespace is **~30s**. This will pull the built codespace from cache when you need it.
  - You will know your environment has finished loading once the `pip install -e .` command has run and the terminal prompt is clear.
  - We recommend you start by navigating to the `tutorial_notebooks` directory on the left panel and go through the `Using_DataJoint_to_Access_Functional_Data.ipynb` Jupyter notebook.
  - Once you are done, we recommend selecting the `Codespaces` menu in the bottom-left corner and then the `Stop Current Codespace` option. By default, GitHub will automatically stop the Codespace after 30 minutes of inactivity.  Once the Codespace is no longer being used, we recommend deleting the Codespace.
  - *Tip*: Each month, GitHub renews a [free-tier](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#monthly-included-storage-and-core-hours-for-personal-accounts) quota of compute and storage. Typically we run into the storage limits before anything else since Codespaces consume storage while stopped. It is best to delete Codespaces when not actively in use and recreate when needed. We'll soon be creating prebuilds to avoid larger build times. Once any portion of your quota is reached, you will need to wait for it to be reset at the end of your cycle or add billing info to your GitHub account to handle overages.
  - *Tip*: GitHub auto names the codespace but you can rename the codespace so that it is easier to identify later.

- Local Environment
  - Fork this repository
  - Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  - Install [Docker](https://docs.docker.com/get-docker/)
  - Install [VSCode](https://code.visualstudio.com/)
  - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  - `git clone` your fork of the repository and open it in VSCode
  - Use the `Dev Containers extension` to `Reopen in Container` (More info in the `Getting started` included with the extension)
  - You will know your environment has finished loading once the `pip install -e .` command has run and the terminal prompt is clear.
  - We recommend you start by navigating to the `tutorial_notebooks` directory on the left panel and go through the `Using_DataJoint_to_Access_Functional_Data.ipynb` Jupyter notebook.

## Technical documentation

Technical documentation on the functional data can be found on the MICrONS project website [here](https://www.microns-explorer.org/cortical-mm3#f-data).

Download the nda v8 database technical documentation, which includes a changelog from the v7 database [here](https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/two_photon_processed_data_and_metadata/database_v8/two_photon_processed_data_and_metadata_technical_documentation_v8.pdf).

Separate from the data contained in the nda database, the MICrONS project website [here](https://www.microns-explorer.org/cortical-mm3#f-data) contains technical docs and access instructions for:

1. Raster- and motion-corrected functional scan tiffs.
2. Raster- and motion-corrected two-photon structural stack tiff
3. Stitched and temporally aligned stimuli for each scan

## Installation Instructions

This package requires access to the functional database. To download the SQL database and the pre-built Docker access images start with the `microns-nda-access` repo [here](https://github.com/cajal/microns-nda-access).

Once inside your properly configured environment run the tutorials below:

## Tutorials

[Using DataJoint to Access Functional Data Tutorial](tutorial_notebooks/Using_DataJoint_to_Access_Functional_Data.ipynb)

[Matched Cell Functional Data Tutorial](tutorial_notebooks/Matched_Cell_Functional_Data.ipynb)

## nda table descriptions

![nda](images/nda_erd.png)

**nda.Scan:** Information on completed scans. Cajal Pipeline: [meso.ScanInfo](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/meso.py#L29)

**nda.ScanInclude:** Scans determined suitable for analysis. 

**nda.Field:** Individual fields of scans. Cajal Pipeline: [meso.ScanInfo.Field](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/meso.py#L54)

**nda.RawManualPupil:** Pupil traces for each scan from the left eye collected at ~20 Hz and semi-automatically segmented. 

**nda.ManualPupil:** Manual pupil traces low-pass filtered with a hamming window to the scan frame rate and linearly interpolated to scan frame times.

**nda.RawTreadmill:** Cylindrical treadmill rostral-caudal position extracted with a rotary optical encoder at ~100Hz and converted into velocity.

**nda.Treadmill:** Treadmill velocities low-pass filtered with a hamming window to the scan frame rate then linearly interpolated to scan frame times.

**nda.ScanTimes:** Timestamps of scan frames in seconds relative to the start of the scan for the first pixel of the first imaging field.

**nda.Stimulus:** For each scan, contains the movie aligned to activity traces in `nda.Activity`.

**nda.Trial:** Contains information for each trial of the movie in `nda.Stimulus`. There are three types of trials, `Clip`, `Monet2`, and `Trippy`. Each unique trial has its own `condition_hash`. To get detailed information for each trial stimulus, join each `condition_hash` according to its corresponding type in one of: `nda.Clip`, `nda.Monet2`, or `nda.Trippy`.

**nda.Clip:** Detailed information for movie clips.

**nda.Monet2:** Detailed information for the Monet2 stimulus.

**nda.Trippy:** Detailed information for the Trippy stimulus.

**nda.RasterCorrection:** Raster phase correction applied to each scan field.

**nda.MotionCorrection:** Motion correction applied to each scan frame.

**nda.MeanIntensity:** Mean intensity of imaging field over time. Cajal Pipeline: [meso.Quality.MeanIntensity](https://github.com/cajal/pipeline/blob/fa202ee43437a67d55719e8ae9769ee9937581d0/python/pipeline/meso.py#L173)

**nda.SummaryImages:** Reference images of the scan field. Cajal Pipeline: [meso.SummaryImages](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L571)

**nda.Stack:** High-res anatomical stack information. Cajal Pipeline: [stack.CorrectedStack](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/stack.py#L733)

**nda.Registration:** Parameters of the affine matrix learned for field registration into the stack. Cajal Pipeline: [stack.Registration.Affine](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/stack.py#L1333)

**nda.Coregistration:** Coregistration transform solutions from the Allen Institute. [em_coregistration](https://github.com/AllenInstitute/em_coregistration/phase3)

**nda.Segmentation:** CNMF segmentation of a field using CaImAn package (https://github.com/simonsfoundation/CaImAn). It records the masks of all segmented cells. Mask_id's are unique per field. Cajal Pipeline: [meso.Segmentation.Mask](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L765)

**nda.Fluorescence:** Records the raw fluorescence traces for each segmented mask. Cajal Pipeline: [meso.Fluorescence.Trace](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1159)

**nda.ScanUnit:** Unit_id assignment that is unique across the entire scan. Includes info about each unit. Cajal Pipeline: [meso.ScanSet.Unit / meso.ScanSet.UnitInfo](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1341)

**nda.UnitHash:** Assigns hash and semantic string to each unique session - scan\_idx - unit\_id triplet.\\

**nda.Activity:** Deconvolved spike trace from the fluorescence trace. Cajal Pipeline: [meso.Activity.Trace](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1501)

**nda.StackUnit:** Unit coordinates in stack reference frame after field registration. `stack_x, stack_y, stack_z` should be used for transformation to EM space using Coregistration. [meso.StackCoordinates.UnitInfo](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1672)

**nda.AreaMembership:** Visual area labels for all units.

**nda.MaskClassification:** Classification of segmented masks into soma or artifact. Uses CaImAn package (https://github.com/simonsfoundation/CaImAn). Cajal Pipeline: [meso.MaskClassification.Type](https://github.com/cajal/pipeline/blob/6f44fdbd186905d95a9a86d6d60ad147df24f9e2/python/pipeline/meso.py#L1478)

**nda.Oracle:** Leave-one-out correlation for repeated videos in stimulus.

For more documentation see: [Cajal Pipeline Documentation](https://cajal.github.io/atlab-docs.github.io/pipeline.html)

## Acknowledgement of Government Sponsorship

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00003. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.
