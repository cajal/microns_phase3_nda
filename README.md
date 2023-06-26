# microns_phase3_nda
nda schema for MICrONS phase3. For more on the MICrONS project please see: [MICrONS Explorer](https://www.microns-explorer.org/)

## Technical documentation
Technical documentation on the functional data can be found [here](https://www.microns-explorer.org/cortical-mm3#f-data).

## Installation Instructions
This package requires access to the functional data. To download the SQL database and the pre-built Docker access images start with the microns-nda-access repo [here](https://github.com/cajal/microns-nda-access).

Once your environment is properly configured run the tutorials below:

## Tutorials:

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
