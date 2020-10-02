# microns_phase3_nda
nda schema for MICrONS phase3

## Installation Instructions
This package requires the em_coregistration package from the Allen Institute:

```bash
pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3
```

Install this package:

```bash
pip3 install git+https://github.com/cajal/microns_phase3_nda.git
```

## Import Instructions

In a jupyter notebook:

```python
from phase3 import nda, func, utils
```

Import datajoint. Configuration instructions: https://docs.datajoint.io/python/setup/01-Install-and-Connect.html

```python
import datajoint as dj
```

## Using the schema

To view schema ERD:
```python
dj.ERD(nda)
```

![nda](images/nda_erd.png)

For tutorial see: [notebook example](notebooks/microns_phase3_nda_examples.ipynb) 

## nda table descriptions

**nda.Scan:** Information on completed scans. Cajal Pipeline: [meso.ScanInfo](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/meso.py#L29)

**nda.Pupil:** Pupil traces for each scan. Cajal Pipeline: [pupil.py](https://github.com/cajal/pipeline/blob/master/python/pipeline/pupil.py)

**nda.Treadmill:** Treadmill for each scan. Cajal Pipeline: [treadmill.py](https://github.com/cajal/pipeline/blob/master/python/pipeline/treadmill.py)

**nda.Stimulus:** Not implemented yet.

**nda.Field:** Individual fields of scans. Cajal Pipeline: [meso.ScanInfo.Field](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/meso.py#L54)

**nda.SummaryImages** Reference images of the scan field. Cajal Pipeline: [meso.SummaryImages](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L571)

**nda.Stack:** High-res anatomical stack information. Cajal Pipeline: [stack.CorrectedStack](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/stack.py#L733)

**nda.Registration:** Parameters of the affine matrix learned for field registration into the stack. Cajal Pipeline: [stack.Registration.Affine](https://github.com/cajal/pipeline/blob/6a8342bf3edb07f5653c61024742258295cd8014/python/pipeline/stack.py#L1333)

**nda.Coregistration:** Coregistration transform solutions from the Allen Institute. [em_coregistration](https://github.com/AllenInstitute/em_coregistration)

**nda.Segmentation:** CNMF segmentation of a field. It records the masks of all segmented cells. Mask_id's are unique per field. Cajal Pipeline: [meso.Segmentation.Mask](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L765)

**nda.Fluorescence:** Records the raw fluorescence traces for each segmented mask. Cajal Pipeline: [meso.Fluorescence.Trace](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1159)

**nda.ScanUnit:** Unit_id assignment that is unique across the entire scan. Includes info about each unit. Cajal Pipeline: [meso.ScanSet.Unit / meso.ScanSet.UnitInfo](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1341)

**nda.Activity:** Deconvolved spike trace from the fluorescence trace. Cajal Pipeline: [meso.Activity.Trace](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1501)

**nda.StackUnit:** Unit coordinates in stack reference frame after field registration. `np_x, np_y, np_z` should be used for transformation to EM space using Coregistration. [meso.StackCoordinates.UnitInfo](https://github.com/cajal/pipeline/blob/921a920478c73687dd78b863fcd05e12bbf1e197/python/pipeline/meso.py#L1672)

For more documentation see: [Cajal Pipeline Documentation](https://cajal.github.io/atlab-docs.github.io/pipeline.html)

## Acknowledgement of Government Sponsorship

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00003. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.
