## Installation    
```bash
conda create -n easymocap python=3.9 -y
conda activate easymocap
python -m pip install -r requirements.txt
python setup.py develop
```
  
Next, remove line 1 in `chumpy/__init__.py` (inside python's `site-packages`)
```bash
from numpy import bool, int, float, complex, object, str, nan, inf
```

## Prepare Data  
```bash
python extract_frames.py --video_path ${Path to Input Video} --output_folder ${Path to Extracted Frames}
```

## Code Execution
### Extract Skeleton    
```bash
PYOPENGL_PLATFORM=egl emc --data config/datasets/svimage.yml --exp config/1v1p/hrnet_pare_finetune.yml --root ${Path to Extracted Frames}
```  

### Convert Skeleton to .bvh Format
```bash
python scripts/postprocess/convert2bvh.py --path ${Path to SMPL JSONs} --out ${Path to Extracted .bvh file}  
```