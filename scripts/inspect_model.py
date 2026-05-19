import os
import sys
from pathlib import Path
try:
    from ultralytics import YOLO
except Exception as e:
    print('Ultralytics import error:', e)
    raise
import torch

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('src/runs/detect/train-9/weights/best.pt')
print('Model path:', model_path)
print('Exists:', model_path.exists())
print('Size (MB):', model_path.stat().st_size / (1024*1024) if model_path.exists() else 'N/A')

# Try loading with ultralytics
try:
    m = YOLO(str(model_path))
    print('Loaded YOLO model object type:', type(m))
    print('model.names:', getattr(m, 'names', None))
except Exception as e:
    print('YOLO load error:', repr(e))

# Try torch.load to inspect checkpoint structure
try:
    ckpt = torch.load(str(model_path), map_location='cpu')
    print('torch.load type:', type(ckpt))
    if hasattr(ckpt, 'keys'):
        print('torch.load keys:', list(ckpt.keys())[:50])
    else:
        print('torch.load repr:', repr(ckpt)[:500])
except Exception as e:
    print('torch.load error:', repr(e))
