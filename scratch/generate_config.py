import torch
try:
    import torch_directml
except:
    pass
import json
import os
import sys

# Add train_pipeline to path to ensure sota_signal_generator can be found if needed
sys.path.append(os.path.join(os.getcwd(), 'train_pipeline'))

ckpt_path = 'train_pipeline/models_sota_v2/checkpoint.pt'
out_path = 'train_pipeline/models_sota_v2/sota_config.json'

if not os.path.exists(ckpt_path):
    print(f"Error: {ckpt_path} not found.")
    sys.exit(1)

print(f"Loading {ckpt_path}...")
# map_location='cpu' is critical for compatibility
ckpt = torch.load(ckpt_path, map_location='cpu')

features = ckpt.get('features', [])
if not features:
    # Use fallback list from sota_signal_generator if missing from ckpt
    from sota_signal_generator import FEATURE_COLS
    features = FEATURE_COLS

config = {
    "seq_len": ckpt.get('seq_len', 120),
    "patch_len": ckpt.get('patch_len', 12),
    "features": features,
    "temperature": ckpt.get('temperature', 1.0),
    "label_col": "tb_label",
    "best_val_macro_f1": ckpt.get('best_f1', 0.0)
}

with open(out_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Successfully generated {out_path}")
print(f"Features found: {len(features)}")
