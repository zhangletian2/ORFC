import json
import os

import os
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
root_dir = os.path.join(_PROJECT_ROOT, "features", "test")
models = ["dinov2_vitl14"]
blks = ["blk05", "blk11", "blk17", "blk23"]
method="residual_trunc_mse_lmbda"
lmbdas=[0.1, 1, 10]

for model in models:
    print(f"model={model}")
    for lmbda in lmbdas:
        print(f"lmbda={lmbda}")
        print(f"BLK\t\tBPFP\t\tMSE")
        for blk in blks:
            json_path = os.path.join(root_dir, model, "split_decoded", f"{method}{lmbda}", f"stat_{blk}.json")
            if not os.path.exists(json_path):
                print(f"{blk} missing, skipping")
                continue
            with open(json_path, "r") as f:
                data = json.load(f)
                results_by_layer = data["results_by_layer"]
                
                layer_data = results_by_layer[blk]
                bpfp = float(layer_data["bpfp"][0])
                mse = float(layer_data["mse"][0])

                print(f"{blk}\t\t{bpfp:.4f}\t\t{mse:.4f}")

