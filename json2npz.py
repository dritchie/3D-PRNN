import argparse
import json
import numpy as np
import os

parser = argparse.ArgumentParser(description='Convert a directory of cuboid JSON files into a directory of NPZ files for the Blender rendering script')
parser.add_argument('--input-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str)
args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = args.input_dir

for entry in os.listdir(args.input_dir):
    name, ext = os.path.splitext(entry)
    if ext == '.json':
        with open(f'{args.input_dir}/{entry}', 'r') as f:
            cuboids = json.loads(f.read())
        # Sometimes we generate an empty list of cuboids; skip these
        if not cuboids:
            continue
        cuboids = cuboids.values()
        scene_cubes = np.array([
            np.array(c['center'] + [c['xd'], c['yd'], c['zd']] + c['xdir'] + c['ydir'])
        for c in cuboids])
        # scene_aps = np.array([0, 0, 0]).reshape(1, 3) # Single dummy attachment point at the origin?
        scene_aps = np.array([])
        np.savez(os.path.join(f'{args.output_dir}/{name}.npz'), scene_cubes, scene_aps)