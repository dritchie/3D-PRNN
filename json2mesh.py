import argparse
import json
import numpy as np
import os
import trimesh
import trimesh.creation
import trimesh.util

parser = argparse.ArgumentParser(description='Convert a directory of cuboid JSON files into a directory of OBJ meshes')
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
        # Turn each cuboid into a mesh
        meshes = []
        for _,cuboid in cuboids.items():
            xform = np.zeros((4, 4))
            xform[0:3, 0] = np.array(cuboid["xdir"])
            xform[0:3, 1] = np.array(cuboid["ydir"])
            xform[0:3, 2] = np.array(cuboid["zdir"])
            xform[0:3, 3] = np.array(cuboid["center"])
            mesh = trimesh.creation.box(
                extents=[cuboid["xd"], cuboid["yd"], cuboid["zd"]],
                transform=xform
            )
            meshes.append(mesh)
        # Combine these together
        combomesh = trimesh.util.concatenate(meshes)
        # Save as OBJ
        combomesh.export(f'{args.output_dir}/{name}.obj')

        