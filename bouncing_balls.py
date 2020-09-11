# Copyright 2020 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import pathlib
import shutil
import tarfile
import tempfile
import uuid

import numpy as np
from google.cloud import storage

import sys; sys.path.append(".")

import kubric.pylab as kb
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=str, default="KLEVR",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--frame_start", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=96)  # 4 seconds
parser.add_argument("--logging_level", type=str, default="INFO")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--output", type=str, default='./output/')  # TODO: actually copy results there

# --- parse argument in a way compatible with blender's REPL
if "--" in sys.argv:
  FLAGS = parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
else:
  FLAGS = parser.parse_args(args=[])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# create a temporary working directory
work_dir = pathlib.Path(tempfile.mkdtemp())

# --- Setup logger
logging.basicConfig(level=FLAGS.logging_level)
logger = logging.getLogger(__name__)

# --- Configures random generator
if FLAGS.seed:
  rnd = np.random.RandomState(FLAGS.seed)
else:
  rnd = np.random.RandomState()

scene = kb.Scene(frame_start=FLAGS.frame_start,
                 frame_end=FLAGS.frame_end,
                 frame_rate=FLAGS.frame_rate,
                 step_rate=FLAGS.step_rate,
                 resolution=(FLAGS.resolution, FLAGS.resolution))


simulator = kb.Simulator(scene)

floor_material = kb.FlatMaterial(color=kb.Color.from_hexstr('#000000'),
                                 indirect_visibility=False)
wall_material = kb.FlatMaterial(color=kb.Color.from_hexstr('#ffffff'),
                                indirect_visibility=False)


# make a "room" of 1m x 1m x 1m with the floor at z=0
floor = kb.Cube(scale=(1, 1, 0.1), position=(0, 0, -0.1),
                material=floor_material, static=True, friction=0)
north_wall = kb.Cube(scale=(1.2, 0.1, 1), position=(0, 1.1, 0.9),
                     material=wall_material, static=True, restitution=1)
south_wall = kb.Cube(scale=(1.2, 0.1, 1), position=(0, -1.1, 0.9),
                     material=wall_material, static=True, restitution=1)
east_wall = kb.Cube(scale=(0.1, 1, 1), position=(1.1, 0, 0.9),
                    material=wall_material, static=True, restitution=1)
west_wall = kb.Cube(scale=(0.1, 1, 1), position=(-1.1, 0, 0.9),
                    material=wall_material, static=True, restitution=1)

simulator.add(floor)  # TODO: add should accept a list
simulator.add(north_wall)
simulator.add(south_wall)
simulator.add(east_wall)
simulator.add(west_wall)


nr_objects = 4
ball_radius = 0.2
spawn_area = (-1, -1, 0), (1, 1, 2 * ball_radius)
velocity_range = (-1, -1, 0), (1, 1, 0)


camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=2.2)  # looks down by default

# no lights needed because of the flat material


objects = []
for i in range(nr_objects):
  ball_material = kb.FlatMaterial(color=kb.Color.from_hsv(float(rnd.rand(1)[0]), 1, 1),
                                  indirect_visibility=False)
  objects.append(kb.Sphere(scale=(ball_radius, ball_radius, ball_radius),
                           material=ball_material,
                           friction=0,
                           restitution=1,
                           velocity=rnd.uniform(*velocity_range)))


def random_position(obj, area):
  effective_area = np.array(area) - ball_radius*np.array([[-1, -1, -1], [1, 1, 1]])
  return rnd.uniform(effective_area[0], effective_area[1])


for obj in objects:
  obj.position = random_position(obj, spawn_area)
  simulator.add(obj)
  collision = True
  trial = 0
  while collision and trial < 100:
    obj.position = random_position(obj, spawn_area)
    collision = simulator.check_overlap(obj)
    trial += 1
    print("  >>>>  TRIAL", trial)
  if collision:
    raise RuntimeError('Failed to place', obj)

# --- run the physics simulation
animation = simulator.run()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
renderer = kb.Blender(scene)
renderer.add(floor)
renderer.add(north_wall)
renderer.add(south_wall)
renderer.add(east_wall)
renderer.add(west_wall)
renderer.add(camera)
scene.camera = camera   # TODO: currently camera has to be added to renderer before assignment. fix!

for obj in objects:
  renderer.add(obj)
  # --- Bake the simulation into keyframes
  for frame_id in range(scene.frame_start, scene.frame_end):
    obj.position = animation[obj]["position"][frame_id]
    obj.quaternion = animation[obj]["quaternion"][frame_id]
    obj.keyframe_insert('position', frame_id)
    obj.keyframe_insert('quaternion', frame_id)

# --- Render or create the .blend file
renderer.render(path=work_dir)


# -------
# post-processing
# -------
import pickle
import json


images_path = work_dir / "images"
exr_path = work_dir / "exr"
layers_path = work_dir / 'layers.pkl'
gt_factors_path = work_dir / 'factors.json'
print("="*80)
print("="*80)
print(images_path)
print(exr_path)
print(layers_path)
print(gt_factors_path)
print("="*80)
print("="*80)

all_objects = [floor, north_wall, east_wall, south_wall, west_wall] + objects

T = scene.frame_end
W, H = scene.resolution
segmentation = np.zeros((T, W, H, 1), dtype=np.uint32)


for frame_id in range(scene.frame_end):
  print('-'*80)
  print(exr_path / f"frame_{frame_id:04d}.exr")
  layers = kb.get_render_layers_from_exr(exr_path / f"frame_{frame_id:04d}.exr")
  print([f"{k}: {v.shape}" for k, v in layers.items()])
  segmentation[frame_id, :, :, 0] = layers['SegmentationIndex'][:, :, 0]

gt_factors = []

for i, obj in enumerate(all_objects):
  # replace crypto-hashes with object index
  object_crypto_ids = kb.mm3hash(obj.uid)
  segmentation[segmentation == object_crypto_ids] = i
  crypto_id = kb.mm3hash(obj.uid)
  gt_factors.append({
      'mass': obj.mass,
      'color': obj.material.color.rgb,
      'animation': animation[obj],
  })

with open(layers_path, 'wb') as f:
  pickle.dump({'segmentation': segmentation}, f)

with open(gt_factors_path, 'w') as f:
  json.dump(gt_factors, f, indent=2, sort_keys=True)

# -------
# export
# -------
target_path = pathlib.Path(FLAGS.output)
target_path.mkdir(parents=True, exist_ok=True)  # ensure exists


uid = uuid.uuid4()
zip_filename = "{}.tar.gz".format(uid)

with tarfile.open(work_dir / zip_filename, "w:gz") as tar:
  tar.add(str(work_dir / 'scene.blend'), f'{uid}/scene.blend')
  tar.add(str(layers_path), f'{uid}/layers.pkl')
  tar.add(str(gt_factors_path), f'{uid}/factors.json')
  tar.add(str(images_path), f'{uid}/images')

if target_path.parts[0] == 'gs:':
  client = storage.Client()
  bucket = client.get_bucket(target_path.parts[1])
  dst_blob_name = pathlib.Path(*target_path.parts[2:]) / zip_filename
  blob = bucket.blob(str(dst_blob_name))
  blob.upload_from_filename(str(work_dir / zip_filename))
else:
  shutil.move(str(work_dir / zip_filename), str(target_path))

