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
import numpy as np

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
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--output", type=str, default='./output/')  # TODO: actually copy results there

# --- parse argument in a way compatible with blender's REPL
if "--" in sys.argv:
  FLAGS = parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
else:
  FLAGS = parser.parse_args(args=[])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

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
                material=floor_material, static=True)
north_wall = kb.Cube(scale=(1.2, 0.1, 1), position=(0, 1.1, 0.9),
                     material=wall_material, static=True)
south_wall = kb.Cube(scale=(1.2, 0.1, 1), position=(0, -1.1, 0.9),
                     material=wall_material, static=True)
east_wall = kb.Cube(scale=(0.1, 1, 1), position=(1.1, 0, 0.9),
                    material=wall_material, static=True)
west_wall = kb.Cube(scale=(0.1, 1, 1), position=(-1.1, 0, 0.9),
                    material=wall_material, static=True)

simulator.add(floor)  # TODO: add should accept a list
simulator.add(north_wall)
simulator.add(south_wall)
simulator.add(east_wall)
simulator.add(west_wall)


spawn_area = (-1, -1, 0), (1, 1, 0.2)
velocity_range = (-1, -1, 0), (1, 1, 0)


camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=2.2)  # looks down by default

# no lights needed because of the flat material

nr_objects = rnd.randint(4, 10)
objects = []
for i in range(nr_objects):
  ball_material = kb.FlatMaterial(color=kb.Color.from_hsv(float(rnd.rand(1)[0]), 1, 1),
                                  indirect_visibility=False)
  objects.append(kb.Sphere(scale=(0.1, 0.1, 0.1), material=ball_material,
                            velocity=rnd.uniform(*velocity_range)))


def random_position(obj, area):
  effective_area = np.array(area) - np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
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
renderer.render(path=FLAGS.output)


