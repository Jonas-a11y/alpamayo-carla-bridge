# Camera configuration

[README](../README.md) · [Model profiles](MODELS.md) · [First-run tutorial](TUTORIALS.md#2-first-edge-drive)

All cameras are RGB CARLA sensors attached to the actor origin. The configured
resolution is shared; the model helper resizes to its own per-camera pixel budgets.
CARLA supplies pinhole images. Matching a nominal FOV does not reproduce the
training dataset's full lens calibration, distortion or vehicle geometry.

## Default rig

| Name | Model ID | FOV | Actor-relative position (m) | CARLA yaw |
| --- | --- | --- | --- | --- |
| front_left | 0 (cross left) | 120° | 0.6, −0.9, 1.6 | −90° |
| front_wide | 1 | 120° | 1.5, 0, 1.6 | 0° |
| front_right | 2 (cross right) | 120° | 0.6, 0.9, 1.6 | 90° |
| rear_left | 3 | 70° | −1.5, −0.7, 1.6 | −135° |
| rear_right | 5 | 70° | −1.5, 0.7, 1.6 | 135° |
| front_tele | 6 | 30° | 1.5, 0, 1.6 | 0° |

Edge and Super use all six cameras. R1 and 1.5 use IDs 0, 1, 2 and 6. The front_left/front_right
names are retained for the cross-camera views and existing video layout.

Camera positions use CARLA X forward, Y right and Z up, relative to the actor.
Negative yaw turns left; positive pitch tilts upward. These positions are
approximate sedan placements and must be inspected for occlusion on the actual
vehicle. The model ego origin is separately configured by `vehicle.rear_axle_x`;
the default −1.4 m is an approximate Model 3 rear-axle offset, not a measured
calibration. Recorded ego states and model history use that origin.

## Temporal profile

| Model | Front wide and tele | Other active cameras | Ego poses |
| --- | --- | --- | --- |
| Edge | −2, −1, −0.5, −0.1, 0 s | −1, −0.1, 0 s | 20 at 0.1 s, ending now |
| R1 / 1.5 / Super | −0.3, −0.2, −0.1, 0 s | −0.3, −0.2, −0.1, 0 s | 16 at 0.1 s, ending now |

The bridge captures every synchronous tick, keeps actual frame IDs/timestamps,
and selects these histories. It never duplicates a current image to fill the
window. A missing camera configuration, capture timeout, skipped frame or
incorrect timestamp is an explicit failure.

Edge's helper applies front-camera pixel bounds of 163840–196608 and other-camera
bounds of 81920–98304. It sorts camera/time fields together, labels relative times,
resizes RGB uint8 images and applies checkpoint normalization.

## Inspecting the rig

On the CARLA host:

~~~bash
python scripts/setup_carla_cameras.py --config config.yaml
~~~

This saves a current snapshot per active camera under `camera_preview/` and
cleans up its actors. Adjust placement, FOV, resolution and rear-axle offset for
the actual blueprint before evaluating a driving run. The existing composite
video displays the four front views; recordings also retain Edge/Super's rear views.
The preview selects the variant from the YAML and needs CARLA and Pillow, but
does not load a model. To preview another rig, set `alpamayo.variant` in a copy
of the config before running the command.
