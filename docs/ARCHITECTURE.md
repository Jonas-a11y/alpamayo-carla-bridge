# Architecture

The installed 'bridge' package exposes 'CarlaBridge', 'AlpamayoInference',
'ObservationBuffer' and the result/state types. The CLI calls the same driving
lifecycle. Optional simulator, model and video imports happen only when used.

## Data flow

1. 'CarlaClient' connects and enables synchronous 0.1-second ticks.
2. 'CameraManager' spawns the configured model rig and queues RGB images together
   with CARLA frame IDs and captured simulation timestamps. Collection discards
   stale frames and rejects missing or future frames.
3. 'EgoMotionTracker' captures the pose for that tick, using the configured
   rear-axle offset from the actor origin. 'ObservationBuffer' retains real
   camera and ego histories. Initial capture runs with the brake applied.
4. History positions and full rotations are reflected from CARLA's left-handed
   coordinates and transformed into the current ego frame (X forward, Y left).
5. Edge inputs contain six cameras, 22 images, captured timestamps, explicit
   navigation and 20 historical poses. R1 inputs contain four cameras with
   four frames each and 16 historical poses.
6. 'AlpamayoInference' calls the matching official loader/helper and trajectory
   sampling method. It validates finite output shapes and returns candidate zero
   and its own reasoning. It propagates failures rather than generating a fallback.
7. 'TrajectoryController' translates ego-frame waypoints into CARLA throttle,
   steering and brake. Speed estimation uses the model's 10 Hz trajectory grid.
8. Optional 'Recorder' saves all current camera views and telemetry. Optional
   'VideoComposer' renders the four front views, reasoning and world-space paths.

## Lifecycle and time

'CarlaBridge.start()' loads the model before creating simulator actors. It then
captures enough history: 21 ticks for Edge's two-second image window, or 16 for
R1's ego history. 'step()' captures the next tick and applies the next decision.
The world waits during inference; this is synchronous simulation, not a
real-time autonomous-driving deployment.

A failed step commands full braking and raises. Context exit destroys sensors
and vehicle and restores previous world settings, including after partial startup
failure. CLI failures return a nonzero exit status. Only the bridge should tick
this synchronous world; a competing client can invalidate its observations.

Camera timestamps are integer microseconds of simulation time, not wall-clock
inference time. Consecutive ticks allow one microsecond of floating-point rounding.
Recorded states retain frame IDs and timestamps. The training rear-axle origin
and camera extrinsics need calibration on the actual CARLA vehicle.

## Model runtimes

Edge and R1 use separate Python 3.12 environments because their Transformers
versions differ. The Edge extra pins its official source revision and dependencies.
R1's documented SDPA setup installs the pinned upstream code without unnecessary
FlashAttention/dataset dependencies. See the README for installation and limits.

CPU checks cover native Edge sample/prompt contracts, mocked model adapters,
sensor synchronization, coordinate conversion, braking and resource cleanup.
They do not establish CUDA inference or closed-loop driving quality.
