# Architecture

The installed `bridge` package exposes `CarlaBridge`, `AlpamayoInference`,
`ObservationBuffer` and the result/state types. The CLI calls the same driving
lifecycle. Optional simulator, model and video imports happen only when used.

[README](../README.md) · [Model contracts](MODELS.md) · [Tutorials](TUTORIALS.md)

## Data flow

1. `CarlaClient` connects and enables synchronous 0.1-second ticks.
2. `CameraManager` spawns the configured model rig and queues RGB images together
   with CARLA frame IDs and captured simulation timestamps. Collection discards
   stale frames and rejects missing or future frames.
3. `EgoMotionTracker` captures the pose for that tick, using the configured
   rear-axle offset from the actor origin. `ObservationBuffer` retains real
   camera and ego histories. Initial capture runs with the brake applied.
4. History positions and full rotations are reflected from CARLA's left-handed
   coordinates and transformed into the current ego frame (X forward, Y left).
5. Edge inputs contain six cameras, 22 images, captured timestamps, explicit
   navigation and 20 historical poses. R1/1.5 inputs contain four cameras with
   four frames each and 16 historical poses; Super uses six cameras with four
   frames each and 16 poses. 1.5 accepts optional navigation; R1 and the standard
   Super trajectory path omit it.
6. `AlpamayoInference` calls the matching official loader/helper and trajectory
   sampling method. It validates finite output shapes and returns candidate zero
   and its own reasoning. It propagates failures rather than generating a fallback.
7. `TrajectoryController` translates ego-frame waypoints into CARLA throttle,
   steering and brake. Speed estimation uses the model's 10 Hz trajectory grid.
8. Optional `Recorder` saves all current camera views and telemetry. Optional
   `VideoComposer` renders the four front views, reasoning and world-space paths.

## Lifecycle and time

`CarlaBridge.start()` loads the model before creating simulator actors. It then
captures enough history: 21 ticks for Edge's two-second image window, or 16 for
R1/1.5/Super's ego history. `step()` captures the next tick and applies the next decision.
The world waits during inference; this is synchronous simulation, not a
real-time autonomous-driving deployment.

A failed step commands full braking and raises. Context exit destroys sensors
and vehicle and restores previous world settings, including after partial startup
failure. CLI failures return a nonzero exit status. Only the bridge should tick
this synchronous world; a competing client can invalidate its observations.
Restarting the same bridge reuses loaded weights and resets history, controller,
collision and recording state. A new model/configuration needs a new bridge.
The configured map may be loaded at connection; cleanup restores tick settings
and removes owned actors, but does not restore a previously loaded map.

Camera timestamps are integer microseconds of simulation time, not wall-clock
inference time. Consecutive ticks allow one microsecond of floating-point rounding.
Recorded states retain frame IDs and timestamps. The training rear-axle origin
and camera extrinsics need calibration on the actual CARLA vehicle.

## Model runtimes

Edge and Nano/Super use separate Python 3.12 environments because their Transformers
versions differ. The Edge extra pins its official source revision and dependencies.
The Nano/Super extra supplies shared inference dependencies; install the three
pinned source packages without unnecessary FlashAttention/dataset dependencies.
See [model setup](MODELS.md) for installation and limits.

CPU checks cover native input/prompt helpers and model imports for all variants,
mocked action experts, sensor synchronization, coordinate conversion, model
selection, restart, braking and resource cleanup.
They do not establish CUDA inference or closed-loop driving quality.
