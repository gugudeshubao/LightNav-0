# Model assets and parameters

Checkpoint layout, the action decoder, `eval_config.json` and every server / CLI parameter.

## Checkpoint

A Hugging Face directory (or a directory containing `hf_ckpt/`) with `config.json`,
`model*.safetensors`, `tokenizer*`, `processor_config.json`, and ideally
`eval_config.json`. The architecture must be stock `Qwen3VLForConditionalGeneration`; the
trajectory / action tokens are ordinary rows of the embedding table.

## Action decoder

An **RVQ action-tokenizer bundle**: a directory with `manifest.json`, the per-level
codebook `.npy` files and the `jacobian_weights` file it names. The released checkpoints
ship it as `action_tokenizer/`, referenced from `eval_config.json`, so it is found without
flags; `--action_tokenizer_bundle` overrides it.

## `eval_config.json` (processing parameters)

If present next to or above the checkpoint (the search walks up four parents, so
`hf_ckpt/`, the `global_step_*` directory and the run root are all covered), it supplies the
processing parameters that must match training and cannot be guessed:

```json
{
  "version": 1,
  "common": {"video_size": [224, 320], "pool_enable": true, "pool_spatial": 2,
             "pool_mode": "avg", "pool_stage": "pre_vit"},
  "tasks": {
    "trackvla": {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "slowfast_tiers": null,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/action_tokenizer"}},
    "vlnce":    {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"}}
  }
}
```

| Key | Meaning |
|---|---|
| `common.video_size` | model input frame size `[H, W]`; every frame is resized to it |
| `common.pool_*` | spatial pooling of vision tokens (`pool_stage` `pre_vit` or `post_vit`) |
| `tasks.<task>.num_history_frames` | history window (frames) fed to the model per step |
| `tasks.<task>.predict_horizon` | trajectory horizon `H` (rows per chunk) |
| `tasks.<task>.video_fps` | frame rate the checkpoint was trained at (send frames at this rate or faster; waypoint rows carry no time base of their own, see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `tasks.<task>.slowfast_tiers` | optional multi-rate history layout (SlowFast checkpoints keep the whole episode) |
| `tasks.<task>.action_tokenizer` | decoder snapshot: `{"method": "rvq", "bundle_path"}`; relative paths resolve against the config's directory |

`tasks` is keyed by task: `trackvla` for tracking (`--task tracking`), `vlnce` for
navigation (`--task vln`, and the Habitat evaluation). Values are read automatically; the CLIs'
`--num_history_frames` (and `--pool_spatial`, where offered) override them. Without the file the code
falls back to conservative defaults (`num_history_frames=16`, no pooling,
`video_size=(224, 320)`) which will not match a trained checkpoint — treat a missing
`eval_config.json` as a warning sign. Full schema: `src/lightnav/eval_config.py`.


### Shipping the action decoder with the checkpoint

`tasks.<task>.action_tokenizer.bundle_path` may be a **relative path**; it is resolved
against the directory holding `eval_config.json` (falling back to the checkpoint
directory). The released checkpoints ship their decoder this way:

```
hf_ckpt/
  config.json, model-*.safetensors, tokenizer*, processor_config.json
  eval_config.json          # bundle_path: "action_tokenizer" (both tasks)
  action_tokenizer/         # manifest.json + codebook_l*.npy + jacobian_weights.npy (+ alpha)
```

With such a layout `lightnav-serve`, `lightnav-predict`, `lightnav-eval-habitat` and
`build_tracking_agent()` need no `--action_tokenizer_bundle`: the
decoder is taken from the task entry that matches the server task (`--task tracking` →
`trackvla`, `--task vln` → `vlnce`), then from the other tasks, then from the sibling
directories `action_tokenizer/<task>` or `action_tokenizer/`. The explicit flag always
wins.

## Server / CLI parameters

`lightnav-serve` flags (environment names are what `scripts/start_servers.sh` and
`docker/entrypoint.sh` read):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model_path` | `MODEL_PATH` | required | checkpoint directory |
| `--action_tokenizer_bundle` | `ACTION_TOKENIZER_BUNDLE` | from the checkpoint | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `--task` | `TASK` | `tracking` | `tracking` (tracking prompt, `tasks.trackvla`) or `vln` (VLN prompt, `tasks.vlnce`) |
| `--backend` | `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `--gpu_memory_utilization` | `GPU_MEM_UTIL` | `0.85` | fraction of GPU memory handed to the vLLM engine |
| `--vllm_quantization` | `VLLM_QUANTIZATION` | auto | explicit vLLM method such as `fp8` or `bitsandbytes`; unset auto-detects serialized quantization metadata or uses the checkpoint dtype |
| `--max_batch_size` | `MAX_BATCH_SIZE` | `8` | max sessions per scheduler tick; also sizes vLLM `max_num_seqs`. `1` = strictly serial |
| `--max_wait_ms` | `MAX_WAIT_MS` | `8` | max wait to fill a batch; a lone request is flushed after 2 ms |
| `--num_history_frames` | `NUM_HISTORY_FRAMES` | checkpoint config | history-window override (normally leave unset) |
| `--pool_spatial` | `POOL_SPATIAL` | checkpoint config | spatial-pooling override |
| `--aspect_mode` | `ASPECT_MODE` | `stretch` | `stretch`: resize every frame to `video_size` (training behaviour); `keep`: per-session size with the camera's aspect ratio at the same pixel budget (4:3 → 288×384 for a 256×448 checkpoint) |
| `--max_new_tokens` | `MAX_NEW_TOKENS` | `8` | lower bound for the per-step decode cap (see below) |
| `--host` / `--port` | `HOST` / `PORT` | `0.0.0.0` / `8050` | bind address |
| `--ready_file` | – | none | file touched once the port is bound (used by the launchers) |
| `--no_warmup` | – | off | skip the synthetic warm-up inference before binding the port |
| `--record_dir` | `RECORD_DIR` | off | record every connection's episodes (frames + per-step records) for `lightnav-render` (see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `--record_fps`, `--record_timeline`, `--no_record_images` | `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES` | `10`, `realtime`, images on | recording defaults written to the manifest / frame storage |
| `--cam_hfov_deg`, `--cam_height`, `--traj_forward_offset`, `--waypoint_dt_s` | `CAM_HFOV_DEG`, `CAM_HEIGHT`, `TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S` | `90`, `0.5`, auto, `0.1` | client-camera geometry and HUD convention used only by the rendered overlay |

### MuJoCo demo parameters

`mujoco_demo` is a separate `uv` project. Run `./run.sh` for its bundled
TurtleBot, or use `uv run --extra microduck vln-mujoco ...` when selecting the
optional MicroDuck backend. MicroDuck needs two external files, the MJCF from
`pollen-robotics/microduck_rl` and an ONNX policy from
`pollen-robotics/microduck-policies`; the download steps are in
[mujoco_demo/README.md](../mujoco_demo/README.md#microduck-optional).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8088` | web-console bind address |
| `--vln-server` | empty | default LightNav WebSocket URL shown in the console |
| `--robot` | `turtlebot` | robot backend: `turtlebot` or `microduck` |
| `--robot-model` | none | external MicroDuck MJCF; required only with `--robot microduck` |
| `--walking-policy` | none | external MicroDuck ONNX policy; required only with `--robot microduck` |

Engine-level environment variables (no flag):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLN_KV_CACHE_GIB` | auto | vLLM KV-cache size in GiB (auto-scales with `max_num_seqs`, floor 2 GiB) |
| `VLN_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA-graph capture (faster start, slower steps) |
| `LIGHTNAV_ATTN` | `sdpa` | attention implementation for the `hf` backend (e.g. `flash_attention_2`) |
| `VLN_EVAL_TEMPERATURE` / `TOP_P` / `TOP_K` | greedy | sampling knobs for experiments; benchmark numbers assume they are unset |

**Decode token budget.** A step emits a *grounding prefix* followed by the action token(s):
one `<act_l{d}_*>` per RVQ level. The server
probes the tokenizer for the grounding families it knows and caps generation at
`max(--max_new_tokens, prefix + action_tokens)`, which saves the decode step otherwise spent
reaching `eos` without ever truncating the action.

| checkpoint family | grounding prefix | tokens |
|---|---|---|
| tracking, legacy | `<tpos_k>` | 1 |
| tracking + grid pointing | `<opos_k>` | 1 |
| navigation + dual pointing | `<apos_A><opos_O>` | 2 |

**GPU sizing.** The 4B checkpoint needs ~10 GB for weights in bf16 plus the vLLM KV cache;
one 24 GB GPU comfortably serves one engine with `--gpu_memory_utilization 0.85`. When
several servers share a GPU, `scripts/start_servers.sh` divides the utilisation by
`SERVERS_PER_GPU`.

Note what `--gpu_memory_utilization` does *not* do here. The engine passes an explicit
`kv_cache_memory_bytes` (`VLN_KV_CACHE_GIB`, floor 2 GiB) and therefore skips vLLM's
profile run, so real usage is roughly weights + that KV cache + CUDA graphs (~15 GB for
the released 4B checkpoint) whatever the fraction says. The fraction still acts as the
ceiling vLLM checks against free memory at start-up, so lower it when the GPU is shared
with another job, and raise `VLN_KV_CACHE_GIB` — not the fraction — when you want a
bigger KV cache.

`--vllm_quantization` is passed directly to the pinned vLLM version and only applies to
`vllm_local`. Use it for an on-load method such as `fp8`; pre-quantized checkpoints should
normally leave it unset so vLLM reads their metadata. A checkpoint being exportable does
not imply that vLLM 0.19.1 can execute its format, so validate model loading and a real
multimodal action decode before treating a format as supported. Selecting `bitsandbytes`
also selects vLLM's required `bitsandbytes` load format; the optional `bitsandbytes`
package must be installed separately.

---

## Input aspect ratio (`--aspect_mode`)

The checkpoints are trained on frames resized to a fixed `common.video_size` (256×448 for
the released checkpoint, ≈16:9). By default (`stretch`) every client frame is resized to
that size regardless of its aspect ratio, exactly as in training — a 4:3 camera therefore
arrives horizontally squeezed.

`--aspect_mode keep` (env `ASPECT_MODE=keep`; also `lightnav-predict --aspect_mode`,
`lightnav-eval-habitat --aspect_mode`, `build_tracking_agent(aspect_mode=)`) instead picks,
per session from its first frame, the size with the **source aspect ratio** whose area is
closest to `video_size` and whose sides are multiples of 32 (patch × merge; and of the pre-ViT
pooling factors when the checkpoint pools before the ViT). The vision-token count per frame
stays at the training budget, so history length and sequence length are unchanged:

| camera | `stretch` | `keep` |
|---|---|---|
| 16:9 (480×270, 1920×1080) | 256×448 | 256×448 (identical) |
| 4:3 (640×480) | 256×448 (squeezed) | 288×384 |
| 1:1 | 256×448 | 352×352 |

Frames within one session must share one size (the first frame decides; `reset` starts
over). `keep` avoids geometric distortion but changes the token grid the model sees, which
the released checkpoint was not trained on — validate on your robot before relying on it.
Full native resolution (feeding the camera's own pixel count) is not supported.

## Offline MuJoCo capture

`mujoco_demo/scripts/capture_replay.py` runs only the bundled TurtleBot simulation,
without a model server, web listener, ROS, or hardware transport. Run it with the
demo environment; headless Linux can use `MUJOCO_GL=egl`.

| Flag | Meaning |
|---|---|
| `--output-dir` | Required fresh absolute directory; existing directories are rejected. |
| `--route` | Required scripted route: `left`, `right`, or `forward`. |
| `--frames` | Number of observations, 1..256; default 24 at simulation-time 4 Hz. |
| `--instruction` | Replay instruction metadata; default `go to the television`. It does not control the scripted route. |

Outputs include lossless PNGs, packed RGB, synchronous base pose/velocity, collision
guard telemetry, RGB hashes, installed package versions, and non-actuating Soma
probe input JSON. The base pose is not a calibrated camera extrinsic. The routes
are open-loop scripted motion in one scene, not independent scenes, model-driven
navigation, or evidence of successful obstacle avoidance.

## Isolated DWA closed-loop simulation

`mujoco_demo/scripts/closed_loop_replay.py` connects model predictions to an
external, pure planning `soma_lightnav_dwa_stdio` executable and feeds simulated
motion back into the next rendered observation. It does not launch ROS or hardware.

| Flag | Meaning |
|---|---|
| `--output-dir` | Required fresh absolute directory, also used for the native Unix socket. Keep the socket path below 104 bytes. |
| `--backend` | `native` (default) or `hf`. No automatic fallback. |
| `--planner` | Required absolute path to the pure DWA stdio executable. |
| `--flash-source`, `--daemon` | Native backend: absolute source directory and inference daemon executable. |
| `--hf-source`, `--hf-model` | HF backend: absolute source directory containing `lightnav/tracking.py` and local checkpoint directory. Requires CUDA and transformers 5.8.0; runs BF16 offline. |
| `--steps` | Maximum model observations, 1..256; default 96. |
| `--points` | Consume all `10` waypoints (default) or indices `[0,2,4,7,9]` for `5`, without scaling. Raw ten-point predictions are always retained. |
| `--instruction` | Model instruction; default `go to the television`. |

Native mode requires `LIGHTNAV_MODEL_DIR`, `LIGHTNAV_EDGELLM_ENGINE_DIR`,
`LIGHTNAV_EDGELLM_VISUAL_ENGINE_DIR`, and `LIGHTNAV_EDGELLM_PLUGIN_PATH`.
It starts and cleans up its own inference daemon; it does not reuse a live service.
Both modes need the demo's MuJoCo dependencies and headless EGL support on Linux.

The stdio planner accepts JSON lines with `simulation_only: true`, `capture_pose`
(`[x,y,yaw]`), `state` (`[x,y,yaw,vx,vy,wz]`), ten `local_trajectory` objects
(`x,y,yaw`), `stop`, `point_count`, and world-XY `obstacles`. Replies must have
`ok: true`, `hardware_actuation: false`, and a bounded `[vx,vy,wz]` command.
Model body yaw remains separate from DWA path-bearing yaw; a near-zero-translation
non-stop prediction uses heading planning rather than being treated as arrival.

Observations run at simulation-time 4 Hz and DWA at 20 Hz. Physics pauses during
inference: this is not real-time deadline acceptance or the production task stack.
Obstacles are sampled by 360 horizontal rays within 4 m, not a complete safety map.
The planner retains its Go2 footprint defaults, which are not TurtleBot calibration.
The simulator's separate collision guard remains enabled. Reports retain PNGs,
RGB hashes, predictions, poses, commands, obstacles, guard events and binary hashes.
`execution_passed` means the harness completed and cleaned up, not navigation success.
A model stop is not independently verified arrival, and nearest-TV XY distance is
only a proxy in this single, fixed scene.

### Pre-robot safety tests

Set `SOMA_DWA_PLANNER` to the absolute pure DWA stdio binary to enable actual
planner/physics integration tests in the demo test suite. This executable must
not be a robot launch entry point. Run pytest with `--runxfail` for acceptance;
freshness requirements are ordinary regression tests, not expected failures.
These tests do not invoke a real model or ROS/HAL.
See the demo README for scope, commands and JUnit clearance measurements.

The web demo enforces `VLN_OBSERVATION_MAX_AGE_S=0.5` from the local monotonic
capture clock, before adopting any reply and on every tracking loop. Unknown
timestamps or expired observations revoke motion and pending MPC results;
expiry does not declare task completion. This is a fixed demo safety policy,
not a new model flag or validated real-robot deadline.

`mujoco_demo/scripts/realtime_demo_probe.py` runs the actual demo MPC and renderer
without a browser or hardware. Physics does not wait for model replies.

| Flag | Meaning |
|---|---|
| `--vln-server` | Required explicit loopback `ws://` model endpoint; no default external service. |
| `--output-dir` | Required fresh absolute directory for telemetry JSON. |
| `--seconds` | Wall-time sampling duration, 1..60 seconds; default 10. |
| `--instruction` | Default `go to the television`. |

The probe does not start the model server, which must be separately isolated.
`loop_completed` only means the harness ran; inspect model replies, physics time,
expiry errors and motion before drawing conclusions. It is not the Soma DWA
production task assembly and does not claim navigation success.

Probe reports include a bounded (last 512 replies) `observation_timings` trace:
capture-to-offer (rendering/encoding/delivery combined), client queue wait,
request round trip, reply-to-consume, total capture-to-consume and adoption result.
All durations use local monotonic time; request time is not isolated GPU latency.

The demo selects `VlnClient(prefetch_frames=False)`: while a request is in flight,
new frames are not admitted to its pending slot. Once it finishes, the next offered
frame is admitted without replacing any previously accepted frame. This avoids
one extra request-length of queue age. The client API retains `True` by default
for other callers. This is demand-driven sampling, not a guarantee of 4 Hz model
history; capture intervals and model behaviour still need validation. The 500 ms
capture-age gate remains unchanged.

Timing rows also preserve local monotonic `captured_at_s` / `consumed_at_s`,
optional model-server stage timings and input-token count. Server stage clocks
may overlap (for example `model_ms` contains engine stages); do not add every
reported duration together or call wall-clock stages pure GPU kernel timing.
Missing/invalid optional metrics are omitted without changing control decisions.

The probe's `continuity` report measures whether the preceding accepted path
remains within its capture-age deadline until the next accepted reply. A positive
gap equals `next_consume - previous_capture - max_age`. Rejected replies do not
renew a path. This is a lower bound on handoff delay: the following controller
solve and command scheduling may add delay. It is read-only diagnostic accounting,
not permission to extend a deadline. Timely individual replies do not guarantee
continuous valid path coverage.

### Continuous cloud / C++ DWA simulation

`mujoco_demo/scripts/cloud_realtime_probe.py` runs MuJoCo independently of a
persistent Soma C++ `FlashServeLightNav` subprocess and calls the existing pure
C++ DWA subprocess for control. No ROS, HAL or physical command transport starts.

| Flag | Meaning |
|---|---|
| `--client` | Required absolute `soma_flash_serve_lightnav_stdio` binary. |
| `--planner` | Required absolute `soma_lightnav_dwa_stdio` binary. |
| `--gateway` | Dedicated SSH tunnel, only `http://127.0.0.1:19882` accepted. |
| `--output-dir` | Required fresh absolute directory; raw RGB and report are retained here. |
| `--seconds` | Wall-clock run budget, 5..60; default 15. |
| `--instruction` | Initial instruction; default `go to the television`. |
| `--switch-instruction` | Optional new task at halfway; immediately revokes the old path. |
| `--inject-gap` | Withhold captures for one second at 75% of the run; not an actual network fault. |
| `--reset-retries` | Explicit session-open recovery budget, 0..20; default 0 (fail immediately). A failed attempt waits 0.5 s before retry; exhausting the budget fails the run. |
| `--transport-drop-seconds` | 0..5, default 0. For runs >=10 s, start an owned loopback TCP relay, close its real connections at one-third of the run, then restore forwarding after this duration. |
| `--session-journal` | Optional absolute persistent cancellation-record file in an existing owner-only (0700) directory. Enables C++ recovery of an acknowledged session after client restart. |

Run simulator and client on the same Linux host: both use CLOCK_MONOTONIC.
The timestamp precedes rendering, disk transfer and encoding. Consumption checks
the original 500 ms deadline and task epoch, including after DWA and on substeps.
History admits actual captures at roughly 4 Hz, with at most eight pending frames;
no artificial catch-up frames or silent dropping. Failed/late inference revokes the
path and opens a new episode explicitly. Task cancellation does not preempt GPU work.
Report real-time factor: render/planner overhead can make physics slower than wall
time even though inference itself does not pause it. Rays, unchanged Go2 footprint
defaults and TurtleBot collision guard are simulation approximations, not robot
calibration or semantic arrival acceptance.
Recovery holds zero commands, clears pending observations and explicitly resets
the model session. `progress.json` records atomic wall-time/epoch/readiness updates
for externally controlled transport-fault experiments. It is not a control input.
Without a session journal, the gateway's 300-second idle session lease still
applies: if a process exits while cancellation cannot reach the server, a new
process may receive 429 until that owned session is explicitly cancelled or
expires. With the journal, the C++ client exclusively locks a 0600 file, checks
endpoint/robot/release binding, cancels the recorded session and opens a new one.
Cancellation failure preserves the record. No trajectory or history is resumed.
Use a stable local gateway URL; a different ephemeral relay port fails binding
checks. Corruption, another live lock owner, unsafe permissions, or an unacknowledged
session-open state fail closed. Lost open acknowledgements still require explicit
reconciliation; the journal does not solve that ambiguity or add server authentication.
The optional TCP relay forwards real bytes to the dedicated SSH tunnel; it never
generates model responses. A closed relay connection exercises actual client HTTP
transport errors, but is not equivalent to Wi-Fi loss, packet blackholing or gateway
restart. External SSH termination experiments are recorded separately.

`mujoco_demo/scripts/cloud_restart_probe.py` validates acknowledged-session recovery
using the real loopback gateway, two owned C++ subprocesses, and a TCP relay.
It infers once, drops transport, verifies cancellation fails, SIGKILLs only its
own first child, restores transport, then starts a new child using the same private
journal and verifies a fresh session plus inference. It never starts robot transport.
Required flags: `--client` (absolute stdio client binary), `--input` (absolute
`soma-lightnav-simulation-input/v1` fixture), and `--output-dir` (fresh absolute
directory). Private state stays inside that directory on the remote device and
must not be published with article materials.

### Isolated language-only PTQ diagnostic

`scripts/lightnav_ptq_candidate.py` compares FP16 HF inference with a language-only
ModelOpt PTQ candidate on fixed navigation recordings. It never deploys a candidate.
Use the isolated CUDA environment with ModelOpt 0.45.0 and transformers 5.8.0;
it requires existing offline weights, not generic-text calibration downloads.

| Flag | Meaning |
|---|---|
| `--model-dir` | Required existing absolute HF checkpoint directory. |
| `--flash-source` | Required existing absolute Flash checkout containing the replay loader. |
| `--calibration-dir` | Required existing absolute navigation replay directory. |
| `--evaluation-dir` | Required separate existing absolute replay directory. |
| `--output-dir` | Required fresh absolute directory for the diagnostic report. |
| `--steps` | Replay length, 1..24 frames; default 8. |
| `--method` | `fp8` (default) or `nvfp4`; PTQ only, no QAT. |
| `--export` | Export a candidate checkpoint inside the output directory. |

Only language-layer weight and input quantizers are enabled. Vision, embeddings,
output head and the separate action tokenizer are excluded. The report compares
token IDs and decoded waypoints against the same FP16 inputs. Separate recordings
from one scene do not establish generalization; a short run is a pipeline check,
not navigation acceptance. Fake-quant HF latency is not native kernel performance.
Native export parity, native latency and closed-loop safety remain separate gates.

## Simulation capture-budget comparison

`mujoco_demo/scripts/cloud_realtime_probe.py --capture-age-ms {500,650,800}`
defaults to 500. The selected value must match the C++ client's reported budget;
a mismatch aborts the run. Soma's standalone `SOMA_LIGHTNAV_SIMULATION_BUDGETS=ON`
build adds `soma_lightnav_sim_650` and `soma_lightnav_sim_800`. This changes both
result admission and the adopted path deadline relative to the original capture
timestamp. It does not change robot defaults, sensor checks, or establish a safe
hardware timeout. These executables have no actuator transport.

Each control record also includes `decision`: the planner reply, endpoint distance
and signed projection, and the reason for motion or a zero command. Local reference
completion is distinct from model stop and never establishes navigation arrival.
Diagnostics do not change planner inputs, commands, or deadline enforcement.

`--finish-short-path-heading` is an opt-in simulation experiment (default off).
When the reference planner reports local completion, every original waypoint is
within 8 cm of its capture origin, and the endpoint remains within 8 cm of the
current position, the simulation adapter uses the model's terminal body yaw with
the existing collision-checked heading planner. It does not extend translation,
shrink the footprint, or change the expiry budget. It is not enabled on hardware.

