### Data Processing

Processed tasks live under `data_process/data/task_*`.

Raw collected task layout:

```text
task_xxx/
  rgb/
  depth/
  states.csv
```

Final training-ready layout:

```text
task_xxx/
  four_channel/
  states_filtered.csv
  phase_proto_targets.npz
```

## Recommended entrypoint

Run preprocessing from the project root:

```powershell
python prepare_act_data.py
```

This pipeline:

1. cleans `states.csv` into `states_clean.csv`
2. synchronizes `states_clean.csv`, `rgb/`, and `depth/`
3. builds `depth_normalized/` and `four_channel/`
4. amplifies gripper motion in `j10` around the dataset mean
5. validates the final training files

The final validation checks:

- `states_filtered.csv` exists and has `frame,j1,j2,j3,j4,j5,j10`
- `frame` is continuous from `0..N-1`
- `four_channel/*.png` matches the CSV frame index exactly
- each four-channel image is `480x640x4 uint8`
- joint values stay inside the fixed physical limits

Tasks that already contain `states_filtered.csv` and `four_channel/` are treated as final-only tasks and still participate in validation.

## Phase-prototype targets

The new training path also expects offline phase-prototype supervision:

```powershell
python tools/build_phase_prototype_targets.py --data-root data_process/data
```

This script:

- fits PCA on flattened `10x6 -> 60D` action targets
- keeps the subspace that explains `85%` variance by default
- clusters that PCA space into prototype centers
- solves the target mixture coefficients `alpha_tgt`
- writes:
  - one global bank file under `data/_phase_proto/`
  - one `phase_proto_targets.npz` per task

Each `phase_proto_targets.npz` contains:

- `frame_index`
- `alpha_tgt`
- `prototype_tgt`
- `residual_tgt`

## Internal modules

- `data_process_1.py`: state/image synchronization, smoothing, static-frame filtering, frame reindexing
- `data_process_2.py`: strict RGB/depth matching and `four_channel/*.png` generation
- `data_loader.py`: strict training-time alignment between `states_filtered.csv`, `four_channel/*.png`, and `phase_proto_targets.npz`
