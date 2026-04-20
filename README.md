# memory_enhanced_ACT

This branch keeps three compatible training / export modes:

- `baseline`: native ACT, no Phase-PCA supervision.
- `pca-residual`: Phase-PCA coordinate head plus residual action head.
- `pca-only`: strict encoder-only Phase-PCA coordinate head, no residual action head and no main decoder path.

The current paper-run convention is:

- `FUTURE_STEPS = 10`
- action target = step delta / `DELTA_QPOS_SCALE`
- `DELTA_QPOS_SCALE = 10.0`
- `NUM_EPOCHS = 25`
- no smoothing filter in preprocessing
- motion filter threshold = `DISTANCE_THRESHOLD = 10`
- gripper `j10` amplification scale = `1.4`

## Dataset Contract

Each task directory must contain:

```text
task_xxx/
  states_filtered.csv
  rgb/
  depth/
  depth_normalized/
  four_channel/
```

For Phase-PCA modes, each task also needs the matching target file:

```text
task_xxx/
  phase_pca8_targets.npz
  phase_pca16_targets.npz
  phase_pca32_targets.npz
```

The dataset root also needs PCA banks:

```text
dataset/
  _phase_pca8/phase_pca8_bank.npz
  _phase_pca16/phase_pca16_bank.npz
  _phase_pca32/phase_pca32_bank.npz
```

`phase_pca*_targets.npz` contains:

- `frame_index`
- `pca_coord_tgt`
- `pca_recon_tgt`
- `residual_tgt`

`phase_pca*_bank.npz` contains:

- `pca_mean`
- `pca_components`
- `pca_coord_mean`
- `pca_coord_std`
- `residual_mean`
- `residual_std`
- `explained_ratio`
- `pca_dim`
- `future_steps`
- `state_dim`
- `target_mode`
- `delta_qpos_scale`

## Preprocess Data

Run the canonical preparation pipeline:

```bash
python prepare_act_data.py --data-root /path/to/dataset --future-steps 10
```

This does:

1. clean raw `states.csv` when needed;
2. sync/filter trajectory and images when the task is still raw;
3. rebuild `four_channel` images;
4. amplify `j10` around the global mean by `1.4`;
5. validate the final training contract.

Prepared tasks are protected from accidental second filtering. If `states_filtered.csv`, `rgb`, `depth`, and `depth_normalized` are already aligned, the filter stage is skipped and only safe final stages are rebuilt.

Build Phase-PCA supervision:

```bash
python tools/build_phase_pca_bank_and_targets.py \
  --data-root /path/to/dataset \
  --output-bank /path/to/dataset/_phase_pca8/phase_pca8_bank.npz \
  --pca-dim 8 \
  --future-steps 10 \
  --image-channels 4 \
  --target-mode delta \
  --delta-qpos-scale 10 \
  --target-filename phase_pca8_targets.npz
```

Repeat with `8`, `16`, and `32`.

Validate Phase-PCA targets:

```bash
python tools/validate_phase_pca_targets.py \
  --data-root /path/to/dataset \
  --target-filename phase_pca8_targets.npz \
  --future-steps 10 \
  --target-mode delta \
  --delta-qpos-scale 10 \
  --output-json /path/to/dataset/_phase_pca8/phase_pca8_validation.json
```

Generate paper / sanity-check figures:

```bash
python tools/visualize_prepared_dataset_report.py \
  --data-root /path/to/dataset \
  --output-dir /path/to/dataset/_paper_report \
  --pca-dims 8,16,32 \
  --future-steps 10 \
  --delta-scale 10
```

## Training Modes

You can still edit `config.py` directly, but for paper runs prefer CLI overrides so every experiment is explicit.

Baseline ACT:

```bash
python training.py \
  --method baseline \
  --data-root /path/to/dataset \
  --exp-name paper_baseline_e25 \
  --num-epochs 25
```

PCA-residual:

```bash
python training.py \
  --method pca-residual \
  --data-root /path/to/dataset \
  --phase-pca-dim 16 \
  --phase-bank-path /path/to/dataset/_phase_pca16/phase_pca16_bank.npz \
  --phase-targets-filename phase_pca16_targets.npz \
  --exp-name paper_pca16res_e25 \
  --num-epochs 25
```

PCA-only:

```bash
python training.py \
  --method pca-only \
  --data-root /path/to/dataset \
  --phase-pca-dim 16 \
  --phase-bank-path /path/to/dataset/_phase_pca16/phase_pca16_bank.npz \
  --phase-targets-filename phase_pca16_targets.npz \
  --exp-name paper_pca16only_e25 \
  --num-epochs 25
```

In `pca-only`, the model uses:

```text
image/qpos -> visual encoder with phase token -> PCA head -> pca_recon action
```

The main transformer decoder, query embedding, `is_pad_head`, and residual head are not used for the action path. Decoder-side parameters are frozen so training is faster and the ablation is structurally clean.

Run all seven paper experiments serially with a dedicated run folder:

```bash
cd /home/ubuntu/code_projects/memory_enhanced_act
export DATA_ROOT=/home/ubuntu/code_projects/memory_enhanced_act/data_process/data
python scripts/run_paper_experiments.py
```

The script runs:

- `paper_baseline_e25`
- `paper_pca8res_e25`
- `paper_pca16res_e25`
- `paper_pca32res_e25`
- `paper_pca8only_e25`
- `paper_pca16only_e25`
- `paper_pca32only_e25`

Outputs are stored outside the default `log/` tree:

```text
paper_runs/run_YYYYMMDD_HHMMSS/
  runner_config.json
  runner_summary.json
  runner_logs/
    01_paper_baseline_e25_stdout.log
    ...
  experiments/
    paper_baseline_e25/
      config.json
      metrics.jsonl
      train_paper_baseline_e25.log
      ckpt_epoch_*.pth
      best_model.pth
    ...
```

You can override common hyperparameters through CLI or env vars, for example:

```bash
python scripts/run_paper_experiments.py \
  --data-root /home/ubuntu/code_projects/memory_enhanced_act/data_process/data \
  --epochs 25 \
  --batch-size 16 \
  --num-workers 16 \
  --log-print-freq 2000 \
  --qpos-noise-std 2.0
```

`scripts/run_paper_experiments.sh` is kept as a simple shell fallback, but the Python runner is the preferred paper-run entry point because it writes a global summary and per-experiment stdout logs.

`--log-print-freq` controls how often intra-epoch batch metrics are printed and appended to `metrics.jsonl`. The Python runner defaults it to `2000`, so paper-run logs stay readable while epoch summaries and validation metrics are still recorded every epoch.

The Python runner disables tqdm progress bars by default because captured progress bars can write one line per batch into `runner_logs/*_stdout.log`. If you want live progress bars for a short debug run, add `--show-progress`.

The Python runner is resumable. If the process is interrupted, run the same
command with the same `--run-root` again:

```bash
python scripts/run_paper_experiments.py \
  --data-root /home/ubuntu/code_projects/memory_enhanced_act/data_process/data \
  --run-root /home/ubuntu/code_projects/memory_enhanced_act/paper_runs/final_320_tasks_e25
```

Resume behavior:

- If an experiment already has `ckpt_epoch_25.pth` and `best_model.pth`, it is skipped.
- If it has a partial checkpoint such as `ckpt_epoch_7.pth`, the runner automatically passes `--resume-ckpt-path` and continues from epoch 8.
- If you really want to ignore existing checkpoints and rerun, add `--fresh`.

## Loss Definitions

Baseline:

```text
loss = RECON_LOSS_WEIGHT * recon_l1
     + KL_WEIGHT * kl
```

PCA-residual:

```text
loss = RECON_LOSS_WEIGHT * recon_l1
     + RESIDUAL_LOSS_WEIGHT * residual_l1
     + PCA_COORD_LOSS_WEIGHT * pca_coord_mse
     + KL_WEIGHT * kl
```

PCA-only:

```text
loss = RECON_LOSS_WEIGHT * recon_l1
     + PCA_COORD_LOSS_WEIGHT * pca_coord_mse
     + KL_WEIGHT * kl
```

`recon_l1` always supervises the final action sequence. `pca_coord_mse` supervises normalized PCA coordinates. `residual_l1` is used only when `USE_RESIDUAL_ACTION=True`.

## Important Config Groups

Baseline / method switches:

- `USE_PHASE_PCA_SUPERVISION`
- `USE_PHASE_TOKEN`
- `USE_RESIDUAL_ACTION`

Training hyperparameters:

- `NUM_EPOCHS`
- `BATCH_SIZE`
- `NUM_WORKERS`
- `LR`
- `LR_BACKBONE`
- `KL_WEIGHT`
- `RECON_LOSS_WEIGHT`
- `PCA_COORD_LOSS_WEIGHT`
- `RESIDUAL_LOSS_WEIGHT`

Architecture:

- `BACKBONE`
- `ENC_LAYERS_ENC`
- `ENC_LAYERS`
- `DEC_LAYERS`
- `HIDDEN_DIM`
- `DIM_FEEDFORWARD`
- `NHEADS`
- `PCA_HEAD_HIDDEN_DIM`
- `PCA_HEAD_DEPTH`

Preprocessing-coupled settings. Do not casually change without rebuilding data / PCA targets:

- `FUTURE_STEPS`
- `PREDICT_DELTA_QPOS`
- `DELTA_QPOS_SCALE`
- `IMAGE_CHANNELS`
- `PHASE_PCA_DIM`
- `PHASE_TARGETS_FILENAME`
- `PHASE_BANK_PATH`

## Qpos Input Noise

`QPOS_INPUT_NOISE_STD_PULSE` adds small Gaussian noise to the current `qpos(t)` input during training only.

- Unit is raw servo pulse, not normalized 0-1 space.
- Labels stay clean.
- Validation, export, and deploy do not add this noise.
- Current default is `2.0` pulses with clipping at `5.0` std.

## Export

Export follows checkpoint mode automatically:

```bash
python deploy/export_torchscript_models.py \
  --act-checkpoint ./log/exp_xxx/best_model.pth \
  --output-dir ./deploy_artifacts_xxx \
  --smoke-test
```

Force PCA-only export from a PCA-residual checkpoint:

```bash
python deploy/export_torchscript_models.py \
  --act-checkpoint ./log/exp_xxx/best_model.pth \
  --phase-bank-path /path/to/dataset/_phase_pca8/phase_pca8_bank.npz \
  --output-dir ./deploy_artifacts_pca8_only \
  --pca-only \
  --smoke-test
```

The exported `deploy_config.yml` records:

- `use_phase_pca_supervision`
- `use_phase_token`
- `use_residual_action`
- `pca_only`
- `phase_pca_dim`
- `predict_delta_qpos`
- `delta_qpos_scale`

The ROS deploy node reads the same artifact shape for baseline, PCA-residual, and PCA-only.
