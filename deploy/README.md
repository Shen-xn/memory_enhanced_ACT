# Deploy

The deploy path now keeps only the baseline single-image ACT export.

Inference path:

```text
rgb + depth + qpos -> preprocess(BGRA) -> act_inference.pt -> action sequence
```

Removed from deploy:

- `me_block`
- `memory_image`
- legacy dual-image ACT deploy variants

## Export

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

Exported files:

- `act_inference.pt`
- `deploy_config.yml`

## deploy_config.yml

The exported config now only stores baseline preprocessing and decoding fields:

- `target_width`
- `target_height`
- `pad_left`
- `pad_top`
- `depth_clip_min`
- `depth_clip_max`
- `state_dim`
- `num_queries`
- `image_channels`
- `predict_delta_qpos`
- `delta_qpos_scale`

## TorchScript wrapper

Wrapper implementation:

- [deploy_wrappers.py](./deploy_wrappers.py)

Responsibilities:

- normalize raw `qpos`
- convert BGRA images into model input tensor order
- decode model output actions back to physical joint space
