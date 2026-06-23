# SmolVLA attention comparison (offline)

This tool compares the action expert's attention to three camera token groups, state, and language. It does
not produce spatial heatmaps inside an image. Displayed percentages are normalized over the five semantic
groups; the JSONL files also retain unnormalized attention mass and every layer/denoise-step measurement.

Defaults are fixed to the locally cached experiment:

- Dataset: `wuc1/bi_so101_ffp_0615-14-12-dagger_merged`
- Models: `bi_so101_ffp_0615-14-12-dagger_merged3cam_discrete_state`,
  `bi_so101_ffp_0615-14-12_merged`, and `bi_so101_ffp_0615-14-12-dagger_merged3cam`
- Episode: longest (`232`, 10,280 frames at 60 FPS)
- Attention sampling: every 12 frames (5 Hz), full 10-step inference
- Video: all 10,280 source frames at 60 FPS, with sampled attention forward-filled

Run the complete longest episode from the repository root:

```bash
uv run --extra smolvla python -m attention_visualization
```

The command sets Hugging Face and Transformers offline modes internally. It fails immediately when a model,
the SmolVLM backbone, dataset metadata, Parquet file, or source video is absent from local storage. It never
downloads missing files.

Outputs are written to:

```text
outputs/attention_visualization/episode_000232/
├── comparison.mp4
├── run_manifest.json
├── bi_so101_ffp_0615-14-12-dagger_merged3cam_discrete_state.attention.jsonl
├── bi_so101_ffp_0615-14-12_merged.attention.jsonl
└── bi_so101_ffp_0615-14-12-dagger_merged3cam.attention.jsonl
```

Sampling is resumable by default. Re-run the same command after interruption; existing sampled frame indices
are skipped. Useful phase controls are:

```bash
# One-frame-per-model integration check (does not render a video)
uv run --extra smolvla python -m attention_visualization --phase sample --max-samples 1

# Limit both attention sampling and the comparison video to the first 10 seconds
uv run --extra smolvla python -m attention_visualization --duration-seconds 10

# Render after every scheduled sample exists
uv run --extra smolvla python -m attention_visualization --phase render --duration-seconds 10
```

`--force` removes outputs for this episode before starting. `--overwrite-video` replaces only the MP4.
Models are loaded one at a time with compilation disabled and released before the next model is loaded.

The current revisions used by the three-model comparison have three camera keys and 12-dimensional bi-arm
state. The two continuous-state checkpoints have an independent projected state token. In the discrete-state
checkpoint, state values are encoded inside the language prompt; the entire prompt is therefore attributed to
Language, and State is shown as N/A because no independent state token exists.

## Merged policy: three cameras versus four cameras

The local cache also contains historical revision `b2e0537` of
`wuc1/bi_so101_ffp_0615-14-12_merged`. Unlike current revision `743efd6`, that checkpoint was trained with
four cameras, including `observation.images.left_camera3`. The episode metadata and local video files retain
that fourth stream even though the current dataset `info.json` no longer advertises it.

Run the full two-row comparison without replacing the existing three-model result:

```bash
HF_DATASETS_CACHE=/tmp/hf-datasets-attn uv run --extra smolvla \
  python -m attention_visualization.merged_camera_comparison
```

This reuses the existing three-camera JSONL when complete, samples only the four-camera checkpoint when it is
missing, and writes:

```text
outputs/attention_visualization/episode_000232/
├── merged_three_vs_four_camera.mp4
├── merged_camera_comparison_manifest.json
└── bi_so101_ffp_0615-14-12_merged.four_camera.attention.jsonl
```

Use `--force-four-camera` to replace only the four-camera samples and `--overwrite-video` to replace only this
comparison video. The original `comparison.mp4` and three-camera attention JSONL are preserved.
