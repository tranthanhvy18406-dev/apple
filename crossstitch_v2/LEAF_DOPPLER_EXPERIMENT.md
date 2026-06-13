# LEAF Doppler Leave-One-Out Experiment

This experiment follows the requested Doppler set:

```text
D = {5, 20, 50, 100, 200, 300, 400, 500} Hz
```

For each target Doppler, the other seven Dopplers are used as source domains.
The target test trace is split by window index in time order:

- support set: first `K=128` windows
- gap: `max(horizons)=10` windows
- query set: windows after the support/gap boundary

Each window uses the existing direct-horizon protocol:

- input: past 16 frames
- output: future horizons `h=1..10`, i.e. `10..100 ms`
- metric: throughput MAE, plus RB/SINR diagnostics

With `test_1000_HDF5.pkl`, `hist_window=16`, and `max_horizon=10`, each target
has `1000 - 16 - 10 + 1 = 975` windows. Therefore, the default query set starts
at window `128 + 10 = 138` and has `975 - 138 = 837` windows.

## Methods

The implementation is in:

```bash
code/Train/run_leaf_doppler_experiment.py
```

It compares:

1. `pooled`: source-only pooled Cross-Stitch. Trains on the seven source
   Doppler `train_9000_HDF5.pkl` files and evaluates the target query directly.
2. `pooled_finetune`: starts from the same pooled checkpoint, then fine-tunes
   all Cross-Stitch parameters on the target support set only.
3. `leaf_crossstitch`: starts from the pooled Cross-Stitch checkpoint, then
   trains a LEAF-style latent extrapolation and sample-adjustment module on the
   source Dopplers. At target time, it initializes a latent from target Doppler
   frequency, adapts that latent on the same 128 support windows, and evaluates
   the query.
4. `target_oracle`: trains the Cross-Stitch model on the target Doppler
   `train_9000_HDF5.pkl` with a time-ordered validation split, then evaluates
   the target query. This is an upper-bound reference, not a fair few-shot
   method.

The LEAF implementation mirrors the paper/code structure at the level needed
for this protocol:

- latent-space model adaptation
- an extrapolator that maps Doppler context to a latent initialization
- a decoder from latent to task-specific Cross-Stitch output adapters
- a meta-learned sample adjustment network used before query prediction

## Full Run

Run from `/4T/xty/crossstitch_v2`:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python code/Train/run_leaf_doppler_experiment.py \
  --output_dir leaf_doppler_leave_one_out \
  --target_dopplers all \
  --methods pooled,finetune,leaf,oracle \
  --support_size 128 \
  --device cuda:0
```

The script is resumable. If `metrics.json` and checkpoints already exist for a
method, they are reused unless `--force` is passed.

## Low-To-High Hard Split

A harder Doppler extrapolation split can be run with fixed low-Doppler source
domains and high-Doppler targets:

```text
train/source: 5, 20, 50, 100 Hz
test/target: 300, 400, 500 Hz
```

Run:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python code/Train/run_leaf_doppler_experiment.py \
  --output_dir leaf_doppler_low_to_high_hard_split_shared \
  --source_dopplers 5,20,50,100 \
  --target_dopplers 300,400,500 \
  --methods pooled,finetune,leaf,oracle \
  --support_size 128 \
  --device cuda:0
```

When `--source_dopplers` is set, source-side checkpoints are shared across all
targets in a directory such as `source_5_20_50_100Hz/`. Target-specific
support adaptation and query metrics are still written under `target_300Hz/`,
`target_400Hz/`, and `target_500Hz/`.

## Faster Debug Run

Use this to validate the pipeline without waiting for full training:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python code/Train/run_leaf_doppler_experiment.py \
  --output_dir /tmp/crossstitch_leaf_debug \
  --target_dopplers 500 \
  --methods pooled,finetune,leaf \
  --horizons 1,2 \
  --support_size 16 \
  --max_source_windows 64 \
  --epochs 1 --patience 1 \
  --finetune_epochs 1 \
  --leaf_epochs 1 --leaf_patience 1 \
  --leaf_support_size 16 --leaf_query_size 16 --leaf_num_task_segments 1 \
  --leaf_inner_steps 1 --leaf_target_inner_steps 1 \
  --batch_size 16 --eval_batch_size 64 \
  --rb_hidden_dim 16 --sinr_hidden_dim 8 --rb_layers 1 --sinr_layers 1 \
  --head_hidden_dim 16 --rb_head_hidden_dim 16 --sinr_head_hidden_dim 16 \
  --horizon_embed_dim 4 \
  --leaf_latent_dim 8 --leaf_hidden_dim 16 \
  --device cpu --force
```

## Outputs

The full run writes:

```text
leaf_doppler_leave_one_out/
  summary_by_target_method.csv
  summary_average_by_method.csv
  target_5Hz/
    pooled/
    finetune/
    leaf/
    oracle/
  ...
```

Each method directory contains:

- `metrics.json`
- `per_horizon_metrics.csv`
- checkpoints and training history where applicable
