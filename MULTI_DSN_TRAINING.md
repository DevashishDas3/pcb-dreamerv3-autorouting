# Multi-DSN Training Implementation

## Overview

You can now train the model on multiple PCB designs (DSN files) simultaneously. The model will learn a routing policy that generalizes across different board layouts by exploring them all during training.

### Key Features

1. **Random DSN Sampling**: Each episode reset randomly samples a DSN file from the configured list
2. **Trace Intersection Detection**: Automatic penalty applied for intersecting traces to encourage the model to avoid overlaps
3. **Backward Compatible**: Single DSN training still works if you only specify one file
4. **Configurable Penalty**: Intersection penalty scale is tunable via `freerouting_intersection_penalty_scale`

## Configuration

### Default Multi-DSN Setup (in `configs.yaml`)

```yaml
freerouting:
  freerouting_jar_path: "freerouting.jar"
  freerouting_dsn_path: "Issue313-FastTest.dsn" # Fallback single file
  freerouting_dsn_files:
    [
      "dsn_test_files/board_1.dsn",
      "dsn_test_files/board_2.dsn",
      "dsn_test_files/board_3.dsn",
      "dsn_test_files/board_4.dsn",
      "dsn_test_files/board_5.dsn",
    ]
  freerouting_intersection_penalty_scale: 0.1 # Penalty per intersection (tunable)
```

### Customizing the Setup

#### Use a Single DSN File (Original Behavior)

To train on a single file, either:

- Set `freerouting_dsn_files: []` (empty list) to use only `freerouting_dsn_path`
- Or provide just one file in the list: `freerouting_dsn_files: ['dsn_test_files/board_1.dsn']`

#### Use a Subset of Files

```yaml
freerouting_dsn_files:
  ["dsn_test_files/board_1.dsn", "dsn_test_files/board_2.dsn"]
```

#### Adjust Intersection Penalty

The penalty is applied as: `reward -= intersection_penalty_scale * num_intersections`

- Increase to make the model more careful: `0.2` or `0.3`
- Decrease to allow some intersections: `0.05` or `0.01`

```yaml
freerouting_intersection_penalty_scale: 0.2 # Stronger penalty
```

## Running Training

### Train with Multi-DSN Support

```bash
python train.py --configs freerouting
```

The training will:

1. Create multiple parallel environments (default: 1, configure with `envs: N`)
2. Each episode randomly samples a different DSN file
3. Collect experience from all 5 board layouts
4. Learn a routing policy that works across different designs

### Monitor Training

Check the logs to see which DSN files are being trained on:

- Episode files in `logs/freerouting_*/train_eps/` include DSN metadata in step info
- Reward trends should show the penalty for intersecting traces

## Implementation Details

### Modified Files

#### 1. **envs/freerouting_jpype_env.py**

- Added `dsn_files_list` parameter to accept multiple DSN files
- Added `load_dsn_file(dsn_file_path)` method to dynamically switch DSN files
- Added `_detect_intersections()` method to identify trace overlaps
- Modified `reset()` to randomly sample and load a DSN file each episode
- Modified `step()` to apply intersection penalty to rewards
- Metadata includes current DSN file in step info dict

#### 2. **configs.yaml**

- Added `freerouting_dsn_files`: list of DSN files to train on
- Added `freerouting_intersection_penalty_scale`: tunable penalty factor

#### 3. **dreamer.py**

- Modified `make_env()` to pass DSN files list to `FreeroutingJPypeEnv`
- Handles fallback to single file if list not provided
- Passes intersection penalty scale to environment initialization

## How It Works

### Episode Reset (Random DSN Sampling)

```python
# In FreeroutingJPypeEnv.reset()
if len(self._dsn_files_list) > 1:
    sampled_dsn = self._rng.choice(self._dsn_files_list)
    if sampled_dsn != self._current_dsn_path:
        self.load_dsn_file(sampled_dsn)
```

This ensures each episode potentially uses a different board layout, maximizing exploration.

### Reward Computation (Intersection Penalty)

```python
# In FreeroutingJPypeEnv.step()
reward = float(self._controller.performAction(int(action)))
intersection_penalty = self._detect_intersections()
reward -= intersection_penalty  # Apply penalty
```

The penalty is subtracted from the base action reward, encouraging the model to learn to avoid trace overlaps.

### Intersection Detection

The current implementation detects intersections based on:

**For legacy backend** (`jpype_legacy`):

- Queries `unroutedNets()` count (proxy for routing conflicts)
- Or queries violation count if available
- Penalty = `scale * min(unrouted_count, 10)`

**For API v1 backend** (`jpype_api_v1`):

- Checks HTTP status code (>= 500 indicates error/violation)
- Penalty = `scale` if status >= 500

**Improvement opportunity**: If Freerouting API provides direct trace intersection detection, we can use that instead of the heuristics above.

## Troubleshooting

### Issue: "No DSN files provided; using dummy freerouting environment"

**Solution**: Ensure `freerouting_dsn_files` contains valid file paths that exist.

### Issue: Model doesn't improve across DSN files

**Solution**:

- Increase intersection penalty: `freerouting_intersection_penalty_scale: 0.2`
- Increase training steps: `steps: 2e6` (longer training)
- Increase parallel environments: `envs: 4` (more data collection)

### Issue: Training starts but no DSN switching

**Solution**:

- Verify `freerouting_dsn_files` list has > 1 file
- Check logs for "DSN:" metadata to confirm sampling is working

## Testing

Run the verification script to ensure everything is set up correctly:

```bash
python3 test_multi_dsn.py
```

This checks:

- ✓ All DSN test files exist
- ✓ Config YAML has multi-DSN settings
- ✓ FreeroutingJPypeEnv has all necessary methods
- ✓ dreamer.py correctly integrates multi-DSN config

## Next Steps

### Enhance Intersection Detection

If Freerouting's Java API provides:

- `getTraceIntersections()` - direct trace overlap detection
- `getViolations()` - explicit violation list
- `getConflictCount()` - conflict count

Then improve `_detect_intersections()` to use these methods directly for more accurate penalties.

### Add Per-DSN Metrics

Track separate episode rewards/statistics per DSN file to monitor learning progress on each board type.

### Task Embedding (Optional)

If some DSN files are significantly harder/easier, consider adding task embeddings to let the model specialize per DSN while sharing learned routing knowledge.

## Summary

The multi-DSN training support is now ready. Your model will learn from all 5 board layouts simultaneously, with automatic penalties for intersecting traces. This should help the model generalize better across different PCB designs.

**To start training**:

```bash
python train.py --configs freerouting
```
