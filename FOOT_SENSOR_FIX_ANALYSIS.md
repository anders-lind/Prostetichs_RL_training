# Foot Sensor Issue - Root Cause Analysis and Fix

## Problem Summary
When running `run_policy_eval.py` on trained models, the gait analysis produces a warning:
```
Warning! Not enough gait data to plot. Skipping plotting.
```

This occurs because the `r_foot` and `l_foot` sensors in the `gait_evaluated_data.json` files contain only zeros (0 non-zero values out of 200+ timesteps), while other sensors like `r_toes` and `l_toes` work correctly (69-77 non-zero values out of 200 timesteps).

## Root Cause Investigation

### Finding 1: Sensor Data Collection
Using the model during evaluation:
- **r_foot**: 200 entries, **0 non-zero** values (>0.1)
- **l_foot**: 200 entries, **0 non-zero** values (>0.1)  
- **r_toes**: 200 entries, **69 non-zero** values (>0.1) ✓ Working
- **l_toes**: 200 entries, **77 non-zero** values (>0.1) ✓ Working

### Finding 2: Sensor Configuration Comparison

Both r_foot and r_toes are touch sensors, but they have different configurations:

| Property | r_foot_touch | r_toes_touch | Difference |
|----------|--------------|--------------|-----------|
| **Position** | 0.01 -0.002 -0.01 | 0.0 -0.002 0.01 | Located in calcn (heel) vs toes body |
| **Size** | .03 .02 .03 | .05 .025 .075 | **FOOT IS 67% SMALLER** |
| **Type** | box | box | Same |
| **Rotation (Euler)** | 0 0 0 | 0 1.5 0 | Toes rotated 1.5 rad on Y-axis |

### Root Cause

The `r_foot_touch` and `l_foot_touch` sensor sites are **too small** to effectively detect ground contact. During walking:

1. The foot sensor site is only 0.03m (30mm) long along the foot direction (x-axis)
2. The actual ground contact surface of the foot is much larger
3. The small box size means most ground contact happens outside the sensor zone
4. The toe sensors work because they're positioned on the toes body which makes distinct contact during toe-off phase

### Model File Locations
- `/models/22muscle_2D/myoLeg22_2D_OPENEXO.xml` (Main model used in training)
- `/models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml` (Reference model)

## Solution

### Changes Made
Increased the foot sensor site sizes from `.03 .02 .03` to `.09 .02 .05`:

**Before:**
```xml
<site name="r_foot_touch" type="box" pos="0.01 -0.002 -0.01" size=".03 .02 .03" euler="0 0 0" class="myo_leg_touch"/>
<site name="l_foot_touch" type="box" pos="0.01 -0.002 0.01" size=".03 .02 .03" euler="0 0 0" class="myo_leg_touch"/>
```

**After:**
```xml
<site name="r_foot_touch" type="box" pos="0.01 -0.002 -0.01" size=".09 .02 .05" euler="0 0 0" class="myo_leg_touch"/>
<site name="l_foot_touch" type="box" pos="0.01 -0.002 0.01" size=".09 .02 .05" euler="0 0 0" class="myo_leg_touch"/>
```

### Scaling Rationale
- **X-axis** (0.03 → 0.09): **3x increase** - Covers more of the heel and midfoot area during contact
- **Y-axis** (0.02 → 0.02): **No change** - Vertical extent already appropriate
- **Z-axis** (0.03 → 0.05): **1.67x increase** - Extends across the foot width

This brings the foot sensor sites closer in scale to the toe sensors while still being appropriate for the heel area.

## Impact

By increasing the foot sensor site sizes, the model's ground reaction force (GRF) sensors will now properly detect:
- Heel strike during foot contact
- Midfoot pressure during stance phase
- Better temporal resolution of the gait cycle

This will enable:
- Proper gait data analysis with non-zero r_foot/l_foot values
- Meaningful biomechanical analysis of the walking pattern
- Generation of realistic gait plots showing foot contact patterns

## Files Modified
1. `/models/22muscle_2D/myoLeg22_2D_OPENEXO.xml`
2. `/models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml`

## Testing the Fix

To verify the fix works:

1. Run the policy evaluator on a trained model:
   ```bash
   python rl_train/run_policy_eval.py A2C_train_1/train_session_20251124-121752_A2C_1_2/
   ```

2. Check the gait_evaluated_data.json to see non-zero values for r_foot/l_foot:
   ```bash
   python debug_foot_sensors_simple.py A2C_train_1/train_session_20251124-121752_A2C_1_2/
   ```

3. Expected output should show:
   ```
   r_foot: 200 entries, [X] non-zero (>0.1)  ← Should no longer be 0
   l_foot: 200 entries, [X] non-zero (>0.1)  ← Should no longer be 0
   ```

## Why This Wasn't Caught Before

The sensor configuration was set up to "add foot sensors for GRF measurements" (as noted in XML comments), but the sites were sized identically to the heel reference sites (`r_heel_btm` and `r_toe_btm` which use size="0.01" for visualization purposes, not sensing). The toe sensors worked better because they were explicitly configured with larger boxes positioned on the more active toes body, which moves distinctly during the swing phase of gait.
