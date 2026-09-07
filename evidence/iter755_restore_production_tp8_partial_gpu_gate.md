# Iteration 755: production restore and partial-GPU TP8 gate

Date: 2026-09-08

## Selected source

The active implementation was restored from commit `549d6b9` and committed
as `04a73bc` after the iter754 rejection.

```text
136757753077f62e9ea0be6971f4360296f27552c4d1957eb3d4b737a186474c  v4_flash_tp_tile_ws_body.inl
756e58862f4da544031b6b12a7c6b042e8ea8f0e3a2c9f4c5184500caaa48a86  v4_flash_tp_native_megamoe.py
63bcc6a144cbcab7482e95fd5263ebb382a78c52334ec91ed01145377d02bb82  v4_flash_tp_tile_ws_megamoe.py
```

## TP8 local-shape run-through

The test used physical GPU2 and `--tp 8`, selecting I=256.  This invokes the
same W13 -> shared FP8 requantization -> W2 -> local fixed-k6 body while
skipping the unavailable eight-rank communication tail.

```text
NATIVE_LOCAL_PROFILE_ONLY {"finite": true, "m": 8, "max_abs": 69632.0, "synchronized": true, "tp": 8}
NATIVE_LOCAL_PROFILE_ONLY {"finite": true, "m": 16, "max_abs": 69632.0, "synchronized": true, "tp": 8}
NATIVE_LOCAL_PROFILE_ONLY {"finite": true, "m": 32, "max_abs": 99328.0, "synchronized": true, "tp": 8}
NATIVE_LOCAL_PROFILE_ONLY {"finite": true, "m": 64, "max_abs": 148480.0, "synchronized": true, "tp": 8}
NATIVE_LOCAL_PROFILE_ONLY {"finite": true, "m": 128, "max_abs": 272384.0, "synchronized": true, "tp": 8}
```

Endpoint diagnostic:

```text
M8: finite=true, l1_x_mismatch_bytes=0, l1_sf_max_abs=0,
    l1_weight_max_abs=0, w2_partials_finite=true,
    w2_partials_shape=[2,6,128,4096]
M128: finite=true, w2_partials_finite=true,
      w2_partials_shape=[2,6,128,4096]
```

The M128 diagnostic's input-order route-row comparison is invalid for the
parallel atomic route builder because duplicate routes to an expert may claim
rows in any order.  It is not used as a correctness result.

## Availability limitation

`nvidia-smi` showed GPUs0 and1 at roughly 84 GiB allocated while only two
408-MiB GPU0 Python processes were container-visible.  Resetting or consuming
the remaining headroom would be unsafe.  Therefore this iteration does not
claim a fresh TP8 all-reduce or cold-L2 latency result.
