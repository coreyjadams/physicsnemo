# GeoTransolver: invariances, geometry encoder, and boundary-condition encoder — scoping

Based on `main` @ `cc444416`. Code read: `physicsnemo/models/geotransolver/`
(`geotransolver.py`, `context_projector.py`, `flare_plus_plus.py`),
`physicsnemo/nn/module/gale.py`, `physicsnemo/nn/module/physics_attention.py`,
and the unified external-aero recipe
(`examples/cfd/external_aerodynamics/unified_external_aero_recipe`).

Related roadmap issues (NVIDIA/physicsnemo-roadmap):

- #2605 New Layers + Architecture Improvements for Generalizability (GeoT/GLOBE combo)
- #1416 Encoding boundary conditions in DoMINO (spatially varying BCs still open)
- #981 Physics-Informed Latent Spaces for Improved Model Generalization
- #1485 GLOBE rotation-invariant encoding of vector-valued BCs
- #2897 GeoTransolver checkpoint equivalent to DoMINO NIM

## How data flows today

- **Per-point input:** raw `xyz` + normals (surface) or `xyz` + SDF + SDF
  normals (volume). An MLP projects it to the hidden size.
- **"Geometry" input:** in both recipe configs this is `interior.points`, the
  sampled query cloud itself. It is not the car. In the volume case the car's
  STL is dropped after the SDF is computed (`DropBoundary`), so the body shape
  reaches the model only through the per-point SDF.
- **Global input:** `U_inf`, as one token, run through the same slice projector
  (`ContextProjector`) as geometry.
- **Context:** geometry tokens, global tokens and optional ball-query tokens are
  joined and fed to every GALE block by cross-attention.
- **Symmetry today:** only augmentation, meaning random z-rotation (which also
  rotates `U_inf`) and a ±1 xy shift.

## What stands in the way

1. **Nothing is built in for translation or rotation.** Raw coordinates go into
   the per-point MLP, into the geometry tokenizer, and into the ball-query
   features (absolute neighbor coordinates, not offsets). `U_inf` is treated as
   three separate scalars. Augmentation only makes the model roughly symmetric
   over the rotations it was trained on.

2. **The geometry tokens depend on where the mesher put points.** (Ran this.)
   - Slice pooling (`_compute_slices_from_projections`) is a plain average over
     points, so densely meshed regions count more.
   - Taking a different random sample at the same density moved the tokens by
     **0.2%**. Refining one region of the same domain moved them by **17%**.
     (Synthetic test on `ContextProjector`, not a real mesh.)
   - Volume meshes are much denser near the wall, so what the tokens really
     encode is the mesh refinement pattern.
   - Inference on a different query layout (a uniform grid, a different N)
     silently changes the context.
   - #2605 says physics attention "naturally gives" discretization invariance.
     That only holds when the sampling density stays the same.

3. **The ball-query features also depend on sampling.**
   - The search keeps the first K points it finds, not the nearest K.
   - Missing neighbors are filled with zeros. That is the origin, and after
     centering the origin is a real location (the car's center).
   - The K neighbors are concatenated in search order into one long vector, so
     the result depends on order and density.
   - A separate small bug: `MultiScaleFeatureExtractor.extract_local_features`
     calls `processor(geometry, spatial_coords)`, the opposite argument order
     from `extract_context_features`. It only works because the recipe passes
     the same tensor for both.

4. **The boundary-condition encoder carries almost nothing.** (Ran this.) A
   single `U_inf` token through `ContextProjector` comes out rank-1: singular
   values are 1, then about 6e-8. All `slice_num` "global tokens" are scaled
   copies of one linear projection of `U_inf`. The scale also depends on
   `slice_num` through the fixed `1e-2` term in the slice normalization. There
   is no way to express boundary-condition type, values that vary across a
   boundary, or which boundary they apply to.

## What "invariance" should mean here

The physical symmetry applies to the whole problem. If you move or rotate the
geometry, the boundary conditions and the query points together, scalar outputs
(p, nut) should stay the same and vector outputs (U) should rotate with the
problem. Anything that breaks the symmetry, such as the ground plane, gravity or
tunnel walls, has to be an input to the model. Once it is, full rotation and
translation symmetry is correct even for a car on the ground. If such an input
is missing, enforcing exact symmetry would hurt.

## Options, roughly by cost

### Translation (cheap)

- Express positions relative to an origin taken from the geometry, such as the
  area-weighted centroid of the body surface.
- Use relative offsets in the ball query.

### Rotation

- **A. Canonical frame (recommended first).** Build a frame from the boundary
  conditions: the flow direction plus the ground normal or gravity, falling back
  to the body's principal axes. Rotate inputs into that frame, run the model
  unchanged, and rotate the vector outputs back.
  - The symmetry is exact.
  - It works with every attention backend, Transformer Engine and sharding.
  - The weak point is a degenerate frame (for example, flow parallel to the
    ground normal). For external aero the boundary conditions almost never
    produce one; averaging over sign flips is the fallback.
- **B. Invariant inputs, vector outputs rebuilt from a local basis (the GLOBE
  approach, #2605).** The model only ever sees invariant per-point features.
  Vectors are output as coefficients on a local basis built from `U_inf`, the
  normal, the SDF gradient and their cross product.
  `physicsnemo/nn/functional/equivariant_ops.py` already has the helpers.
  - No frame needs to be chosen.
  - The attention loses absolute position and needs relative-position handling.
    That is real research risk.
- **C. Fully equivariant attention** (tensor products, as in SE(3)-Transformers
  or Equiformers). Skip it: the cost is high, and A or B already give the
  guarantee.

### Discretization

- Pass a per-point measure (cell area or volume, scaled by the subsample ratio)
  and weight the slice pooling and context projectors by it. This is one central
  change in `_compute_slices_from_projections`, and it can default to off. The
  output side (slices back to points) is already fine.
- Rewrite the ball query to aggregate neighbors by a measure-weighted mean of
  relative offsets, with padded slots masked out.

### Geometry encoder

- Encode the body surface, sampled by area, separately from the query points:
  position (in the canonical frame), normals, perhaps curvature, pooled with the
  measure weights into tokens.
- Query points keep the SDF and the offset to the nearest surface point, and
  ball-query the surface rather than each other.
- This is also where shared "geometry encoder" work from #2605 or #2897 could
  plug in.

### Boundary-condition encoder

- **Global conditions:**
  - Convert to dimensionless numbers (Re, Mach).
  - `U_inf`'s direction defines the frame, and its magnitude is folded into the
    dimensionless numbers.
  - Condition every block with adaLN or FiLM (the global condition sets a
    per-layer scale and shift), or with an MLP that produces several distinct
    tokens, instead of the rank-1 projector.
- **Boundary values that vary over space (#1416):**
  - `DomainMesh.boundaries` is already a dict keyed by named patches.
  - Give each boundary face its type (wall / inlet / outlet / symmetry /
    ground), its values, its normal and its measure.
  - Tokenize those faces as their own context stream.
- **"Trainers":**
  - A dataset where the boundary conditions actually vary. Otherwise the encoder
    can't learn them.
  - Optionally, enforce wall conditions exactly in the output, e.g. multiply
    velocity by an SDF-based factor so it is zero at the wall.

## Proposed order

0. **Invariance test suite first**, run against the current model as a baseline.
   It covers translation, rotation, local refinement, changing N, and shuffling
   point order, measured on the fields and on integrated forces. It's cheap, and
   every later step gets judged against it.
1. **Discretization:** measure-weighted pooling, the ball-query rewrite and
   fixing the argument swap. Opt-in flags, so old checkpoints still load.
2. **Body-surface geometry encoder**, separate from the query points.
3. **Canonical-frame wrapper** at model level, so inference users get the
   guarantee and not only training.
4. **Boundary-condition encoder:** typed global conditioning, then
   boundary-patch tokens, plus a dataset with varying boundary conditions.
5. **Research branch:** option B, to fold into #2605.

## Open questions

- **"Trainers":** training recipes and datasets with varying boundary
  conditions, losses that enforce the boundary conditions, or both? Which
  dataset has real boundary-condition variation (HiLift angle of attack?
  something internal)?
- **Rotation:** canonical frame first (exact and cheap), or go straight to the
  GLOBE-style invariant version?
- **Packaging:** extend `GeoTransolver` behind flags, or build a new model under
  `experimental/` toward the #2605 "Mesh Transformer"?

## Caveats

Only the code was read and small CPU checks run on the context projector.
Nothing was trained, and the accuracy gain from each fix is unmeasured. The 17%
figure comes from a synthetic refinement test, not a real mesh. Exact symmetry
will hurt if any symmetry-breaking input (ground plane, tunnel walls) is left
out of the model inputs.
