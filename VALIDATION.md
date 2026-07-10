# BL-05 validation record

Validation date: 2026-07-10 on the target HP OMEN laptop.

## Passed

- JavaScript syntax checks for `physics-model.js`, `shaders.js`, `webgpu-engine.js`, and `app.js` using Node 24.14.0.
- Physics-reference and numerical-invariant tests:
  - the enthalpy curve is strictly monotone and its inverse round-trips from −50 to 300 °C;
  - the compact GPU inverse stays within its stated interpolation tolerance;
  - all three live grids initialize to 135.00 g and a consistent measured footprint;
  - aggregate donor scaling keeps a four-face adversarial update nonnegative and mass-conservative;
  - mobile depth grows layer by layer instead of switching the whole 19 mm block to liquid flow;
  - bounded face exchange cannot reverse a height jump or turn one peak into a checkerboard trough;
  - receiver scaling prevents four simultaneous donors from creating a new local maximum;
  - mobile-layer enthalpy weights sum to one and conserve their shared-face energy flux;
  - horizontal heat exchange is pairwise conservative and zero across a wet/dry face;
  - nonlinear contact exchange is monotone and equal-and-opposite from 0.1 µm to 20 mm films;
  - butter plus substrate solar absorption never exceeds incident irradiance;
  - layered transition telemetry does not collapse the eight-layer state to mean temperature.
- Static interface contracts: 45 unique HTML IDs, keyboard/focus/ARIA hooks, required files, three-pass flow, implicit boundary/contact paths, and local-server hardening are present.
- Local server checks: GET and HEAD return the correct MIME types; POST returns 405; missing or foreign Host returns 403; malformed NUL paths return 400; encoded traversal returns 404; CSP, `nosniff`, referrer, COOP, and CORP headers are present.
- Every live WGSL entry point passed semantic validation and HLSL lowering with official Dawn/Tint `v20260423.175430`:
  - `computeRawFaces`
  - `computeDonorScale`
  - `computeReceiverScale`
  - `simulate`
  - `backgroundVertex`
  - `backgroundFragment`
  - `butterVertex`
  - `butterWallVertex`
  - `butterFragment`
- All nine lowered entry points passed Shader Model 6.0 validation through Chrome 150's `dxcompiler.dll`.
- The complete 1600×900 interface was rendered and visually inspected through the `?ui-preview` QA route after the realism pass. The butter now uses a flat cut-block silhouette, restrained pale material, subtle knife texture, vertical faces, and an anchored contact shadow rather than the earlier playful wedge.
- The corrected compute and render paths were exercised interactively on the GTX 1660 Ti in Chrome 150 with the Warm pan preset at 60×. At 01:47 simulated time the block showed a smooth basal puddle at 16% transition; at 05:54 it formed a continuous late-stage puddle at 91% transition. Mass remained 100.00%, all pipelines stayed live, and the browser console reported no warnings or errors. The former cell-pillar/vertical-fence failure did not recur.

## Interactive hardware acceptance

Chrome 150 created the D3D12 adapter, all four compute pipelines, all render pipelines, and 4× MSAA Photo pipelines on the target GTX 1660 Ti. Interactive testing reached 44 fps during the accelerated mid-melt inspection; the most computationally expensive 91% liquid state remained responsive while the test harness captured telemetry and a full-frame image.

## Modeling limits

The current flow solve is conservative, positivity-preserving, and receiver-bounded, but it remains an educational shallow-volume approximation. Explicit capillary pressure/contact-angle dynamics are intentionally disabled until an implicit surface-energy solver and convergence criterion are added. A compact render reconstruction smooths cell-averaged geometry; it does not claim to be a capillary physics solve. The engine exposes limiter activity and advances the simulation clock only by time actually computed.

## Reproduction commands

Use a current Node.js from the project directory:

```powershell
node .\tests\physics.test.mjs
node .\tests\math.test.mjs
node .\tests\static.test.mjs
node .\tests\export-shaders.mjs
```

Generated WGSL snapshots are in `artifacts/wgsl/`. They were checked with Dawn/Tint and Chrome DXC, not with a hand-written parser.
