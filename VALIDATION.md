# BL-05 validation record

Validation date: 2026-07-10 on the target HP OMEN laptop.

## Passed

- JavaScript syntax checks for `physics-model.js`, `shaders.js`, `webgpu-engine.js`, and `app.js` using Node 24.14.0.
- Physics-reference and numerical-invariant tests:
  - the enthalpy curve is strictly monotone and its inverse round-trips from −50 to 300 °C;
  - the compact GPU inverse stays within its stated interpolation tolerance;
  - all three live grids initialize to 135.00 g and a consistent measured footprint;
  - aggregate donor scaling keeps a four-face adversarial update nonnegative and mass-conservative;
  - horizontal heat exchange is pairwise conservative and zero across a wet/dry face;
  - nonlinear contact exchange is monotone and equal-and-opposite from 0.1 µm to 20 mm films;
  - butter plus substrate solar absorption never exceeds incident irradiance;
  - layered transition telemetry does not collapse the eight-layer state to mean temperature.
- Static interface contracts: 45 unique HTML IDs, keyboard/focus/ARIA hooks, required files, three-pass flow, implicit boundary/contact paths, and local-server hardening are present.
- Local server checks: GET and HEAD return the correct MIME types; POST returns 405; missing or foreign Host returns 403; malformed NUL paths return 400; encoded traversal returns 404; CSP, `nosniff`, referrer, COOP, and CORP headers are present.
- Every live WGSL entry point passed semantic validation and HLSL lowering with official Dawn/Tint `v20260423.175430`:
  - `computeRawFaces`
  - `computeDonorScale`
  - `simulate`
  - `backgroundVertex`
  - `backgroundFragment`
  - `butterVertex`
  - `butterWallVertex`
  - `butterFragment`
- All eight lowered entry points passed Shader Model 6.0 validation through Chrome 150's `dxcompiler.dll`.
- The complete 1600×900 interface was rendered and visually inspected through the `?ui-preview` QA route after the realism pass. The butter now uses a flat cut-block silhouette, restrained pale material, subtle knife texture, vertical faces, and an anchored contact shadow rather than the earlier playful wedge.

## Environment limitation

Current Chromium headless processes on this Windows 26H1 preview build do not expose the installed D3D12 WebGPU adapter reliably. The final isolated Edge headless attempt reached the app's explicit “No compatible WebGPU adapter” path; this is a browser/headless-adapter limitation, not a shader parse failure. The installed interactive Chrome/Edge builds, GTX 1660 Ti, driver 592.27, and Direct3D 12 are the intended runtime.

The visible interactive browser profile was not modified during automated QA. The one-click launcher is therefore the final real-hardware acceptance check: it reports either `WebGPU · n fps` (plus `flow ×…` when the adaptive timestep is active) or a focused, retryable initialization error. Shader semantics and DirectX lowering were validated independently.

## Modeling limits

The current flow solve is conservative and positivity-preserving, but it is an educational shallow-volume approximation. Explicit capillary pressure/contact-angle dynamics are intentionally disabled until an implicit solver and convergence criterion are added. The engine exposes limiter activity and advances the simulation clock only by time actually computed; it does not silently claim an unresolved accelerated timestep.

## Reproduction commands

Use a current Node.js from the project directory:

```powershell
node .\tests\physics.test.mjs
node .\tests\math.test.mjs
node .\tests\static.test.mjs
node .\tests\export-shaders.mjs
```

Generated WGSL snapshots are in `artifacts/wgsl/`. They were checked with Dawn/Tint and Chrome DXC, not with a hand-written parser.
