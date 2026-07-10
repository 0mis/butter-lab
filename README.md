# Butter Lab BL-05

**Live experiment:** [https://0mis.github.io/butter-lab/](https://0mis.github.io/butter-lab/)

BL-05 is a GPU-accelerated butter-melting laboratory tuned and validated on this computer:

- HP OMEN 15-dc1xxx
- Intel Core i7-9750H (6 cores / 12 threads)
- NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM, driver 592.27
- 16 GB RAM and a 1920×1080 display
- Chrome 150 or Edge 149

The app uses standards-based WebGPU: Direct3D 12 on this Windows validation machine and the browser's native Metal backend on current iPhones. It does not require Node, Python, Unreal, Unity, or CUDA Toolkit at runtime.

## Launch

Double-click **Launch Butter Lab.cmd**. A small loopback-only server starts at `http://127.0.0.1:4178/` and opens the lab in installed Chrome (or Edge as a fallback). Keep the terminal window open while using the app; close it or press `Ctrl+C` to stop the server.

If another program already uses port 4178:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch.ps1 -Port 4180
```

Chrome or Edge hardware acceleration must be enabled. The app asks Windows for the high-performance WebGPU adapter and reports a clear error if the browser does not expose it.

## iPhone and mobile browsers

Current iPhones enter a phone-safe profile automatically: a 160×100 thermodynamic grid, a 10× default time scale, lower render resolution, a bounded number of submitted substeps, and GPU-queue backpressure. Before the interface appears, BL-05 runs and reads back one real compute step; invalid thermal data therefore becomes an explicit startup error instead of visible `NaN` values.

For the most reliable iPhone session, open the shared link in Safari. Embedded social-media browser sheets can use the same WebKit engine but have different lifetime and resource behavior. The Kitchen preset is intentionally slow at 22 °C; use **Warm pan** when you want to see a clear melt within a short test.

## What the simulation actually does

Each GPU cell stores butter thickness, a live substrate temperature, and eight vertical specific-enthalpy layers. Every adaptive substep computes:

1. one unique hydrostatic/tilt flux for every shared cell face;
2. donor and receiver scales that preserve mass, nonnegative height, and the local hydraulic-head maximum;
3. a bounded face exchange that prevents explicit thin-film flow from reversing a cell-to-cell height jump;
4. phase-dependent Bingham-style yield and viscosity using only the thermally mobile share of the eight-layer column;
5. mobile-layer-weighted enthalpy advection instead of moving the still-solid crystal network as liquid;
6. shared-face horizontal conduction, stable vertical conduction, and implicit equal-and-opposite substrate contact;
7. implicit convection/radiation and a single butter/substrate sunlight budget.

The phase-transition function clamps its transcendental input after reaching the mathematical asymptote, and both implicit thermal searches stay within the modeled −40 to 120 °C range. This avoids backend-dependent overflow under relaxed mobile GPU math without changing the calibrated butter transition.

When raw flow would outrun the explicit grid, BL-05 reduces the physical timestep and advances the displayed simulation clock only by the time actually solved. The engine badge shows `flow ×…` while that safeguard is active.

The energy-transition curve uses three published milk-fat DSC peaks. A separate solid-fat-content curve controls mechanical strength; this distinction prevents the common mistake of treating calorimetric transition fraction as literal liquid mass fraction.

The renderer reads the same GPU material field used by the solver. It reconstructs the cell-averaged top field with a compact tent filter and draws cut walls only where a coherent solid edge meets a dry neighbor; liquid-to-liquid gradients stay continuous instead of becoming vertical fences. Surface-layer temperature and crystal state drive a pale emulsion albedo, waxy-to-oily roughness, restrained microstructure, thin-edge scattering, and Fresnel sheen under a neutral studio light. Camera, Thermal, and Structure views are different renderings of one evolving state—there are no image crossfades or scripted melt animations.

## Controls

- **Room temperature** controls air and radiative surroundings.
- **Surface temperature** is the bulk setpoint toward which the finite countertop layer relaxes.
- **Airflow** changes the convective heat-transfer coefficient.
- **Direct sunlight** supplies 0–1000 W/m² of short-wave irradiance.
- **Counter tilt** adds a tangential gravity component.
- **Contact surface** changes conductance, thermal inertia, empirical surface mobility, absorption, and appearance.
- **Localized heater** applies a Gaussian heat flux wherever you drag on the surface.
- **Compute profile** changes both simulation-grid and internal render resolution.
- **Photo** pauses the solver, raises render resolution, and enables 4× MSAA for inspection.

## Scope and honesty

The model is a physically grounded educational approximation, not a certified food-process solver. It is strongest for a butter block melting and spreading on a countertop, where the depth is small relative to its footprint. The flow law includes hydrostatic pressure, tilt, phase-dependent yield, viscosity, and an empirical surface-mobility prior. It does not yet solve capillary pressure, contact angle, a precursor film, internal water droplets, protein films, evaporation, bubbling, splashing, fracture, or a micrometre-scale lubrication layer. Real butter varies with breed, season, formulation, salt, water fraction, working history, and crystal tempering.

A full 3D MPM reference is documented as the next validation phase, not mislabeled as something this browser is already doing. On a 6 GB GTX 1660 Ti, the current shallow-volume model spends the available resolution on visible thermal gradients, contact physics, and a smooth surface instead of exposing a coarse particle grid.

See [SOURCES.md](./SOURCES.md) for the calibration ledger and citations.

## Developer verification

The project has no runtime dependencies. The pure JavaScript physics-reference checks can be run with any current Node.js:

```powershell
node .\tests\physics.test.mjs
node .\tests\math.test.mjs
node .\tests\static.test.mjs
node .\tests\export-shaders.mjs
```

Codex used its bundled Node runtime because the machine's system Node registration is stale and the executable is missing.
