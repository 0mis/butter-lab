# BL-05 source and calibration ledger

Last reviewed: 2026-07-10.

## Architecture decision

- [Odyssey-2 Max](https://odyssey.ml/introducing-odyssey-2-max) is a private-beta generative world model trained on several hundred NVIDIA B200 GPUs. It is useful context for learned visual dynamics, but it is not a locally runnable, conservation-authoritative solver for this laptop.
- [WebGPU specification](https://gpuweb.github.io/gpuweb/) and [WGSL specification](https://www.w3.org/TR/WGSL/) define the compute and render paths used here. Chrome exposes WebGPU on Windows through Direct3D 12.
- [WebKit features in Safari 26.0](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/#webgpu) documents WebGPU and compute support on iOS. BL-05 still uses a smaller, queue-bounded phone profile because browser support does not make desktop-sized workloads free on a mobile GPU.
- [Three.js WebGPU documentation](https://threejs.org/docs/pages/WebGPURenderer.html) informed the browser feasibility study, but BL-05 uses the WebGPU API directly to keep the solver and renderer explicit and dependency-free.
- [NVIDIA Warp](https://nvidia.github.io/warp/) and [Newton](https://newton-physics.github.io/newton/stable/) remain the recommended native reference harnesses for a later 3D validation phase.

## Phase change and continuum mechanics

- Stomakhin et al., [Augmented MPM for Phase-Change and Varied Materials](https://disneyanimation.com/publications/augmented-mpm-for-phase-change-and-varied-materials/) — the main reference for coupled heat transport and material change in MPM.
- Chen et al., [A Momentum-Conserving Implicit Material Point Method for Surface Energies with Spatial Gradients](https://arxiv.org/abs/2101.12408) — surface energy, contact angle, thermomechanical phase change, and convective boundary concepts.
- Treviño Kala, [Thermomechanical Material Point Methods for Simulation of Burning and Melting Solids](https://escholarship.org/uc/item/3zd1p6z6) — modern melting-solid treatment and surface-energy context.
- Staron et al., [Butter on a hot pan: self-regulating dynamics of melt-lubricated sliding](https://arxiv.org/abs/2603.09494) — March 2026 experimental evidence that the basal melt film can be tens of micrometres. The experiments use ice and paraffin, not edible butter, so BL-05 treats this as a sub-grid mechanism and does not claim direct calibration from it.
- [CK-MPM](https://arxiv.org/abs/2412.10399) — the 2025 compact-kernel MPM direction recommended for a future native 3D reference on this GPU.

## Butter calibration priors

- Tomaszewska-Gras, [Melting and crystallization DSC profiles of milk fat](https://link.springer.com/article/10.1007/s10973-013-3087-2) — milk-fat DSC behavior and sample-history sensitivity.
- [DSC supplementary measurements](https://www.rsc.org/suppdata/d1/fo/d1fo00259g/d1fo00259g1.pdf) — transition peaks at approximately 14.24, 18.34, and 31.01 °C with a 35.23 °C endset. BL-05 scales the reported milk-fat transition enthalpy by a representative 81.1% butterfat fraction, yielding about 43 kJ/kg for the whole butter.
- Yang, Saunders, and Mohan, [Effect of Temperature on the Rheological, Textural, and Sensory Properties of Butters](https://openprairie.sdstate.edu/dairy_pubs/96/) — measured loss of storage/loss modulus and increased spreadability from 5–25 °C, with meaningful variation among products.
- [Milk-fat rheology study](https://pubs.rsc.org/en/content/articlehtml/2023/sm/d3sm01097j) — temperature-dependent anhydrous milk-fat viscosity and crystal-network behavior.
- [Thermal properties of butter](https://paperzz.com/doc/9024465/thermal-properties-of-butter) — historical direct measurements supporting initial density, conductivity, and sensible heat-capacity ranges.

Current priors in code:

- density: 911 kg/m³;
- conductivity: 0.24 W/(m·K);
- sensible heat capacity: 2.05 kJ/(kg·K);
- transition enthalpy: 43.47 kJ/kg distributed across three DSC bands;
- molten-fat viscosity near 40 °C: 0.03 Pa·s;
- surface tension literature prior: 0.03 N/m (recorded for a future implicit capillary solve; not active in BL-05 flow);
- refractive index: 1.455, giving dielectric F0 ≈ 0.034.

These values are calibration starting points. They are not universal constants for every butter.

## Numerical treatment in this release

- The [WGSL floating-point accuracy rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy) permit `tanh` accuracy to inherit from `sinh/cosh`, whose exponential implementation can overflow for unnecessarily large inputs. BL-05 clamps the already-saturated transition argument to ±10 and bounds its implicit thermal brackets before evaluating transcendental math.
- [TapML](https://jw-liu.xyz/assets/pdf/tapml.pdf) reports a closely matching Metal fast-math `tanh` NaN and a stable-function repair. This is implementation evidence for the portability guard, not a butter calibration source.
- Apple, [Optimize machine learning for Metal apps](https://developer.apple.com/videos/play/wwdc2025/236/), recommends controlling passes, bandwidth, and resource pressure on Apple GPUs. BL-05 keeps enthalpy in f32, defaults phones to Efficient, caps substeps, and waits on submitted mobile GPU work.
- Herson et al., [Dripping Thin Films for Real-time Digital Painting](https://eliemichel.github.io/dripping-thin-films/documents/herson26dripping_thin_films.pdf) — the current real-time reference for shared-edge thin-film transport that constrains both donor availability and receiver capacity. BL-05 adapts that idea to a hydraulic-head bound on its regular WebGPU grid.
- Vantzos et al., [Real-Time Viscous Thin Films](https://mirelabc.github.io/publications/rtvtf.pdf) — mass-preserving, nonnegative GPU thin-film evolution and the longer-term reference for adding a controlled surface-energy solve.
- Mass uses one signed flux per shared face. A 50% mobile-depth donor bound, a tilt-aware receiver bound, and a local monotone face budget prevent both negative columns and new grid-scale towers without clipping or deleting mass.
- Each thermal layer receives a normalized mechanical-fluidity weight. Flow mobility uses the equivalent mobile depth rather than total block height cubed, and transported enthalpy comes preferentially from those mobile layers while remaining conservative across a shared face.
- The displayed simulation clock advances only by the timestep actually dispatched. Telemetry feeds limiter activity back into the next timestep; the engine badge exposes this as `flow ×…` instead of silently claiming the requested acceleration.
- Horizontal conduction uses the common overlap thickness at each face and is antisymmetric. Vertical layer exchange uses a stable pair relaxation. Countertop/butter contact is a nonlinear backward-Euler, equal-and-opposite energy transfer.
- Sunlight is partitioned once using a Beer–Lambert-style optical depth: absorbed butter energy plus transmitted/absorbed substrate energy never exceeds incident irradiance.
- Capillary pressure and contact-angle wetting are intentionally disabled. Making the fourth-order capillary operator quantitative requires an implicit solve and convergence criterion; a clipped explicit term would be misleading at this grid and timestep.

## Boundary conditions

- [ASHRAE Fundamentals, heat transfer](https://handbook.ashrae.org/Handbooks/F17/SI/f17_ch04/f17_ch04_si.aspx) — forced-convection correlation background.
- [NIST thermal radiation formulation](https://nvlpubs.nist.gov/nistpubs/ir/2006/ir7292.pdf) — Stefan–Boltzmann exchange.
- [NREL AM1.5 reference solar spectrum](https://www.nrel.gov/grid/solar-resource/spectra-am1.5) — basis for the 0–1000 W/m² sunlight control range.

## Known omissions

BL-05 does not currently model capillary/contact-angle dynamics, dispersed water droplets, emulsion breaking, evaporation, foaming, browning, salt solution, proteins, microbial behavior, fracture, cutting, splashing, or explicit 3D air flow. Surface-specific conductance and mobility values are engineering priors. Validation against recorded center-temperature, transition-state, and footprint experiments remains necessary before quantitative use.
