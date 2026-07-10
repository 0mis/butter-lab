const sharedPhysics = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const RHO: f32 = 911.0;
const GRAVITY: f32 = 9.80665;
const BUTTER_K: f32 = 0.24;
const CP: f32 = 2050.0;
const SIGMA: f32 = 5.6703744e-8;

struct Cell {
  geom: vec4f,       // height, substrate temperature, mean butter temperature, mean solid-fat content
  thermal0: vec4f,   // specific enthalpy for vertical layers 0..3 (bottom to top)
  thermal1: vec4f,   // specific enthalpy for vertical layers 4..7
};

struct Params {
  grid: vec4f,       // width, height, dx, dz
  timing: vec4f,     // dt, domain width, domain depth, simulation time
  environment: vec4f,// ambient C, substrate setpoint C, sunlight W/m2, airflow m/s
  material: vec4f,   // contact H, substrate alpha, substrate areal capacity, empirical surface mobility
  material2: vec4f,  // relaxation seconds, solar absorptivity, material id, view mode
  tilt: vec4f,       // slope x, slope z, heater x, heater z
  heater: vec4f,     // heat flux W/m2, radius m, active, ambient specific enthalpy
};

fn saturate(value: f32) -> f32 { return clamp(value, 0.0, 1.0); }

fn transition(t: f32, peak: f32, width: f32) -> f32 {
  return 0.5 * (1.0 + tanh((t - peak) / width));
}

fn temperatureToEnthalpy(t: f32) -> f32 {
  return CP * t
    + 9740.0 * transition(t, 14.24, 1.5)
    + 12537.0 * transition(t, 18.34, 1.5)
    + 21196.0 * transition(t, 31.01, 2.0);
}

fn enthalpyToTemperature(value: f32) -> f32 {
  // Compact monotone inverse LUT. Dense knots surround each DSC transition;
  // linear lookup avoids hundreds of transcendental evaluations per column.
  if (value < 0.0) { return value / CP; }
  if (value < 10250.044) { return mix(0.0, 5.0, value / 10250.044); }
  if (value < 16402.385) { return mix(5.0, 8.0, (value - 10250.044) / 6152.341); }
  if (value < 20534.213) { return mix(8.0, 10.0, (value - 16402.385) / 4131.828); }
  if (value < 25070.503) { return mix(10.0, 12.0, (value - 20534.213) / 4536.290); }
  if (value < 32835.731) { return mix(12.0, 14.0, (value - 25070.503) / 7765.228); }
  if (value < 42219.588) { return mix(14.0, 16.0, (value - 32835.731) / 9383.857); }
  if (value < 51447.200) { return mix(16.0, 18.0, (value - 42219.588) / 9227.612); }
  if (value < 62037.212) { return mix(18.0, 20.0, (value - 51447.200) / 10590.012); }
  if (value < 67284.751) { return mix(20.0, 22.0, (value - 62037.212) / 5247.539); }
  if (value < 73577.140) { return mix(22.0, 25.0, (value - 67284.751) / 6292.389); }
  if (value < 80672.674) { return mix(25.0, 28.0, (value - 73577.140) / 7095.534); }
  if (value < 89435.903) { return mix(28.0, 30.0, (value - 80672.674) / 8763.229); }
  if (value < 103330.747) { return mix(30.0, 32.0, (value - 89435.903) / 13894.844); }
  if (value < 114838.002) { return mix(32.0, 35.0, (value - 103330.747) / 11507.255); }
  if (value < 125470.358) { return mix(35.0, 40.0, (value - 114838.002) / 10632.356); }
  return (value - 43473.0) / CP;
}

fn smoothMix(a: f32, b: f32, amount: f32) -> f32 {
  let x = saturate(amount);
  let eased = x * x * (3.0 - 2.0 * x);
  return mix(a, b, eased);
}

fn solidFatContent(t: f32) -> f32 {
  if (t <= -10.0) { return 0.62; }
  if (t < 0.0) { return smoothMix(0.62, 0.54, (t + 10.0) / 10.0); }
  if (t < 10.0) { return smoothMix(0.54, 0.43, t / 10.0); }
  if (t < 20.0) { return smoothMix(0.43, 0.18, (t - 10.0) / 10.0); }
  if (t < 30.0) { return smoothMix(0.18, 0.05, (t - 20.0) / 10.0); }
  if (t < 35.0) { return smoothMix(0.05, 0.005, (t - 30.0) / 5.0); }
  if (t < 40.0) { return smoothMix(0.005, 0.0, (t - 35.0) / 5.0); }
  return 0.0;
}

fn enthalpyAt(cell: Cell, layer: u32) -> f32 {
  if (layer < 4u) { return cell.thermal0[layer]; }
  return cell.thermal1[layer - 4u];
}
`;

export const computeShader = /* wgsl */ `
${sharedPhysics}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<Cell>;
@group(0) @binding(2) var<storage, read_write> destination: array<Cell>;

struct FlowScratch {
  values: vec4f, // immutable raw +x/+z fluxes for one substep
};
@group(0) @binding(3) var<storage, read_write> flowScratch: array<FlowScratch>;
@group(0) @binding(4) var<storage, read_write> donorScales: array<f32>;

fn width() -> i32 { return i32(params.grid.x); }
fn height() -> i32 { return i32(params.grid.y); }

fn inside(x: i32, z: i32) -> bool {
  return x >= 0 && x < width() && z >= 0 && z < height();
}

fn cellIndex(x: i32, z: i32) -> u32 {
  let safeX = clamp(x, 0, width() - 1);
  let safeZ = clamp(z, 0, height() - 1);
  return u32(safeZ * width() + safeX);
}

fn loadCell(x: i32, z: i32) -> Cell { return source[cellIndex(x, z)]; }
fn cellHeight(x: i32, z: i32) -> f32 { return loadCell(x, z).geom.x; }

fn meanSolidFat(cell: Cell) -> f32 {
  var total = 0.0;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    total += solidFatContent(enthalpyToTemperature(enthalpyAt(cell, layer)));
  }
  return total * 0.125;
}

fn rawFaceFlux(x0: i32, z0: i32, x1: i32, z1: i32, axis: u32) -> f32 {
  if (!inside(x0, z0) || !inside(x1, z1)) { return 0.0; }
  let a = loadCell(x0, z0);
  let b = loadCell(x1, z1);
  let h0 = max(a.geom.x, 0.0);
  let h1 = max(b.geom.x, 0.0);
  if (max(h0, h1) < 2.0e-7) { return 0.0; }

  let spacing = select(params.grid.w, params.grid.z, axis == 0u);
  let slope = select(params.tilt.y, params.tilt.x, axis == 0u);
  // Hydrostatic and tilted-gravity drive. Capillary pressure is deliberately
  // omitted until it can be solved implicitly rather than timestep-clipped.
  let pressureDrive = RHO * GRAVITY * (h0 - h1) / spacing;
  let drive = pressureDrive + RHO * GRAVITY * slope;
  let hFace = max(0.5 * (h0 + h1), 8.0e-6);
  let weightedTemperature = (h0 * a.geom.z + h1 * b.geom.z) / max(h0 + h1, 1.0e-8);
  let sfc = (h0 * meanSolidFat(a) + h1 * meanSolidFat(b)) / max(h0 + h1, 1.0e-8);
  let liquidViscosity = 0.03 * exp(0.05 * (40.0 - weightedTemperature));
  let normalizedSfc = clamp(sfc / 0.43, 0.0, 1.45);
  let yieldStress = 50000.0 * normalizedSfc * normalizedSfc * normalizedSfc;
  let bingham = clamp(yieldStress / (hFace * abs(drive) + 1.0e-4), 0.0, 1.0);
  let yieldedProfile = max(0.0, 1.0 - 1.5 * bingham + 0.5 * bingham * bingham * bingham);
  let networkFluidity = 1.0 - smoothstep(0.035, 0.22, sfc);
  let mobility = params.material.w * networkFluidity * networkFluidity
    * hFace * hFace * hFace / (3.0 * max(liquidViscosity, 0.02));
  return mobility * yieldedProfile * drive;
}

fn limitedFace(x0: i32, z0: i32, x1: i32, z1: i32, axis: u32) -> f32 {
  if (!inside(x0, z0) || !inside(x1, z1)) { return 0.0; }
  let scratchA = flowScratch[cellIndex(x0, z0)].values;
  let raw = select(scratchA.y, scratchA.x, axis == 0u);
  let thetaA = donorScales[cellIndex(x0, z0)];
  let thetaB = donorScales[cellIndex(x1, z1)];
  return raw * select(thetaB, thetaA, raw >= 0.0);
}

fn horizontalHeatFace(x0: i32, z0: i32, x1: i32, z1: i32, layer: u32, axis: u32) -> f32 {
  if (!inside(x0, z0) || !inside(x1, z1)) { return 0.0; }
  let a = loadCell(x0, z0);
  let b = loadCell(x1, z1);
  let overlap = min(max(a.geom.x, 0.0), max(b.geom.x, 0.0));
  if (overlap <= 0.0) { return 0.0; }
  let spacing = select(params.grid.w, params.grid.z, axis == 0u);
  let ta = enthalpyToTemperature(enthalpyAt(a, layer));
  let tb = enthalpyToTemperature(enthalpyAt(b, layer));
  return (BUTTER_K / RHO) * overlap * (ta - tb) / spacing;
}

fn verticalPairDelta(centerH: f32, neighborH: f32, dt: f32, spacing: f32) -> f32 {
  let centerT = enthalpyToTemperature(centerH);
  let neighborT = enthalpyToTemperature(neighborH);
  let deltaT = neighborT - centerT;
  if (abs(deltaT) < 1.0e-6) { return 0.0; }
  let apparentCp = clamp(abs((neighborH - centerH) / deltaT), CP, 25000.0);
  let alpha = BUTTER_K / (RHO * apparentCp);
  let relaxed = 1.0 - exp(-2.0 * alpha * dt / max(spacing * spacing, 1.0e-14));
  return 0.5 * (neighborH - centerH) * clamp(relaxed, 0.0, 1.0);
}

fn surfaceResidual(t: f32, hStar: f32, h: f32, dt: f32, hAir: f32, solarButter: f32) -> f32 {
  let ambient = params.environment.x;
  let tKelvin = t + 273.15;
  let ambientKelvin = ambient + 273.15;
  let convectionFlux = hAir * (ambient - t);
  let radiationFlux = 0.94 * SIGMA * (ambientKelvin * ambientKelvin * ambientKelvin * ambientKelvin
    - tKelvin * tKelvin * tKelvin * tKelvin);
  let capacityEnergy = RHO * h * (temperatureToEnthalpy(t) - hStar) * 0.125;
  return capacityEnergy - dt * (convectionFlux + radiationFlux + solarButter);
}

fn contactTemperatureResidual(t: f32, h0: f32, h: f32, substrateT: f32, substrateCapacity: f32, dt: f32) -> f32 {
  let q = RHO * h * (temperatureToEnthalpy(t) - h0) * 0.125;
  let updatedSubstrate = substrateT - q / substrateCapacity;
  return q - dt * params.material.x * (updatedSubstrate - t);
}

@compute @workgroup_size(8, 8, 1)
fn computeRawFaces(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let z = i32(gid.y);
  if (x >= width() || z >= height()) { return; }
  let qRight = rawFaceFlux(x, z, x + 1, z, 0u);
  let qUp = rawFaceFlux(x, z, x, z + 1, 1u);
  flowScratch[cellIndex(x, z)].values = vec4f(qRight, qUp, 0.0, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn computeDonorScale(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let z = i32(gid.y);
  if (x >= width() || z >= height()) { return; }
  let index = cellIndex(x, z);
  let current = flowScratch[index].values;
  let qRight = current.x;
  let qUp = current.y;
  let qLeft = select(0.0, flowScratch[cellIndex(x - 1, z)].values.x, x > 0);
  let qDown = select(0.0, flowScratch[cellIndex(x, z - 1)].values.y, z > 0);
  let outgoing = max(qRight, 0.0) / params.grid.z + max(-qLeft, 0.0) / params.grid.z
    + max(qUp, 0.0) / params.grid.w + max(-qDown, 0.0) / params.grid.w;
  let h = max(loadCell(x, z).geom.x, 0.0);
  var theta = 1.0;
  if (outgoing > 0.0) {
    if (h > 0.0) {
      theta = min(1.0, 0.90 * h / max(params.timing.x * outgoing, 1.0e-30));
    } else {
      theta = 0.0;
    }
  }
  donorScales[index] = theta;
}

fn upwindEnthalpy(a: Cell, b: Cell, layer: u32, flux: f32) -> f32 {
  return select(enthalpyAt(b, layer), enthalpyAt(a, layer), flux >= 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let z = i32(gid.y);
  if (x >= width() || z >= height()) { return; }

  let center = loadCell(x, z);
  let left = loadCell(x - 1, z);
  let right = loadCell(x + 1, z);
  let down = loadCell(x, z - 1);
  let up = loadCell(x, z + 1);
  let dt = params.timing.x;
  let dx = params.grid.z;
  let dz = params.grid.w;

  let qRight = limitedFace(x, z, x + 1, z, 0u);
  let qLeft = limitedFace(x - 1, z, x, z, 0u);
  let qUp = limitedFace(x, z, x, z + 1, 1u);
  let qDown = limitedFace(x, z - 1, x, z, 1u);

  let oldHeight = max(center.geom.x, 0.0);
  let newHeight = oldHeight - dt * ((qRight - qLeft) / dx + (qUp - qDown) / dz);

  var oldTemperatures: array<f32, 8>;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    oldTemperatures[layer] = enthalpyToTemperature(enthalpyAt(center, layer));
  }

  var newArealEnthalpy: array<f32, 8>;
  var newEnthalpies: array<f32, 8>;
  let layerDepth = max(oldHeight / 8.0, 1.0e-8);
  let ambientEnthalpy = params.heater.w;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    let hSpecific = enthalpyAt(center, layer);
    var arealEnthalpy = oldHeight * hSpecific;
    let rightH = upwindEnthalpy(center, right, layer, qRight);
    let leftH = upwindEnthalpy(left, center, layer, qLeft);
    let upH = upwindEnthalpy(center, up, layer, qUp);
    let downH = upwindEnthalpy(down, center, layer, qDown);
    arealEnthalpy -= dt * ((qRight * rightH - qLeft * leftH) / dx + (qUp * upH - qDown * downH) / dz);

    let heatRight = horizontalHeatFace(x, z, x + 1, z, layer, 0u);
    let heatLeft = horizontalHeatFace(x - 1, z, x, z, layer, 0u);
    let heatUp = horizontalHeatFace(x, z, x, z + 1, layer, 1u);
    let heatDown = horizontalHeatFace(x, z - 1, x, z, layer, 1u);
    arealEnthalpy -= dt * ((heatRight - heatLeft) / dx + (heatUp - heatDown) / dz);

    if (oldHeight > 0.0 && layer > 0u) {
      arealEnthalpy += oldHeight * verticalPairDelta(hSpecific, enthalpyAt(center, layer - 1u), dt, layerDepth);
    }
    if (oldHeight > 0.0 && layer < 7u) {
      arealEnthalpy += oldHeight * verticalPairDelta(hSpecific, enthalpyAt(center, layer + 1u), dt, layerDepth);
    }
    newArealEnthalpy[layer] = arealEnthalpy;
  }

  let substrateLap = (left.geom.y - 2.0 * center.geom.y + right.geom.y) / (dx * dx)
    + (down.geom.y - 2.0 * center.geom.y + up.geom.y) / (dz * dz);
  let px = (f32(x) / max(params.grid.x - 1.0, 1.0) - 0.5) * params.timing.y;
  let pz = (f32(z) / max(params.grid.y - 1.0, 1.0) - 0.5) * params.timing.z;
  let heaterDistance2 = (px - params.tilt.z) * (px - params.tilt.z) + (pz - params.tilt.w) * (pz - params.tilt.w);
  let heaterFlux = params.heater.x * params.heater.z * exp(-heaterDistance2 / max(2.0 * params.heater.y * params.heater.y, 1.0e-6));
  let transmission = exp(-max(newHeight, 0.0) / 2.0e-4);
  let solarButter = 0.72 * params.environment.z * (1.0 - transmission);
  let substrateSolar = saturate(params.material2.y) * params.environment.z * transmission;
  let relaxation = (params.environment.y - center.geom.y) / max(params.material2.x, 1.0);
  let substrateSource = (substrateSolar + heaterFlux) / max(params.material.z, 1000.0);
  var newSubstrate = center.geom.y + dt * (params.material.y * substrateLap + relaxation + substrateSource);

  if (newHeight > 1.0e-12) {
    // The free-surface solve is implicit in enthalpy, so convection and
    // radiation remain monotone even for a micrometre-scale film.
    let topHStar = newArealEnthalpy[7] / newHeight;
    let topTStar = enthalpyToTemperature(topHStar);
    let airCoefficient = clamp(4.2 + 7.1 * sqrt(max(params.environment.w, 0.0)), 3.0, 25.0);
    var lowT = min(topTStar, params.environment.x) - 150.0;
    var highT = max(topTStar, params.environment.x) + 600.0;
    for (var iteration = 0u; iteration < 28u; iteration = iteration + 1u) {
      let middleT = 0.5 * (lowT + highT);
      if (surfaceResidual(middleT, topHStar, newHeight, dt, airCoefficient, solarButter) > 0.0) {
        highT = middleT;
      } else {
        lowT = middleT;
      }
    }
    newArealEnthalpy[7] = newHeight * temperatureToEnthalpy(0.5 * (lowT + highT));

    // Backward-Euler contact exchange is solved as one equal-and-opposite
    // energy transfer between the substrate and bottom butter layer.
    let substrateCapacity = max(params.material.z, 1000.0);
    let bottomHStar = newArealEnthalpy[0] / newHeight;
    let bottomTEstimate = enthalpyToTemperature(bottomHStar);
    var lowContactT = min(bottomTEstimate, newSubstrate) - 200.0;
    var highContactT = max(bottomTEstimate, newSubstrate) + 200.0;
    for (var iteration = 0u; iteration < 24u; iteration = iteration + 1u) {
      let middleContactT = 0.5 * (lowContactT + highContactT);
      if (contactTemperatureResidual(middleContactT, bottomHStar, newHeight, newSubstrate, substrateCapacity, dt) > 0.0) {
        highContactT = middleContactT;
      } else {
        lowContactT = middleContactT;
      }
    }
    let contactTemperature = 0.5 * (lowContactT + highContactT);
    let contactEnergy = RHO * newHeight * (temperatureToEnthalpy(contactTemperature) - bottomHStar) * 0.125;
    newArealEnthalpy[0] += 8.0 * contactEnergy / RHO;
    newSubstrate -= contactEnergy / substrateCapacity;
  }

  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    if (newHeight > 1.0e-12) {
      newEnthalpies[layer] = newArealEnthalpy[layer] / newHeight;
    } else {
      newEnthalpies[layer] = ambientEnthalpy;
    }
  }

  var meanTemperature = 0.0;
  var meanSfc = 0.0;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    let layerTemperature = enthalpyToTemperature(newEnthalpies[layer]);
    meanTemperature += layerTemperature;
    meanSfc += solidFatContent(layerTemperature);
  }
  meanTemperature *= 0.125;
  meanSfc *= 0.125;

  var output: Cell;
  output.geom = vec4f(newHeight, newSubstrate, meanTemperature, meanSfc);
  output.thermal0 = vec4f(newEnthalpies[0], newEnthalpies[1], newEnthalpies[2], newEnthalpies[3]);
  output.thermal1 = vec4f(newEnthalpies[4], newEnthalpies[5], newEnthalpies[6], newEnthalpies[7]);
  destination[cellIndex(x, z)] = output;
}
`;

const sharedRender = /* wgsl */ `
${sharedPhysics}

struct Scene {
  viewProjection: mat4x4f,
  inverseViewProjection: mat4x4f,
  cameraPosition: vec4f,
  lightDirection: vec4f,
  viewport: vec4f,   // width, height, time, mode
  appearance: vec4f, // surface id, exposure, photo quality, reserved
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> field: array<Cell>;

fn gridWidth() -> i32 { return i32(params.grid.x); }
fn gridHeight() -> i32 { return i32(params.grid.y); }
fn safeIndex(x: i32, z: i32) -> u32 {
  return u32(clamp(z, 0, gridHeight() - 1) * gridWidth() + clamp(x, 0, gridWidth() - 1));
}
fn heightAt(x: i32, z: i32) -> f32 { return field[safeIndex(x, z)].geom.x; }

fn worldToGrid(position: vec2f) -> vec2f {
  return vec2f(
    (position.x / params.timing.y + 0.5) * (params.grid.x - 1.0),
    (position.y / params.timing.z + 0.5) * (params.grid.y - 1.0)
  );
}

fn sampleCellNearest(position: vec2f) -> Cell {
  let coordinate = worldToGrid(position);
  return field[safeIndex(i32(round(coordinate.x)), i32(round(coordinate.y)))];
}

fn hash21(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2f(1.0, 0.0)), u.x),
             mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn aces(color: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((color * (a * color + vec3f(b))) / (color * (c * color + vec3f(d)) + vec3f(e)), vec3f(0.0), vec3f(1.0));
}

fn thermalColor(value: f32) -> vec3f {
  let x = saturate((value - 0.0) / 55.0);
  let c0 = vec3f(0.035, 0.055, 0.22);
  let c1 = vec3f(0.04, 0.56, 0.82);
  let c2 = vec3f(0.22, 0.86, 0.49);
  let c3 = vec3f(1.0, 0.76, 0.12);
  let c4 = vec3f(0.94, 0.12, 0.035);
  if (x < 0.25) { return mix(c0, c1, x * 4.0); }
  if (x < 0.5) { return mix(c1, c2, (x - 0.25) * 4.0); }
  if (x < 0.75) { return mix(c2, c3, (x - 0.5) * 4.0); }
  return mix(c3, c4, (x - 0.75) * 4.0);
}

fn vignette(color: vec3f, pixel: vec2f) -> vec3f {
  let uv = pixel / scene.viewport.xy;
  let centered = uv * 2.0 - vec2f(1.0);
  let amount = 1.0 - 0.075 * dot(centered, centered);
  return color * clamp(amount, 0.86, 1.0);
}

fn countertopShadow(position: vec3f) -> f32 {
  let light = normalize(scene.lightDirection.xyz);
  let planarLength = max(length(light.xz), 1.0e-4);
  let planarDirection = light.xz / planarLength;
  let verticalPerMeter = light.y / planarLength;
  var occlusion = 0.0;
  for (var step = 1; step <= 18; step = step + 1) {
    let distance = f32(step) * 0.0038;
    let samplePosition = position.xz + planarDirection * distance;
    let coordinate = worldToGrid(samplePosition);
    if (coordinate.x < 0.0 || coordinate.y < 0.0 || coordinate.x >= params.grid.x || coordinate.y >= params.grid.y) { continue; }
    let blocker = sampleCellNearest(samplePosition).geom.x;
    let rayHeight = position.y + verticalPerMeter * distance;
    occlusion += smoothstep(-0.0002, 0.0010, blocker - rayHeight - 0.00035);
  }
  return clamp(exp(-0.24 * occlusion), 0.16, 1.0);
}
`;

export const backgroundShader = /* wgsl */ `
${sharedRender}

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn backgroundVertex(@builtin(vertex_index) vertexIndex: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var output: FullscreenOut;
  output.position = vec4f(positions[vertexIndex], 0.99999, 1.0);
  output.uv = positions[vertexIndex] * 0.5 + vec2f(0.5);
  return output;
}

fn surfaceMaterial(position: vec3f, id: i32) -> vec3f {
  let p = position.xz;
  if (id == 0) {
    let broad = valueNoise(p * 8.0);
    let vein = smoothstep(0.77, 0.9, abs(sin((p.x * 21.0 + p.y * 9.0 + valueNoise(p * 17.0) * 3.2))));
    return mix(vec3f(0.30, 0.305, 0.29), vec3f(0.56, 0.55, 0.51), broad * 0.45) + vein * vec3f(0.12, 0.115, 0.10);
  }
  if (id == 1) {
    let brush = 0.035 * sin(p.x * 2900.0) + 0.018 * valueNoise(vec2f(p.x * 900.0, p.y * 25.0));
    return vec3f(0.32 + brush, 0.335 + brush, 0.33 + brush);
  }
  if (id == 2) {
    let grain = 0.5 + 0.5 * sin(p.x * 115.0 + valueNoise(p * vec2f(18.0, 4.0)) * 5.0);
    let pore = valueNoise(p * 310.0) * 0.04;
    return mix(vec3f(0.105, 0.050, 0.027), vec3f(0.29, 0.145, 0.075), grain * 0.65) + pore;
  }
  let ceramic = valueNoise(p * 95.0) * 0.025;
  return vec3f(0.43 + ceramic, 0.42 + ceramic, 0.38 + ceramic);
}

@fragment
fn backgroundFragment(input: FullscreenOut) -> @location(0) vec4f {
  // The fullscreen triangle's interpolated UV already follows clip-space Y
  // (0 at NDC -1, 1 at NDC +1), so no framebuffer-origin flip is needed.
  let ndc = vec2f(input.uv.x * 2.0 - 1.0, input.uv.y * 2.0 - 1.0);
  var nearPoint = scene.inverseViewProjection * vec4f(ndc, 0.0, 1.0);
  var farPoint = scene.inverseViewProjection * vec4f(ndc, 1.0, 1.0);
  nearPoint /= nearPoint.w;
  farPoint /= farPoint.w;
  let rayDirection = normalize(farPoint.xyz - nearPoint.xyz);
  let rayOrigin = scene.cameraPosition.xyz;
  var planeDistance = -1.0;
  if (rayDirection.y < -1.0e-5) { planeDistance = -rayOrigin.y / rayDirection.y; }

  var color: vec3f;
  if (planeDistance > 0.0) {
    let world = rayOrigin + rayDirection * planeDistance;
    let materialId = i32(round(scene.appearance.x));
    let base = surfaceMaterial(world, materialId);
    let light = normalize(scene.lightDirection.xyz);
    let view = normalize(rayOrigin - world);
    let halfVector = normalize(light + view);
    let roughness = select(select(select(0.38, 0.58, materialId == 3), 0.62, materialId == 2), 0.25, materialId == 1);
    let diffuse = 0.34 + 0.66 * max(light.y, 0.0);
    let specular = pow(max(halfVector.y, 0.0), mix(140.0, 18.0, roughness));
    let shadow = countertopShadow(vec3f(world.x, 0.00005, world.z));
    let radial = length(world.xz);
    let studioPool = exp(-radial * radial * 6.0);
    color = base * (0.28 + diffuse * shadow) + vec3f(1.0, 0.98, 0.92) * specular * shadow * (1.0 - roughness) * 0.72;
    color += studioPool * vec3f(0.040, 0.040, 0.037);
    let gridCoordinate = worldToGrid(world.xz);
    let gridX = i32(round(gridCoordinate.x));
    let gridZ = i32(round(gridCoordinate.y));
    let centerHeight = heightAt(gridX, gridZ);
    let neighborHeight = max(max(heightAt(gridX - 1, gridZ), heightAt(gridX + 1, gridZ)),
      max(heightAt(gridX, gridZ - 1), heightAt(gridX, gridZ + 1)));
    let contactRing = (1.0 - smoothstep(0.0, 2.0e-5, centerHeight)) * smoothstep(3.0e-4, 4.0e-3, neighborHeight);
    color *= 1.0 - 0.20 * contactRing;

    if (i32(round(scene.viewport.w)) == 1) {
      let substrate = sampleCellNearest(world.xz).geom.y;
      color = mix(color * 0.12, thermalColor(substrate), 0.86);
    }
  } else {
    let horizon = saturate(0.5 + 0.5 * rayDirection.y);
    color = mix(vec3f(0.018, 0.018, 0.017), vec3f(0.145, 0.148, 0.142), pow(horizon, 1.8));
    let window = pow(max(dot(rayDirection, normalize(vec3f(-0.38, 0.58, -0.72))), 0.0), 220.0);
    color += window * vec3f(3.7, 3.6, 3.35);
  }

  color *= exp2(scene.appearance.y);
  color = aces(max(color, vec3f(0.0)));
  color = vignette(color, input.position.xy);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
`;

export const butterShader = /* wgsl */ `
${sharedRender}

struct ButterVertexOut {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) worldNormal: vec3f,
  @location(2) properties: vec4f, // height, temperature, sfc, substrate temperature
};

@vertex
fn butterVertex(@builtin(vertex_index) vertexIndex: u32) -> ButterVertexOut {
  let x = i32(vertexIndex % u32(gridWidth()));
  let z = i32(vertexIndex / u32(gridWidth()));
  let cell = field[vertexIndex];
  let worldX = (f32(x) / max(params.grid.x - 1.0, 1.0) - 0.5) * params.timing.y;
  let worldZ = (f32(z) / max(params.grid.y - 1.0, 1.0) - 0.5) * params.timing.z;
  let derivativeX = (heightAt(x + 1, z) - heightAt(x - 1, z)) / (2.0 * params.grid.z);
  let derivativeZ = (heightAt(x, z + 1) - heightAt(x, z - 1)) / (2.0 * params.grid.w);
  let normal = normalize(vec3f(-derivativeX, 1.0, -derivativeZ));
  let surfaceTemperature = enthalpyToTemperature(cell.thermal1.w);
  let surfaceSfc = solidFatContent(surfaceTemperature);
  let world = vec3f(worldX, max(cell.geom.x, 0.0) + 0.00004, worldZ);
  var output: ButterVertexOut;
  output.position = scene.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = normal;
  output.properties = vec4f(cell.geom.x, surfaceTemperature, surfaceSfc, cell.geom.y);
  return output;
}

@vertex
fn butterWallVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> ButterVertexOut {
  let cellInstance = instanceIndex / 4u;
  let side = instanceIndex % 4u;
  let x = i32(cellInstance % u32(gridWidth()));
  let z = i32(cellInstance / u32(gridWidth()));
  let cell = field[cellInstance];
  let h = max(cell.geom.x, 0.0);
  var neighborX = x;
  var neighborZ = z;
  var outward = vec3f(0.0);
  if (side == 0u) { neighborX = x - 1; outward = vec3f(-1.0, 0.0, 0.0); }
  if (side == 1u) { neighborX = x + 1; outward = vec3f(1.0, 0.0, 0.0); }
  if (side == 2u) { neighborZ = z - 1; outward = vec3f(0.0, 0.0, -1.0); }
  if (side == 3u) { neighborZ = z + 1; outward = vec3f(0.0, 0.0, 1.0); }
  let neighborInside = neighborX >= 0 && neighborX < gridWidth() && neighborZ >= 0 && neighborZ < gridHeight();
  let neighborH = select(0.0, heightAt(neighborX, neighborZ), neighborInside);
  let boundary = h >= 2.0e-5 && h - neighborH > 4.5e-4;

  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let centerX = (f32(x) / max(params.grid.x - 1.0, 1.0) - 0.5) * params.timing.y;
  let centerZ = (f32(z) / max(params.grid.y - 1.0, 1.0) - 0.5) * params.timing.z;
  var world = vec3f(0.0);
  if (abs(outward.x) > 0.5) {
    world = vec3f(centerX + outward.x * params.grid.z * 0.5, mix(neighborH, h, corner.y), centerZ + (corner.x - 0.5) * params.grid.w);
  } else {
    world = vec3f(centerX + (corner.x - 0.5) * params.grid.z, mix(neighborH, h, corner.y), centerZ + outward.z * params.grid.w * 0.5);
  }
  if (!boundary) { world = vec3f(0.0); }

  let bottomTemperature = enthalpyToTemperature(cell.thermal0.x);
  let surfaceTemperature = enthalpyToTemperature(cell.thermal1.w);
  let wallTemperature = mix(bottomTemperature, surfaceTemperature, corner.y);
  let wallSfc = solidFatContent(wallTemperature);
  var output: ButterVertexOut;
  output.position = scene.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = outward;
  output.properties = vec4f(select(0.0, h, boundary), wallTemperature, wallSfc, cell.geom.y);
  return output;
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(1.0 - saturate(cosTheta), 5.0);
}

fn distributionGGX(normal: vec3f, halfVector: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let nDotH = max(dot(normal, halfVector), 0.0);
  let denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * denominator * denominator, 1.0e-5);
}

fn geometrySchlickGGX(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = r * r / 8.0;
  return nDotV / max(nDotV * (1.0 - k) + k, 1.0e-5);
}

fn structureColor(sfc: f32) -> vec3f {
  let network = smoothstep(0.02, 0.25, sfc);
  return mix(vec3f(0.075, 0.43, 0.63), vec3f(0.98, 0.75, 0.20), network);
}

fn selfShadow(position: vec3f) -> f32 {
  let light = normalize(scene.lightDirection.xyz);
  let planarLength = max(length(light.xz), 1.0e-4);
  let planarDirection = light.xz / planarLength;
  let verticalPerMeter = light.y / planarLength;
  var occlusion = 0.0;
  for (var step = 1; step <= 12; step = step + 1) {
    let distance = f32(step) * 0.0028;
    let samplePosition = position.xz + planarDirection * distance;
    let blocker = sampleCellNearest(samplePosition).geom.x;
    let rayHeight = position.y + verticalPerMeter * distance;
    occlusion += smoothstep(-0.0002, 0.0010, blocker - rayHeight - 0.00045);
  }
  return clamp(exp(-0.20 * occlusion), 0.22, 1.0);
}

@fragment
fn butterFragment(input: ButterVertexOut) -> @location(0) vec4f {
  if (input.properties.x < 2.0e-5) { discard; }
  let viewMode = i32(round(scene.viewport.w));
  if (viewMode == 1) {
    var thermal = thermalColor(input.properties.y);
    let contour = 0.92 + 0.08 * sin(input.properties.y * PI);
    thermal *= contour;
    return vec4f(pow(aces(thermal * 1.35), vec3f(1.0 / 2.2)), 1.0);
  }
  if (viewMode == 2) {
    let structure = structureColor(input.properties.z);
    return vec4f(pow(aces(structure * 1.2), vec3f(1.0 / 2.2)), 1.0);
  }

  let geometricNormal = normalize(input.worldNormal);
  let crystalOffset = input.worldPosition.y * 310.0;
  let macroNoise = valueNoise(input.worldPosition.xz * vec2f(54.0, 71.0));
  let crystal = valueNoise(input.worldPosition.xz * 620.0 + vec2f(crystalOffset, crystalOffset));
  let fineCrystal = valueNoise(input.worldPosition.xz * vec2f(1450.0, 1120.0));
  let knifeTexture = valueNoise(input.worldPosition.xz * vec2f(230.0, 48.0));
  let epsilon = 0.00028;
  let detailScale = 1800.0;
  let gradientX = valueNoise((input.worldPosition.xz + vec2f(epsilon, 0.0)) * detailScale)
    - valueNoise((input.worldPosition.xz - vec2f(epsilon, 0.0)) * detailScale);
  let gradientZ = valueNoise((input.worldPosition.xz + vec2f(0.0, epsilon)) * detailScale)
    - valueNoise((input.worldPosition.xz - vec2f(0.0, epsilon)) * detailScale);
  let pixelFootprint = max(fwidth(input.worldPosition.x), fwidth(input.worldPosition.z));
  let detailFade = 1.0 - smoothstep(0.35, 1.2, pixelFootprint * detailScale);
  let micro = vec3f(gradientX, 0.0, gradientZ) * detailFade;
  let network = smoothstep(0.02, 0.24, input.properties.z);
  let normal = normalize(geometricNormal + micro * (0.0025 + 0.0065 * network));
  let light = normalize(scene.lightDirection.xyz);
  let view = normalize(scene.cameraPosition.xyz - input.worldPosition);
  let halfVector = normalize(light + view);
  let nDotL = max(dot(normal, light), 0.0);
  let nDotV = max(dot(normal, view), 0.0);
  let hDotV = max(dot(halfVector, view), 0.0);

  let oilPatch = smoothstep(0.54, 0.86, valueNoise(input.worldPosition.xz * 115.0 + vec2f(3.7, 8.1))) * (1.0 - network);
  let pore = smoothstep(0.86, 0.985, fineCrystal) * network;
  var roughness = mix(0.12, 0.52, network) + (crystal - 0.5) * mix(0.025, 0.065, network);
  roughness -= oilPatch * 0.055;
  roughness += pore * 0.06;
  roughness = clamp(roughness, 0.075, 0.62);
  let baseWarmth = saturate((input.properties.y - 8.0) / 34.0);
  var albedo = mix(vec3f(0.84, 0.75, 0.51), vec3f(0.88, 0.73, 0.39), baseWarmth);
  albedo *= 0.975 + (macroNoise - 0.5) * 0.035 * network + (crystal - 0.5) * 0.022;
  albedo *= 1.0 + (knifeTexture - 0.5) * network * 0.018 - pore * 0.025;
  let f0 = vec3f(0.034);
  let fresnel = fresnelSchlick(hDotV, f0);
  let distribution = distributionGGX(normal, halfVector, roughness);
  let geometry = geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
  let specular = distribution * geometry * fresnel / max(4.0 * nDotV * nDotL, 1.0e-4);
  let diffuseWeight = (vec3f(1.0) - fresnel) * (1.0 / PI);
  let wrappedDiffuse = saturate((dot(normal, light) + 0.34) / 1.34);
  let meanFreePath = mix(0.0020, 0.0012, network);
  let thicknessScatter = exp(-input.properties.x / meanFreePath);
  let backScatter = pow(saturate(dot(-view, light)), 3.0) * thicknessScatter;
  let shadow = selfShadow(input.worldPosition);

  let directDiffuse = diffuseWeight * albedo * (0.82 * nDotL + 0.18 * wrappedDiffuse);
  let directSpecular = specular * nDotL;
  var color = (directDiffuse + directSpecular) * vec3f(2.65, 2.58, 2.44) * shadow;
  color += albedo * vec3f(0.43, 0.42, 0.38) * (0.24 + 0.34 * max(geometricNormal.y, 0.0));
  color += albedo * vec3f(1.0, 0.91, 0.70) * backScatter * 0.11;

  color *= exp2(scene.appearance.y);
  color = aces(max(color, vec3f(0.0)));
  color = vignette(color, input.position.xy);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
`;
