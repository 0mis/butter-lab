

const PI: f32 = 3.141592653589793;
const RHO: f32 = 911.0;
const GRAVITY: f32 = 9.80665;
const BUTTER_K: f32 = 0.24;
const CP: f32 = 2050.0;
const SIGMA: f32 = 5.6703744e-8;
const THERMAL_MIN_C: f32 = -40.0;
const THERMAL_MAX_C: f32 = 120.0;

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

fn finiteOr(value: f32, fallback: f32) -> f32 {
  // WGSL intentionally has no isNaN/isInf built-ins. IEEE-754 exponent bits
  // classify both infinities and NaNs without invoking more floating math.
  let bits = bitcast<u32>(value);
  let nonFinite = (bits & 0x7f800000u) == 0x7f800000u;
  return select(value, fallback, nonFinite);
}

fn finiteClamp(value: f32, low: f32, high: f32, fallback: f32) -> f32 {
  return clamp(finiteOr(value, fallback), low, high);
}

fn transition(t: f32, peak: f32, width: f32) -> f32 {
  // Some mobile shader backends lower large tanh inputs through exponentials.
  // The phase curve is already at its asymptote by |x| = 10, so clamping here
  // prevents Inf/Inf without changing the modeled transition.
  return 0.5 * (1.0 + tanh(clamp((t - peak) / width, -10.0, 10.0)));
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
  let safeValue = finiteOr(value, temperatureToEnthalpy(7.0));
  if (safeValue < 0.0) { return safeValue / CP; }
  if (safeValue < 10250.044) { return mix(0.0, 5.0, safeValue / 10250.044); }
  if (safeValue < 16402.385) { return mix(5.0, 8.0, (safeValue - 10250.044) / 6152.341); }
  if (safeValue < 20534.213) { return mix(8.0, 10.0, (safeValue - 16402.385) / 4131.828); }
  if (safeValue < 25070.503) { return mix(10.0, 12.0, (safeValue - 20534.213) / 4536.290); }
  if (safeValue < 32835.731) { return mix(12.0, 14.0, (safeValue - 25070.503) / 7765.228); }
  if (safeValue < 42219.588) { return mix(14.0, 16.0, (safeValue - 32835.731) / 9383.857); }
  if (safeValue < 51447.200) { return mix(16.0, 18.0, (safeValue - 42219.588) / 9227.612); }
  if (safeValue < 62037.212) { return mix(18.0, 20.0, (safeValue - 51447.200) / 10590.012); }
  if (safeValue < 67284.751) { return mix(20.0, 22.0, (safeValue - 62037.212) / 5247.539); }
  if (safeValue < 73577.140) { return mix(22.0, 25.0, (safeValue - 67284.751) / 6292.389); }
  if (safeValue < 80672.674) { return mix(25.0, 28.0, (safeValue - 73577.140) / 7095.534); }
  if (safeValue < 89435.903) { return mix(28.0, 30.0, (safeValue - 80672.674) / 8763.229); }
  if (safeValue < 103330.747) { return mix(30.0, 32.0, (safeValue - 89435.903) / 13894.844); }
  if (safeValue < 114838.002) { return mix(32.0, 35.0, (safeValue - 103330.747) / 11507.255); }
  if (safeValue < 125470.358) { return mix(35.0, 40.0, (safeValue - 114838.002) / 10632.356); }
  return (safeValue - 43473.0) / CP;
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

fn sanitizeCell(cell: Cell) -> Cell {
  let fallbackEnthalpy = temperatureToEnthalpy(7.0);
  let lowEnthalpy = temperatureToEnthalpy(THERMAL_MIN_C);
  let highEnthalpy = temperatureToEnthalpy(THERMAL_MAX_C);
  var safe: Cell;
  safe.geom = vec4f(
    max(finiteOr(cell.geom.x, 0.0), 0.0),
    finiteClamp(cell.geom.y, THERMAL_MIN_C, THERMAL_MAX_C, 22.0),
    finiteClamp(cell.geom.z, THERMAL_MIN_C, THERMAL_MAX_C, 7.0),
    finiteClamp(cell.geom.w, 0.0, 0.62, solidFatContent(7.0))
  );
  safe.thermal0 = vec4f(
    finiteClamp(cell.thermal0.x, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal0.y, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal0.z, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal0.w, lowEnthalpy, highEnthalpy, fallbackEnthalpy)
  );
  safe.thermal1 = vec4f(
    finiteClamp(cell.thermal1.x, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal1.y, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal1.z, lowEnthalpy, highEnthalpy, fallbackEnthalpy),
    finiteClamp(cell.thermal1.w, lowEnthalpy, highEnthalpy, fallbackEnthalpy)
  );
  return safe;
}


@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<Cell>;
@group(0) @binding(2) var<storage, read_write> destination: array<Cell>;

struct FlowScratch {
  values: vec4f, // raw +x/+z fluxes, reserved, reserved
};
@group(0) @binding(3) var<storage, read_write> flowScratch: array<FlowScratch>;
@group(0) @binding(4) var<storage, read_write> donorScales: array<vec2f>;

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

fn loadCell(x: i32, z: i32) -> Cell { return sanitizeCell(source[cellIndex(x, z)]); }
fn cellHeight(x: i32, z: i32) -> f32 { return loadCell(x, z).geom.x; }

fn mobileLayerWeightAt(cell: Cell, layer: u32) -> f32 {
  let temperature = enthalpyToTemperature(enthalpyAt(cell, layer));
  let network = smoothstep(0.04, 0.20, solidFatContent(temperature));
  let fluidity = 1.0 - network;
  return finiteClamp(fluidity * fluidity, 0.0, 1.0, 0.0);
}

fn mobileLayerState(cell: Cell) -> vec2f {
  let layerDepth = max(cell.geom.x, 0.0) * 0.125;
  var mobileDepth = 0.0;
  var weightSum = 0.0;
  var weightedTemperature = 0.0;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    let temperature = enthalpyToTemperature(enthalpyAt(cell, layer));
    let weight = mobileLayerWeightAt(cell, layer);
    mobileDepth += layerDepth * weight;
    weightSum += weight;
    weightedTemperature += weight * temperature;
  }
  let temperature = select(cell.geom.z, weightedTemperature / max(weightSum, 1.0e-8), weightSum > 1.0e-8);
  return vec2f(mobileDepth, temperature);
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
  let mobileA = mobileLayerState(a);
  let mobileB = mobileLayerState(b);
  if (max(mobileA.x, mobileB.x) < 2.0e-7) { return 0.0; }
  let flowDepth = clamp(0.5 * (mobileA.x + mobileB.x), 0.0, 0.03);
  let weightedTemperature = clamp(
    (mobileA.x * mobileA.y + mobileB.x * mobileB.y) / max(mobileA.x + mobileB.x, 1.0e-8),
    -20.0,
    100.0
  );
  let sfc = solidFatContent(weightedTemperature);
  let liquidViscosity = clamp(0.03 * exp(0.05 * (40.0 - weightedTemperature)), 0.02, 2.5);
  let normalizedSfc = clamp(sfc / 0.43, 0.0, 1.45);
  let yieldStress = 50000.0 * normalizedSfc * normalizedSfc * normalizedSfc;
  let bingham = clamp(yieldStress / (max(flowDepth, 8.0e-6) * abs(drive) + 1.0e-4), 0.0, 1.0);
  let yieldedProfile = max(0.0, 1.0 - 1.5 * bingham + 0.5 * bingham * bingham * bingham);
  let mobility = params.material.w * flowDepth * flowDepth * flowDepth / (3.0 * liquidViscosity);
  let physicalFlux = mobility * yieldedProfile * drive;

  // The raw thin-film coefficient becomes extremely stiff at block depth.
  // Bound the shared face exchange to a local monotone update instead of
  // clipping cell heights after the fact (which would destroy mass).
  let levelingBudget = 0.10 * abs(h0 - h1);
  let slopeActivation = min(abs(slope) * 4.0, 1.0);
  let transportBudget = 0.012 * min(h0, h1) * slopeActivation;
  let maximumFlux = (levelingBudget + transportBudget) * spacing / max(params.timing.x, 1.0e-6);
  return finiteOr(clamp(physicalFlux, -maximumFlux, maximumFlux), 0.0);
}

fn storedFaceFlux(x0: i32, z0: i32, x1: i32, z1: i32, axis: u32) -> f32 {
  if (!inside(x0, z0) || !inside(x1, z1)) { return 0.0; }
  let scratchA = flowScratch[cellIndex(x0, z0)].values;
  return finiteOr(select(scratchA.y, scratchA.x, axis == 0u), 0.0);
}

fn limitedFace(x0: i32, z0: i32, x1: i32, z1: i32, axis: u32) -> f32 {
  if (!inside(x0, z0) || !inside(x1, z1)) { return 0.0; }
  let raw = storedFaceFlux(x0, z0, x1, z1, axis);
  let scalesA = donorScales[cellIndex(x0, z0)];
  let scalesB = donorScales[cellIndex(x1, z1)];
  let donorA = finiteClamp(scalesA.x, 0.0, 1.0, 0.0);
  let donorB = finiteClamp(scalesB.x, 0.0, 1.0, 0.0);
  let receiverA = finiteClamp(scalesA.y, 0.0, 1.0, 0.0);
  let receiverB = finiteClamp(scalesB.y, 0.0, 1.0, 0.0);
  let donor = select(donorB, donorA, raw >= 0.0);
  let receiver = select(receiverA, receiverB, raw >= 0.0);
  return finiteOr(raw * min(donor, receiver), 0.0);
}

fn hydraulicHead(x: i32, z: i32) -> f32 {
  let px = (f32(x) / max(params.grid.x - 1.0, 1.0) - 0.5) * params.timing.y;
  let pz = (f32(z) / max(params.grid.y - 1.0, 1.0) - 0.5) * params.timing.z;
  return max(cellHeight(x, z), 0.0) - params.tilt.x * px - params.tilt.y * pz;
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
  // Only thermally mobile layers are available to a donor during this step;
  // the still-solid crystal network remains coherent instead of being
  // exported with a neighboring liquid film.
  let h = mobileLayerState(loadCell(x, z)).x;
  var theta = 1.0;
  if (outgoing > 0.0) {
    if (h > 0.0) {
      theta = min(1.0, 0.50 * h / max(params.timing.x * outgoing, 1.0e-30));
    } else {
      theta = 0.0;
    }
  }
  donorScales[index] = vec2f(finiteClamp(theta, 0.0, 1.0, 0.0), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn computeReceiverScale(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let z = i32(gid.y);
  if (x >= width() || z >= height()) { return; }
  let index = cellIndex(x, z);
  let qRight = storedFaceFlux(x, z, x + 1, z, 0u);
  let qLeft = storedFaceFlux(x - 1, z, x, z, 0u);
  let qUp = storedFaceFlux(x, z, x, z + 1, 1u);
  let qDown = storedFaceFlux(x, z - 1, x, z, 1u);
  let dx = params.grid.z;
  let dz = params.grid.w;
  let incoming = max(-qRight, 0.0) / dx + max(qLeft, 0.0) / dx
    + max(-qUp, 0.0) / dz + max(qDown, 0.0) / dz;
  let h = max(loadCell(x, z).geom.x, 0.0);
  let px = (f32(x) / max(params.grid.x - 1.0, 1.0) - 0.5) * params.timing.y;
  let pz = (f32(z) / max(params.grid.y - 1.0, 1.0) - 0.5) * params.timing.z;
  let localMaximumHead = max(hydraulicHead(x, z), max(max(hydraulicHead(x - 1, z), hydraulicHead(x + 1, z)),
    max(hydraulicHead(x, z - 1), hydraulicHead(x, z + 1))));
  let heightCap = localMaximumHead + params.tilt.x * px + params.tilt.y * pz + 2.0e-5;
  var receiverTheta = 1.0;
  if (incoming > 0.0) {
    let roomRate = 0.90 * max(heightCap - h, 0.0) / max(params.timing.x, 1.0e-6);
    receiverTheta = clamp(roomRate / incoming, 0.0, 1.0);
  }
  var scales = donorScales[index];
  scales.y = finiteClamp(receiverTheta, 0.0, 1.0, 0.0);
  donorScales[index] = scales;
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
  let newHeight = finiteOr(oldHeight - dt * ((qRight - qLeft) / dx + (qUp - qDown) / dz), oldHeight);

  var centerMobileWeights: array<f32, 8>;
  var leftMobileWeights: array<f32, 8>;
  var rightMobileWeights: array<f32, 8>;
  var downMobileWeights: array<f32, 8>;
  var upMobileWeights: array<f32, 8>;
  var centerMobileSum = 0.0;
  var leftMobileSum = 0.0;
  var rightMobileSum = 0.0;
  var downMobileSum = 0.0;
  var upMobileSum = 0.0;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    centerMobileWeights[layer] = mobileLayerWeightAt(center, layer);
    leftMobileWeights[layer] = mobileLayerWeightAt(left, layer);
    rightMobileWeights[layer] = mobileLayerWeightAt(right, layer);
    downMobileWeights[layer] = mobileLayerWeightAt(down, layer);
    upMobileWeights[layer] = mobileLayerWeightAt(up, layer);
    centerMobileSum += centerMobileWeights[layer];
    leftMobileSum += leftMobileWeights[layer];
    rightMobileSum += rightMobileWeights[layer];
    downMobileSum += downMobileWeights[layer];
    upMobileSum += upMobileWeights[layer];
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
    let centerFraction = centerMobileWeights[layer] / max(centerMobileSum, 1.0e-8);
    let rightFraction = rightMobileWeights[layer] / max(rightMobileSum, 1.0e-8);
    let leftFraction = leftMobileWeights[layer] / max(leftMobileSum, 1.0e-8);
    let upFraction = upMobileWeights[layer] / max(upMobileSum, 1.0e-8);
    let downFraction = downMobileWeights[layer] / max(downMobileSum, 1.0e-8);
    let rightTransport = 8.0 * qRight * rightH * select(rightFraction, centerFraction, qRight >= 0.0);
    let leftTransport = 8.0 * qLeft * leftH * select(centerFraction, leftFraction, qLeft >= 0.0);
    let upTransport = 8.0 * qUp * upH * select(upFraction, centerFraction, qUp >= 0.0);
    let downTransport = 8.0 * qDown * downH * select(centerFraction, downFraction, qDown >= 0.0);
    arealEnthalpy -= dt * ((rightTransport - leftTransport) / dx + (upTransport - downTransport) / dz);

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
    newArealEnthalpy[layer] = finiteOr(arealEnthalpy, oldHeight * hSpecific);
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
  var newSubstrate = finiteClamp(
    center.geom.y + dt * (params.material.y * substrateLap + relaxation + substrateSource),
    THERMAL_MIN_C,
    THERMAL_MAX_C,
    center.geom.y
  );

  if (newHeight > 1.0e-12) {
    // The free-surface solve is implicit in enthalpy, so convection and
    // radiation remain monotone even for a micrometre-scale film.
    let lowEnthalpy = temperatureToEnthalpy(THERMAL_MIN_C);
    let highEnthalpy = temperatureToEnthalpy(THERMAL_MAX_C);
    let topHStar = finiteClamp(newArealEnthalpy[7] / newHeight, lowEnthalpy, highEnthalpy, enthalpyAt(center, 7u));
    let topTStar = finiteClamp(enthalpyToTemperature(topHStar), THERMAL_MIN_C, THERMAL_MAX_C, center.geom.z);
    let airCoefficient = clamp(4.2 + 7.1 * sqrt(max(params.environment.w, 0.0)), 3.0, 25.0);
    var lowT = THERMAL_MIN_C;
    var highT = THERMAL_MAX_C;
    for (var iteration = 0u; iteration < 28u; iteration = iteration + 1u) {
      let middleT = 0.5 * (lowT + highT);
      if (surfaceResidual(middleT, topHStar, newHeight, dt, airCoefficient, solarButter) > 0.0) {
        highT = middleT;
      } else {
        lowT = middleT;
      }
    }
    newArealEnthalpy[7] = finiteOr(
      newHeight * temperatureToEnthalpy(0.5 * (lowT + highT)),
      newHeight * topHStar
    );

    // Backward-Euler contact exchange is solved as one equal-and-opposite
    // energy transfer between the substrate and bottom butter layer.
    let substrateCapacity = max(params.material.z, 1000.0);
    let bottomHStar = finiteClamp(newArealEnthalpy[0] / newHeight, lowEnthalpy, highEnthalpy, enthalpyAt(center, 0u));
    let bottomTEstimate = finiteClamp(enthalpyToTemperature(bottomHStar), THERMAL_MIN_C, THERMAL_MAX_C, center.geom.z);
    var lowContactT = THERMAL_MIN_C;
    var highContactT = THERMAL_MAX_C;
    for (var iteration = 0u; iteration < 24u; iteration = iteration + 1u) {
      let middleContactT = 0.5 * (lowContactT + highContactT);
      if (contactTemperatureResidual(middleContactT, bottomHStar, newHeight, newSubstrate, substrateCapacity, dt) > 0.0) {
        highContactT = middleContactT;
      } else {
        lowContactT = middleContactT;
      }
    }
    let contactTemperature = finiteClamp(0.5 * (lowContactT + highContactT), THERMAL_MIN_C, THERMAL_MAX_C, bottomTEstimate);
    let contactEnergy = finiteOr(RHO * newHeight * (temperatureToEnthalpy(contactTemperature) - bottomHStar) * 0.125, 0.0);
    newArealEnthalpy[0] += 8.0 * contactEnergy / RHO;
    newSubstrate = finiteClamp(newSubstrate - contactEnergy / substrateCapacity, THERMAL_MIN_C, THERMAL_MAX_C, center.geom.y);
  }

  let minimumEnthalpy = temperatureToEnthalpy(THERMAL_MIN_C);
  let maximumEnthalpy = temperatureToEnthalpy(THERMAL_MAX_C);
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    if (newHeight > 1.0e-12) {
      newEnthalpies[layer] = finiteClamp(
        newArealEnthalpy[layer] / newHeight,
        minimumEnthalpy,
        maximumEnthalpy,
        enthalpyAt(center, layer)
      );
    } else {
      newEnthalpies[layer] = ambientEnthalpy;
    }
  }

  var meanTemperature = 0.0;
  var meanSfc = 0.0;
  for (var layer = 0u; layer < 8u; layer = layer + 1u) {
    let layerTemperature = finiteClamp(
      enthalpyToTemperature(newEnthalpies[layer]),
      THERMAL_MIN_C,
      THERMAL_MAX_C,
      center.geom.z
    );
    meanTemperature += layerTemperature;
    meanSfc += solidFatContent(layerTemperature);
  }
  meanTemperature *= 0.125;
  meanSfc *= 0.125;

  var output: Cell;
  output.geom = vec4f(newHeight, newSubstrate, meanTemperature, meanSfc);
  output.thermal0 = vec4f(newEnthalpies[0], newEnthalpies[1], newEnthalpies[2], newEnthalpies[3]);
  output.thermal1 = vec4f(newEnthalpies[4], newEnthalpies[5], newEnthalpies[6], newEnthalpies[7]);
  destination[cellIndex(x, z)] = sanitizeCell(output);
}
