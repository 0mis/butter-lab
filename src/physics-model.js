export const BUTTER = Object.freeze({
  density: 911, // kg m^-3
  conductivity: 0.24, // W m^-1 K^-1
  heatCapacity: 2050, // J kg^-1 K^-1 outside the transition bands
  referenceTemperature: 0,
  transitionPeaks: Object.freeze([14.24, 18.34, 31.01]),
  transitionWidths: Object.freeze([1.5, 1.5, 2.0]),
  // Whole-butter latent contributions. The measured milk-fat DSC weights are
  // scaled by 0.811, a representative butter fat fraction.
  latentWeights: Object.freeze([9740, 12537, 21196]), // J kg^-1
  surfaceTension: 0.03, // N m^-1, initial calibration prior
  liquidViscosity40C: 0.03, // Pa s
  refractiveIndex: 1.455,
  initialMass: 0.135, // kg, normalized across compute profiles
});

export const DOMAIN = Object.freeze({
  width: 0.32,
  depth: 0.20,
  layers: 8,
});

export const ENTHALPY_INVERSE_KNOTS = Object.freeze([
  [-20, -41000], [0, 0], [5, 10250.044], [8, 16402.385], [10, 20534.213],
  [12, 25070.503], [14, 32835.731], [16, 42219.588], [18, 51447.2], [20, 62037.212],
  [22, 67284.751], [25, 73577.14], [28, 80672.674], [30, 89435.903], [32, 103330.747],
  [35, 114838.002], [40, 125470.358], [50, 145973], [70, 186973], [100, 248473],
]);

export const QUALITY_PROFILES = Object.freeze({
  efficient: Object.freeze({ width: 192, height: 120, label: "Efficient · 23,040 columns" }),
  balanced: Object.freeze({ width: 256, height: 160, label: "Balanced · 40,960 columns" }),
  high: Object.freeze({ width: 320, height: 200, label: "High · 64,000 columns" }),
});

export const MATERIALS = Object.freeze({
  marble: Object.freeze({
    id: 0,
    label: "Marble",
    contactConductance: 320,
    substrateDiffusivity: 1.15e-6,
    substrateCapacity: 46500,
    setpointRelaxation: 180,
    surfaceMobility: 0.82,
    solarAbsorptivity: 0.48,
  }),
  steel: Object.freeze({
    id: 1,
    label: "Stainless steel",
    contactConductance: 1050,
    substrateDiffusivity: 4.0e-6,
    substrateCapacity: 71000,
    setpointRelaxation: 85,
    surfaceMobility: 1.10,
    solarAbsorptivity: 0.36,
  }),
  wood: Object.freeze({
    id: 2,
    label: "Walnut",
    contactConductance: 58,
    substrateDiffusivity: 1.4e-7,
    substrateCapacity: 9200,
    setpointRelaxation: 420,
    surfaceMobility: 0.58,
    solarAbsorptivity: 0.72,
  }),
  ceramic: Object.freeze({
    id: 3,
    label: "Ceramic",
    contactConductance: 185,
    substrateDiffusivity: 6.5e-7,
    substrateCapacity: 31000,
    setpointRelaxation: 250,
    surfaceMobility: 0.72,
    solarAbsorptivity: 0.55,
  }),
});

export const ENVIRONMENT_PRESETS = Object.freeze({
  kitchen: Object.freeze({ ambient: 22, surface: 22, airflow: 0.2, sunlight: 0, tilt: 0, material: "marble" }),
  summer: Object.freeze({ ambient: 31, surface: 34, airflow: 0.8, sunlight: 820, tilt: 0, material: "marble" }),
  cold: Object.freeze({ ambient: 18, surface: 8, airflow: 0.1, sunlight: 0, tilt: 0, material: "marble" }),
  pan: Object.freeze({ ambient: 24, surface: 58, airflow: 0.2, sunlight: 0, tilt: 2.5, material: "steel" }),
});

export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const smoothstep = (edge0, edge1, value) => {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
};

const transition = (temperature, peak, width) =>
  0.5 * (1 + Math.tanh((temperature - peak) / width));

export function temperatureToEnthalpy(temperature) {
  let result = BUTTER.heatCapacity * (temperature - BUTTER.referenceTemperature);
  for (let index = 0; index < BUTTER.transitionPeaks.length; index += 1) {
    result += BUTTER.latentWeights[index] * transition(
      temperature,
      BUTTER.transitionPeaks[index],
      BUTTER.transitionWidths[index],
    );
  }
  return result;
}

export function enthalpyToTemperature(enthalpy) {
  // The enthalpy curve is monotone. Bisection is slower than a LUT, but this
  // reference function is deterministic and is used only for initialization,
  // tests, and telemetry. The GPU evaluates the same curve in WGSL.
  const totalLatent = BUTTER.latentWeights.reduce((sum, value) => sum + value, 0);
  if (enthalpy <= temperatureToEnthalpy(-20)) return enthalpy / BUTTER.heatCapacity;
  if (enthalpy >= temperatureToEnthalpy(50)) return (enthalpy - totalLatent) / BUTTER.heatCapacity;
  let low = -20;
  let high = 50;
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const middle = 0.5 * (low + high);
    if (temperatureToEnthalpy(middle) < enthalpy) low = middle;
    else high = middle;
  }
  return 0.5 * (low + high);
}

export function enthalpyToTemperatureFast(enthalpy) {
  if (enthalpy <= ENTHALPY_INVERSE_KNOTS[0][1]) return enthalpy / BUTTER.heatCapacity;
  for (let index = 1; index < ENTHALPY_INVERSE_KNOTS.length; index += 1) {
    const [rightTemperature, rightEnthalpy] = ENTHALPY_INVERSE_KNOTS[index];
    const [leftTemperature, leftEnthalpy] = ENTHALPY_INVERSE_KNOTS[index - 1];
    if (enthalpy <= rightEnthalpy) {
      return leftTemperature + (rightTemperature - leftTemperature) *
        ((enthalpy - leftEnthalpy) / (rightEnthalpy - leftEnthalpy));
    }
  }
  const totalLatent = BUTTER.latentWeights.reduce((sum, value) => sum + value, 0);
  return (enthalpy - totalLatent) / BUTTER.heatCapacity;
}

export function transitionFraction(temperature) {
  const total = BUTTER.latentWeights.reduce((sum, value) => sum + value, 0);
  let transitioned = 0;
  for (let index = 0; index < BUTTER.transitionPeaks.length; index += 1) {
    transitioned += BUTTER.latentWeights[index] * transition(
      temperature,
      BUTTER.transitionPeaks[index],
      BUTTER.transitionWidths[index],
    );
  }
  return transitioned / total;
}

export function solidFatContent(temperature) {
  // Monotone calibration prior derived from published butter-oil reference
  // values. Enthalpy transition and mechanical solid-fat content are kept
  // deliberately separate.
  const points = [
    [-10, 0.62],
    [0, 0.54],
    [10, 0.43],
    [20, 0.18],
    [30, 0.05],
    [35, 0.005],
    [40, 0],
    [80, 0],
  ];
  if (temperature <= points[0][0]) return points[0][1];
  for (let index = 1; index < points.length; index += 1) {
    const [rightT, rightValue] = points[index];
    const [leftT, leftValue] = points[index - 1];
    if (temperature <= rightT) {
      const amount = (temperature - leftT) / (rightT - leftT);
      // Smooth interpolation avoids abrupt changes in yield stress.
      const eased = amount * amount * (3 - 2 * amount);
      return leftValue + (rightValue - leftValue) * eased;
    }
  }
  return 0;
}

export function liquidViscosity(temperature) {
  return BUTTER.liquidViscosity40C * Math.exp(0.05 * (40 - temperature));
}

export function yieldStress(temperature) {
  const normalized = clamp(solidFatContent(temperature) / 0.43, 0, 1.45);
  return 50_000 * normalized ** 3;
}

export function radiativeHeatFlux(surfaceTemperature, surroundingsTemperature, emissivity = 0.94) {
  const sigma = 5.670374419e-8;
  const surfaceKelvin = surfaceTemperature + 273.15;
  const surroundingsKelvin = surroundingsTemperature + 273.15;
  return emissivity * sigma * (surroundingsKelvin ** 4 - surfaceKelvin ** 4);
}

export function airflowHeatCoefficient(speed) {
  // Smooth flat-plate-inspired engineering approximation across the UI range.
  // It preserves natural convection at zero airflow and remains within the
  // common 3–25 W m^-2 K^-1 calibration envelope.
  return clamp(4.2 + 7.1 * Math.sqrt(Math.max(0, speed)), 3, 25);
}

export function initialButterHeight(x, z) {
  const halfX = 0.057;
  const halfZ = 0.034;
  const radius = 0.003;
  const qx = Math.abs(x) - (halfX - radius);
  const qz = Math.abs(z) - (halfZ - radius);
  const sdf = Math.hypot(Math.max(qx, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qz), 0) - radius;
  if (sdf >= 0.00055) return 0;
  // A real cut block is mostly planar with a sub-millimetre softened edge.
  const edge = 1 - smoothstep(-0.00055, 0.00055, sdf);
  const blade = 0.00010 * Math.sin((x + halfX) * 145 + 0.38 * Math.sin(z * 72)) +
    0.000035 * Math.sin(x * 690 + z * 41);
  const crown = 0.0188 - 0.00016 * ((x / halfX) ** 2 + (z / halfZ) ** 2);
  return Math.max(0, edge * (crown + blade));
}

export function formatSimulationClock(seconds) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remaining = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(1).padStart(4, "0")}`;
}

export function buildInitialState(width, height, initialTemperature, substrateTemperature) {
  const stride = 12;
  const data = new Float32Array(width * height * stride);
  const dx = DOMAIN.width / (width - 1);
  const dz = DOMAIN.depth / (height - 1);
  const enthalpy = temperatureToEnthalpy(initialTemperature);
  const sfc = solidFatContent(initialTemperature);
  for (let zIndex = 0; zIndex < height; zIndex += 1) {
    const z = (zIndex / (height - 1) - 0.5) * DOMAIN.depth;
    for (let xIndex = 0; xIndex < width; xIndex += 1) {
      const x = (xIndex / (width - 1) - 0.5) * DOMAIN.width;
      const offset = (zIndex * width + xIndex) * stride;
      const h = initialButterHeight(x, z);
      data[offset] = h;
      data[offset + 1] = substrateTemperature;
      data[offset + 2] = initialTemperature;
      data[offset + 3] = sfc;
      for (let layer = 0; layer < DOMAIN.layers; layer += 1) data[offset + 4 + layer] = enthalpy;
    }
  }
  const unscaledMass = computeMass(data, width, height, dx, dz);
  const heightScale = BUTTER.initialMass / unscaledMass;
  let footprintCells = 0;
  for (let index = 0; index < width * height; index += 1) {
    data[index * stride] *= heightScale;
    if (data[index * stride] > 2e-5) footprintCells += 1;
  }
  const modeledMass = computeMass(data, width, height, dx, dz);
  return { data, dx, dz, modeledMass, initialFootprint: footprintCells * dx * dz };
}

export function computeMass(data, width, height, dx, dz) {
  let volume = 0;
  const stride = 12;
  for (let index = 0; index < width * height; index += 1) volume += data[index * stride] * dx * dz;
  return volume * BUTTER.density;
}

export function solarEnergyPartition(irradiance, height, substrateAbsorptivity, opticalDepth = 2e-4) {
  const transmission = Math.exp(-Math.max(0, height) / opticalDepth);
  return {
    butter: Math.max(0, irradiance) * 0.72 * (1 - transmission),
    substrate: Math.max(0, irradiance) * clamp(substrateAbsorptivity, 0, 1) * transmission,
  };
}

export function donorScale(height, dt, qRight, qLeft, qUp, qDown, dx, dz) {
  const outgoing = Math.max(qRight, 0) / dx + Math.max(-qLeft, 0) / dx +
    Math.max(qUp, 0) / dz + Math.max(-qDown, 0) / dz;
  if (outgoing <= 0 || height <= 0) return 1;
  return Math.min(1, 0.9 * height / Math.max(dt * outgoing, 1e-30));
}

export function limitedFaceFlux(rawFlux, donorScaleA, donorScaleB) {
  return rawFlux * (rawFlux >= 0 ? donorScaleA : donorScaleB);
}

export function conservativeHorizontalHeatFace(heightA, heightB, enthalpyA, enthalpyB, spacing) {
  const overlap = Math.min(Math.max(0, heightA), Math.max(0, heightB));
  if (overlap <= 0) return 0;
  return (BUTTER.conductivity / BUTTER.density) * overlap *
    (enthalpyToTemperatureFast(enthalpyA) - enthalpyToTemperatureFast(enthalpyB)) / spacing;
}

export function solveContactEnergy(bottomEnthalpy, height, substrateTemperature, substrateCapacity, conductance, dt) {
  if (height <= 0 || substrateCapacity <= 0 || conductance <= 0 || dt <= 0) return 0;
  const bottomTemperature = enthalpyToTemperature(bottomEnthalpy);
  const difference = substrateTemperature - bottomTemperature;
  const butterToSubstrate = BUTTER.density * height *
    (temperatureToEnthalpy(substrateTemperature) - bottomEnthalpy) / DOMAIN.layers;
  const substrateToButter = substrateCapacity * difference;
  let low = difference < 0 ? Math.max(butterToSubstrate, substrateToButter) : 0;
  let high = difference > 0 ? Math.min(butterToSubstrate, substrateToButter) : 0;
  const residual = (energy) => {
    const butterTemperature = enthalpyToTemperature(
      bottomEnthalpy + DOMAIN.layers * energy / (BUTTER.density * height),
    );
    const updatedSubstrate = substrateTemperature - energy / substrateCapacity;
    return energy - dt * conductance * (updatedSubstrate - butterTemperature);
  };
  for (let iteration = 0; iteration < 48; iteration += 1) {
    const middle = 0.5 * (low + high);
    if (residual(middle) > 0) high = middle;
    else low = middle;
  }
  return 0.5 * (low + high);
}

export function layeredStateAverages(enthalpies) {
  let temperature = 0;
  let transitioned = 0;
  let sfc = 0;
  for (const enthalpy of enthalpies) {
    const layerTemperature = enthalpyToTemperatureFast(enthalpy);
    temperature += layerTemperature;
    transitioned += transitionFraction(layerTemperature);
    sfc += solidFatContent(layerTemperature);
  }
  const count = Math.max(1, enthalpies.length);
  return { temperature: temperature / count, transition: transitioned / count, sfc: sfc / count };
}
