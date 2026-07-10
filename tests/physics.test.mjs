import assert from "node:assert/strict";
import {
  BUTTER,
  DOMAIN,
  FLOW_STABILITY,
  MATERIALS,
  airflowHeatCoefficient,
  buildInitialState,
  cellStateIsFinite,
  computeMass,
  conservativeHorizontalHeatFace,
  donorScale,
  enthalpyToTemperature,
  enthalpyToTemperatureFast,
  layeredStateAverages,
  limitedFaceFlux,
  mobileLayerState,
  mobileLayerWeights,
  radiativeHeatFlux,
  receiverScale,
  solarEnergyPartition,
  solveContactEnergy,
  solidFatContent,
  stabilityLimitedFaceFlux,
  temperatureToEnthalpy,
  transitionFraction,
  yieldStress,
} from "../src/physics-model.js";

for (let temperature = -10; temperature <= 80; temperature += 0.5) {
  const recovered = enthalpyToTemperature(temperatureToEnthalpy(temperature));
  assert.ok(Math.abs(recovered - temperature) < 1e-5, `enthalpy inversion failed at ${temperature} C`);
}
for (const temperature of [-1e6, -500, -40, 120, 700, 1e6]) {
  assert.ok(Number.isFinite(temperatureToEnthalpy(temperature)), `extreme transition input must remain finite at ${temperature} C`);
}
assert.ok(Number.isNaN(enthalpyToTemperature(NaN)));
assert.ok(Number.isNaN(enthalpyToTemperatureFast(Infinity)));
assert.ok(Number.isNaN(solidFatContent(NaN)));

let maximumFastInverseError = 0;
for (let temperature = -50; temperature <= 300; temperature += 0.1) {
  const recovered = enthalpyToTemperatureFast(temperatureToEnthalpy(temperature));
  maximumFastInverseError = Math.max(maximumFastInverseError, Math.abs(recovered - temperature));
}
assert.ok(maximumFastInverseError < 0.8, `compact GPU inverse LUT error is too high: ${maximumFastInverseError} C`);

let previousEnthalpy = -Infinity;
for (let temperature = -10; temperature <= 80; temperature += 0.1) {
  const enthalpy = temperatureToEnthalpy(temperature);
  assert.ok(enthalpy > previousEnthalpy, "enthalpy curve must remain strictly monotone");
  previousEnthalpy = enthalpy;
}

assert.ok(transitionFraction(5) < 0.02, "cold butter should have little transition enthalpy consumed");
assert.ok(transitionFraction(40) > 0.99, "warm butter should consume almost all transition enthalpy");
assert.ok(solidFatContent(10) > solidFatContent(20));
assert.ok(solidFatContent(20) > solidFatContent(30));
assert.ok(yieldStress(10) > yieldStress(30) * 100, "crystal network should dominate cold rheology");

{
  const height = 0.019;
  const cold = mobileLayerState(height, Array(8).fill(7));
  const basalMelt = mobileLayerState(height, [40, ...Array(7).fill(7)]);
  const halfMelt = mobileLayerState(height, [...Array(4).fill(40), ...Array(4).fill(7)]);
  const liquid = mobileLayerState(height, Array(8).fill(40));
  assert.equal(cold.depth, 0, "a cold crystal network must not become a mobile full-height film");
  assert.ok(basalMelt.depth > height * 0.12 && basalMelt.depth < height * 0.13);
  assert.ok(halfMelt.depth > height * 0.49 && halfMelt.depth < height * 0.51);
  assert.ok(liquid.depth > height * 0.999);

  const temperatures = [40, 35, 30, 25, 20, 15, 10, 7];
  const weights = mobileLayerWeights(temperatures);
  assert.ok(Math.abs(weights.reduce((sum, weight) => sum + weight, 0) - 1) < 1e-12);
  const q = 3.2e-5;
  const enthalpies = temperatures.map(temperatureToEnthalpy);
  const physicalLayerEnergyFlux = weights.reduce(
    (sum, weight, layer) => sum + (8 * q * weight * enthalpies[layer]) / DOMAIN.layers,
    0,
  );
  const expectedMobileEnergyFlux = q * weights.reduce(
    (sum, weight, layer) => sum + weight * enthalpies[layer],
    0,
  );
  assert.ok(Math.abs(physicalLayerEnergyFlux - expectedMobileEnergyFlux) < 1e-12,
    "mobile-layer enthalpy advection must conserve its weighted face energy");
}

assert.ok(radiativeHeatFlux(10, 30) > 0);
assert.ok(radiativeHeatFlux(40, 20) < 0);
assert.ok(airflowHeatCoefficient(4) > airflowHeatCoefficient(0));

for (const material of Object.values(MATERIALS)) {
  assert.ok(material.contactConductance > 0);
  assert.ok(material.substrateCapacity > 0);
  assert.ok(material.substrateDiffusivity > 0);
  assert.ok(material.surfaceMobility > 0);
}

const initial = buildInitialState(128, 80, 7, 22);
const mass = computeMass(initial.data, 128, 80, DOMAIN.width / 127, DOMAIN.depth / 79);
assert.ok(Math.abs(mass - initial.modeledMass) < 1e-8);
assert.ok(mass > 0.05 && mass < 0.20, `modeled butter mass should be plausible, got ${mass} kg`);
assert.ok(initial.initialFootprint > 0.006 && initial.initialFootprint < 0.009);
const firstWetOffset = initial.data.findIndex((value, index) => index % 12 === 0 && value > 2e-5);
const firstWetCell = initial.data.slice(firstWetOffset, firstWetOffset + 12);
assert.equal(cellStateIsFinite(firstWetCell), true);
for (let channel = 0; channel < 12; channel += 1) {
  const withNaN = firstWetCell.slice();
  withNaN[channel] = NaN;
  assert.equal(cellStateIsFinite(withNaN), false, `NaN state lane ${channel} must be rejected`);
  const withInfinity = firstWetCell.slice();
  withInfinity[channel] = Infinity;
  assert.equal(cellStateIsFinite(withInfinity), false, `infinite state lane ${channel} must be rejected`);
}

const liveMasses = [];
for (const [width, height] of [[160, 100], [256, 160], [320, 200]]) {
  const state = buildInitialState(width, height, 7, 22);
  liveMasses.push(state.modeledMass);
  assert.ok(state.initialFootprint > 0.006 && state.initialFootprint < 0.009);
}
for (const liveMass of liveMasses) assert.ok(Math.abs(liveMass - BUTTER.initialMass) < 1e-7);

for (const material of Object.values(MATERIALS)) {
  for (const height of [0, 2e-5, 2e-4, 2e-3]) {
    const solar = solarEnergyPartition(1000, height, material.solarAbsorptivity);
    assert.ok(solar.butter >= 0 && solar.substrate >= 0);
    assert.ok(solar.butter + solar.substrate <= 1000 + 1e-9, "partitioned sunlight cannot exceed incident irradiance");
  }
}

const wetHeatFlux = conservativeHorizontalHeatFace(
  0.001,
  0.003,
  temperatureToEnthalpy(40),
  temperatureToEnthalpy(10),
  0.001,
);
assert.ok(wetHeatFlux > 0);
assert.equal(conservativeHorizontalHeatFace(0.001, 0, temperatureToEnthalpy(40), temperatureToEnthalpy(10), 0.001), 0);
const heatA = 0.001 * temperatureToEnthalpy(40) - 0.04 * wetHeatFlux / 0.001;
const heatB = 0.003 * temperatureToEnthalpy(10) + 0.04 * wetHeatFlux / 0.001;
assert.ok(Math.abs((heatA + heatB) - (0.001 * temperatureToEnthalpy(40) + 0.003 * temperatureToEnthalpy(10))) < 1e-8);

for (const height of [1e-7, 2e-5, 1e-3, 2e-2]) {
  const initialBottomH = temperatureToEnthalpy(5);
  const initialSubstrateT = 60;
  const substrateCapacity = MATERIALS.steel.substrateCapacity;
  const energy = solveContactEnergy(
    initialBottomH,
    height,
    initialSubstrateT,
    substrateCapacity,
    MATERIALS.steel.contactConductance,
    0.04,
  );
  const updatedBottomH = initialBottomH + 8 * energy / (BUTTER.density * height);
  const updatedBottomT = enthalpyToTemperature(updatedBottomH);
  const updatedSubstrateT = initialSubstrateT - energy / substrateCapacity;
  assert.ok(updatedBottomT >= 5 && updatedBottomT <= updatedSubstrateT + 1e-8);
  assert.ok(updatedSubstrateT <= 60 && updatedSubstrateT >= updatedBottomT - 1e-8);
  const before = BUTTER.density * height * initialBottomH / 8 + substrateCapacity * initialSubstrateT;
  const after = BUTTER.density * height * updatedBottomH / 8 + substrateCapacity * updatedSubstrateT;
  assert.ok(Math.abs(after - before) < 1e-6, "implicit contact must conserve equal-and-opposite energy");
}

const layered = layeredStateAverages([
  ...Array(4).fill(temperatureToEnthalpy(0)),
  ...Array(4).fill(temperatureToEnthalpy(30)),
]);
const expectedLayerTransition = 0.5 * (transitionFraction(0) + transitionFraction(30));
assert.ok(Math.abs(layered.transition - expectedLayerTransition) < 1e-6);
assert.ok(Math.abs(layered.transition - transitionFraction(15)) > 0.1, "telemetry must not collapse layered phase state to mean temperature");

{
  const width = 6;
  const height = 5;
  const dx = 0.01;
  const dz = 0.012;
  const dt = 0.04;
  const values = Array.from({ length: width * height }, (_, index) => 8e-4 + (index % 7) * 1.7e-4);
  const qx = Array.from({ length: height }, (_, z) =>
    Array.from({ length: width - 1 }, (_, x) => (((x * 17 + z * 11) % 9) - 4) * 5e-5));
  const qz = Array.from({ length: height - 1 }, (_, z) =>
    Array.from({ length: width }, (_, x) => (((x * 13 + z * 7) % 11) - 5) * 4e-5));
  const scales = values.map((columnHeight, index) => {
    const x = index % width;
    const z = Math.floor(index / width);
    return donorScale(
      columnHeight,
      dt,
      x < width - 1 ? qx[z][x] : 0,
      x > 0 ? qx[z][x - 1] : 0,
      z < height - 1 ? qz[z][x] : 0,
      z > 0 ? qz[z - 1][x] : 0,
      dx,
      dz,
    );
  });
  const updated = values.map((columnHeight, index) => {
    const x = index % width;
    const z = Math.floor(index / width);
    const right = x < width - 1 ? limitedFaceFlux(qx[z][x], scales[index], scales[index + 1]) : 0;
    const left = x > 0 ? limitedFaceFlux(qx[z][x - 1], scales[index - 1], scales[index]) : 0;
    const up = z < height - 1 ? limitedFaceFlux(qz[z][x], scales[index], scales[index + width]) : 0;
    const down = z > 0 ? limitedFaceFlux(qz[z - 1][x], scales[index - width], scales[index]) : 0;
    return columnHeight - dt * ((right - left) / dx + (up - down) / dz);
  });
  assert.ok(updated.every((value) => value >= -1e-12), "aggregate donor scaling must preserve nonnegative height");
  assert.ok(Math.abs(updated.reduce((sum, value) => sum + value, 0) - values.reduce((sum, value) => sum + value, 0)) < 1e-12,
    "shared limited faces must conserve total height");
}

{
  const spacing = DOMAIN.width / 255;
  const dt = 0.04;
  const high = 0.019;
  const low = 0;
  const rawFlux = 1.0;
  const flux = stabilityLimitedFaceFlux(rawFlux, high, low, spacing, dt);
  const transferredHeight = dt * flux / spacing;
  assert.ok(Math.abs(transferredHeight - FLOW_STABILITY.levelingFraction * (high - low)) < 1e-12);
  assert.ok(high - transferredHeight > low + transferredHeight,
    "one face update must not reverse a liquid height jump");

  const isolatedPeakAfter = high - 4 * transferredHeight;
  assert.ok(isolatedPeakAfter >= transferredHeight,
    "four bounded faces must not turn an isolated peak into a checkerboard trough");
  assert.ok(Math.abs(isolatedPeakAfter + 4 * transferredHeight - high) < 1e-12,
    "face stability limiting must remain conservative");
  assert.equal(stabilityLimitedFaceFlux(Infinity, high, low, spacing, dt), 0);

  const depressed = 0.001;
  const localMaximum = 0.019;
  const incomingRate = 4;
  const receive = receiverScale(depressed, localMaximum, dt, incomingRate);
  const receivedHeight = depressed + dt * incomingRate * receive;
  assert.ok(receivedHeight <= localMaximum + FLOW_STABILITY.receiverEpsilon + 1e-12,
    "receiver scaling must prevent several donors from creating a new maximum");
}

assert.equal(BUTTER.latentWeights.length, 3);
assert.equal(DOMAIN.layers, 8);

console.log(`Physics model checks passed. Live-grid masses: ${liveMasses.map((value) => `${(value * 1000).toFixed(2)} g`).join(", ")}.`);
