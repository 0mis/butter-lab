import { WebGPUButterEngine } from "./webgpu-engine.js?v=0.6.0";
import {
  BUTTER,
  DOMAIN,
  ENVIRONMENT_PRESETS,
  MATERIALS,
  QUALITY_PROFILES,
  enthalpyToTemperature,
  enthalpyToTemperatureFast,
  formatSimulationClock,
  solidFatContent,
  transitionFraction,
} from "./physics-model.js?v=0.6.0";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const canvas = $("#lab-canvas");
const bootOverlay = $("#boot-overlay");
const bootStatus = $("#boot-status");
const bootProgress = $("#boot-progress");
const errorPanel = $("#error-panel");
const errorCopy = $("#error-copy");
const engineBadge = $("#engine-badge");
const qualityCopy = $("#quality-copy");
const historyCanvas = $("#history-canvas");

let engine = null;
let playing = true;
let photoMode = false;
let viewMode = 0;
let timeScale = 30;
let lastFrameTime = performance.now();
let lastTelemetryTime = -Infinity;
let frameCounterTime = performance.now();
let framesSinceCounter = 0;
let currentFps = 0;
let history = [];
let initialFootprint = 0;
let telemetryGeneration = 0;
let pointerMode = null;
let pointerLast = null;
let resumeAfterPhoto = false;
let fatalError = false;
let flowThetaMin = 1;
let flowLimitedFraction = 0;

const controls = {
  ambient: $("#ambient-temp"),
  surface: $("#surface-temp"),
  airflow: $("#airflow"),
  sunlight: $("#sunlight"),
  tilt: $("#tilt"),
  heaterPower: $("#heater-power"),
};

const outputs = {
  ambient: $("#ambient-temp-out"),
  surface: $("#surface-temp-out"),
  airflow: $("#airflow-out"),
  sunlight: $("#sunlight-out"),
  tilt: $("#tilt-out"),
  heaterPower: $("#heater-power-out"),
};

function updateSliderFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const fill = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty("--fill", `${fill}%`);
}

function updateControlReadouts() {
  outputs.ambient.value = `${Number(controls.ambient.value).toFixed(1)} °C`;
  outputs.surface.value = `${Number(controls.surface.value).toFixed(1)} °C`;
  outputs.airflow.value = `${Number(controls.airflow.value).toFixed(1)} m/s`;
  outputs.sunlight.value = `${Math.round(Number(controls.sunlight.value))} W/m²`;
  outputs.tilt.value = `${Number(controls.tilt.value).toFixed(1)}°`;
  outputs.heaterPower.value = `${Number(controls.heaterPower.value).toFixed(1)} kW/m²`;
  Object.values(controls).forEach(updateSliderFill);
}

function syncEnvironment() {
  if (!engine) return;
  engine.setEnvironment({
    ambient: Number(controls.ambient.value),
    surface: Number(controls.surface.value),
    airflow: Number(controls.airflow.value),
    sunlight: Number(controls.sunlight.value),
    tilt: Number(controls.tilt.value),
  });
}

function clearPresetSelection() {
  $$(".preset").forEach((button) => {
    button.classList.remove("is-active");
    button.setAttribute("aria-pressed", "false");
  });
}

function selectMaterial(name) {
  $$(".material").forEach((button) => {
    const selected = button.dataset.material === name;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  engine?.setMaterial(name);
}

function applyPreset(name) {
  const preset = ENVIRONMENT_PRESETS[name];
  if (!preset) return;
  controls.ambient.value = preset.ambient;
  controls.surface.value = preset.surface;
  controls.airflow.value = preset.airflow;
  controls.sunlight.value = preset.sunlight;
  controls.tilt.value = preset.tilt;
  selectMaterial(preset.material);
  updateControlReadouts();
  syncEnvironment();
  $$(".preset").forEach((button) => {
    const selected = button.dataset.preset === name;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function setViewMode(nextMode) {
  viewMode = nextMode;
  engine?.setViewMode(nextMode);
  [["#view-camera", 0], ["#view-thermal", 1], ["#view-structure", 2]].forEach(([selector, mode]) => {
    const selected = nextMode === mode;
    $(selector).classList.toggle("is-active", selected);
    $(selector).setAttribute("aria-pressed", String(selected));
  });
}

function setPlaying(active) {
  playing = active;
  $("#play-icon").textContent = active ? "Ⅱ" : "▶";
  $("#play-label").textContent = active ? "Pause" : "Run";
}

function setPhotoMode(active) {
  photoMode = active;
  document.body.classList.toggle("photo-active", active);
  $("#photo-mode").classList.toggle("is-active", active);
  $("#photo-mode").setAttribute("aria-pressed", String(active));
  engine?.setPhotoMode(active);
  if (active) {
    resumeAfterPhoto = playing;
    setPlaying(false);
  } else if (resumeAfterPhoto) {
    setPlaying(true);
  }
}

function resetExperiment() {
  if (!engine?.reset()) return;
  telemetryGeneration += 1;
  history = [];
  initialFootprint = engine?.initialFootprint || 0;
  flowThetaMin = 1;
  flowLimitedFraction = 0;
  lastTelemetryTime = -Infinity;
  setMetricText("#metric-temp", "7.0", "°C");
  setMetricText("#metric-melt", "0.0", "%");
  setMetricText("#metric-mass", "100.0", "%");
  $("#metric-state").textContent = "crystal network intact";
  $("#metric-spread").textContent = "no measurable spread";
  $("#metric-eta").textContent = "estimating transition time";
  drawHistory();
}

function setMetricText(selector, value, unit) {
  $(selector).innerHTML = `${value}<small>${unit}</small>`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "estimating transition time";
  if (seconds < 90) return `transition estimate ${Math.round(seconds)} s`;
  if (seconds < 7200) return `transition estimate ${(seconds / 60).toFixed(1)} min`;
  return "transition not reached in this condition";
}

function estimateTransitionTime() {
  const samples = history.slice(-18).filter((item) => item.melt > 0.01 && item.melt < 0.96);
  if (samples.length < 5) return Infinity;
  const meanTime = samples.reduce((sum, item) => sum + item.time, 0) / samples.length;
  const meanMelt = samples.reduce((sum, item) => sum + item.melt, 0) / samples.length;
  let numerator = 0;
  let denominator = 0;
  for (const item of samples) {
    numerator += (item.time - meanTime) * (item.melt - meanMelt);
    denominator += (item.time - meanTime) ** 2;
  }
  const slope = numerator / Math.max(denominator, 1e-9);
  if (slope <= 1e-6) return Infinity;
  return Math.max(0, (0.95 - samples.at(-1).melt) / slope);
}

function analyzeState(snapshot) {
  if (!engine || !snapshot) return null;
  const data = snapshot.state || snapshot;
  const donorScales = snapshot.donorScale || null;
  const width = engine.profile.width;
  const height = engine.profile.height;
  const cellArea = engine.dx * engine.dz;
  let volume = 0;
  let weightedTemperature = 0;
  let weightedTransition = 0;
  let weightedSfc = 0;
  let footprintCells = 0;
  let maxRadius = 0;
  let thetaMin = 1;
  let wetCells = 0;
  let limitedCells = 0;
  const centerX = (width - 1) / 2;
  const centerZ = (height - 1) / 2;
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 12;
    const h = Math.max(0, data[offset]);
    if (h <= 0) continue;
    volume += h * cellArea;
    weightedTemperature += h * data[offset + 2];
    let layerTransition = 0;
    let layerSfc = 0;
    for (let layer = 0; layer < 8; layer += 1) {
      const layerTemperature = enthalpyToTemperatureFast(data[offset + 4 + layer]);
      layerTransition += transitionFraction(layerTemperature);
      layerSfc += solidFatContent(layerTemperature);
    }
    weightedTransition += h * layerTransition * 0.125;
    weightedSfc += h * layerSfc * 0.125;
    if (h > 2e-5) {
      footprintCells += 1;
      wetCells += 1;
      if (donorScales) {
        const theta = Math.max(0, Math.min(1, donorScales[index]));
        thetaMin = Math.min(thetaMin, theta);
        if (theta < 0.999) limitedCells += 1;
      }
      const x = index % width;
      const z = Math.floor(index / width);
      const radius = Math.hypot((x - centerX) * engine.dx, (z - centerZ) * engine.dz);
      maxRadius = Math.max(maxRadius, radius);
    }
  }
  if (volume <= 0) return null;
  const mass = volume * BUTTER.density;
  const centerOffset = (Math.floor(height / 2) * width + Math.floor(width / 2)) * 12;
  const coreTemperature = 0.5 * (
    enthalpyToTemperature(data[centerOffset + 7]) +
    enthalpyToTemperature(data[centerOffset + 8])
  );
  return {
    mass,
    massBalance: mass / engine.initialMass,
    meanTemperature: weightedTemperature / (volume / cellArea),
    transition: weightedTransition / (volume / cellArea),
    sfc: weightedSfc / (volume / cellArea),
    coreTemperature,
    footprint: footprintCells * cellArea,
    maxRadius,
    thetaMin,
    limitedFraction: wetCells ? limitedCells / wetCells : 0,
  };
}

function updateTelemetry(metrics) {
  if (!metrics || !engine) return;
  if (!initialFootprint) initialFootprint = engine.initialFootprint || metrics.footprint;
  flowThetaMin = metrics.thetaMin;
  flowLimitedFraction = metrics.limitedFraction;
  engine.updateFlowLimiter(flowThetaMin, flowLimitedFraction);
  setMetricText("#metric-temp", metrics.meanTemperature.toFixed(1), "°C");
  $("#metric-core").textContent = `core ${metrics.coreTemperature.toFixed(1)} °C`;
  setMetricText("#metric-melt", (metrics.transition * 100).toFixed(1), "%");
  setMetricText("#metric-mass", (metrics.massBalance * 100).toFixed(2), "%");
  $("#metric-grams").textContent = `${(metrics.mass * 1000).toFixed(1)} g modeled`;
  setMetricText("#metric-area", (metrics.footprint * 10_000).toFixed(1), "cm²");
  const spread = (metrics.footprint / Math.max(initialFootprint, 1e-9) - 1) * 100;
  $("#metric-spread").textContent = spread < 0.5 ? "no measurable spread" : `${spread.toFixed(1)}% larger contact area`;

  if (metrics.sfc > 0.27) $("#metric-state").textContent = "crystal network intact";
  else if (metrics.sfc > 0.12) $("#metric-state").textContent = "softening · viscoplastic creep";
  else if (metrics.sfc > 0.035) $("#metric-state").textContent = "yielding into a continuous puddle";
  else $("#metric-state").textContent = "mostly liquid fat network";

  history.push({ time: engine.simulationTime, temperature: metrics.meanTemperature, melt: metrics.transition });
  if (history.length > 180) history.shift();
  $("#metric-eta").textContent = formatDuration(estimateTransitionTime());
  drawHistory();
}

function drawHistory() {
  const rect = historyCanvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(10, Math.floor(rect.width * ratio));
  const height = Math.max(10, Math.floor(rect.height * ratio));
  if (historyCanvas.width !== width || historyCanvas.height !== height) {
    historyCanvas.width = width;
    historyCanvas.height = height;
  }
  const context = historyCanvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(255,255,255,.07)";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = (row / 4) * height;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  if (history.length < 2) return;
  const firstTime = history[0].time;
  const lastTime = Math.max(history.at(-1).time, firstTime + 1);
  const xFor = (time) => ((time - firstTime) / (lastTime - firstTime)) * width;
  const plot = (valueFor, color, lineWidth) => {
    context.beginPath();
    history.forEach((item, index) => {
      const x = xFor(item.time);
      const y = (1 - valueFor(item)) * (height - 5) + 2.5;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.strokeStyle = color;
    context.lineWidth = lineWidth * ratio;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();
  };
  plot((item) => Math.max(0, Math.min(1, (item.temperature - 5) / 45)), "rgba(236,205,105,.95)", 1.3);
  plot((item) => item.melt, "rgba(116,178,190,.85)", 1.0);
}

async function requestTelemetry() {
  if (!engine) return;
  const generation = telemetryGeneration;
  try {
    const snapshot = await engine.readState();
    if (generation !== telemetryGeneration || !snapshot) return;
    updateTelemetry(analyzeState(snapshot));
  } catch (error) {
    console.warn("Telemetry readback skipped:", error);
  }
}

function updateEngineBadge() {
  engineBadge.classList.add("is-ready");
  const flowStatus = flowLimitedFraction > 0.01 ? ` · flow ×${engine.flowStepScale.toFixed(2)}` : "";
  engineBadge.innerHTML = `<i></i> WebGPU · ${currentFps || "—"} fps${flowStatus}`;
}

function drawInterfacePreview() {
  const ratio = Math.min(window.devicePixelRatio || 1, 1.25);
  canvas.width = Math.floor(canvas.clientWidth * ratio);
  canvas.height = Math.floor(canvas.clientHeight * ratio);
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const background = context.createRadialGradient(width * 0.42, height * 0.38, 10, width * 0.42, height * 0.45, width * 0.82);
  background.addColorStop(0, "#5c5749");
  background.addColorStop(0.44, "#2d2c27");
  background.addColorStop(1, "#0d0d0b");
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);
  const counter = context.createLinearGradient(0, height * 0.25, 0, height);
  counter.addColorStop(0, "rgba(73,73,68,.76)");
  counter.addColorStop(1, "rgba(31,31,29,.95)");
  context.fillStyle = counter;
  context.fillRect(0, height * 0.28, width, height * 0.72);
  context.globalAlpha = 0.13;
  for (let vein = 0; vein < 9; vein += 1) {
    const y = height * (0.30 + vein * 0.078);
    context.strokeStyle = vein % 3 ? "#b7b4aa" : "#555650";
    context.lineWidth = ratio * (vein % 3 ? 0.7 : 1.2);
    context.beginPath();
    context.moveTo(-20 * ratio, y + ((vein * 17) % 23) * ratio);
    context.bezierCurveTo(width * 0.24, y - (22 + vein * 3) * ratio, width * 0.61, y + (10 - vein) * ratio, width + 20 * ratio, y - 18 * ratio);
    context.stroke();
  }
  context.globalAlpha = 1;
  const centerX = width * 0.40;
  const centerY = height * 0.53;
  context.save();
  context.filter = `blur(${14 * ratio}px)`;
  context.fillStyle = "rgba(0,0,0,.54)";
  context.beginPath();
  context.ellipse(centerX + 7 * ratio, centerY + 58 * ratio, 154 * ratio, 34 * ratio, -0.03, 0, Math.PI * 2);
  context.fill();
  context.restore();
  context.save();
  context.filter = `blur(${3 * ratio}px)`;
  context.fillStyle = "rgba(0,0,0,.36)";
  context.beginPath();
  context.ellipse(centerX - 2 * ratio, centerY + 58 * ratio, 126 * ratio, 10 * ratio, -0.02, 0, Math.PI * 2);
  context.fill();
  context.restore();

  const point = (x, y) => [centerX + x * ratio, centerY + y * ratio];
  const backLeft = point(-139, -48);
  const backRight = point(124, -29);
  const frontRight = point(117, 3);
  const frontLeft = point(-132, -14);
  const lowerRight = point(112, 68);
  const lowerLeft = point(-127, 54);
  const endLower = point(119, 53);

  const front = context.createLinearGradient(0, centerY - 8 * ratio, 0, centerY + 72 * ratio);
  front.addColorStop(0, "#e7cc7b");
  front.addColorStop(0.56, "#c7a95c");
  front.addColorStop(1, "#b2944d");
  context.fillStyle = front;
  context.beginPath();
  context.moveTo(...frontLeft);
  context.lineTo(...frontRight);
  context.lineTo(...lowerRight);
  context.lineTo(...lowerLeft);
  context.closePath();
  context.fill();
  context.save();
  context.clip();
  context.lineWidth = 0.55 * ratio;
  for (let cut = 0; cut < 8; cut += 1) {
    const x = centerX + (-112 + cut * 31 + ((cut * 17) % 11)) * ratio;
    context.strokeStyle = `rgba(92,70,31,${0.020 + (cut % 4) * 0.006})`;
    context.beginPath();
    context.moveTo(x, centerY - 12 * ratio);
    context.bezierCurveTo(x - 2 * ratio, centerY + 10 * ratio, x + 3 * ratio, centerY + 34 * ratio, x + 1 * ratio, centerY + 62 * ratio);
    context.stroke();
  }
  context.restore();

  const end = context.createLinearGradient(backRight[0], backRight[1], endLower[0], endLower[1]);
  end.addColorStop(0, "#c7aa62");
  end.addColorStop(1, "#9c7e3f");
  context.fillStyle = end;
  context.beginPath();
  context.moveTo(...backRight);
  context.lineTo(...frontRight);
  context.lineTo(...lowerRight);
  context.lineTo(...endLower);
  context.closePath();
  context.fill();

  const top = context.createLinearGradient(backLeft[0], backLeft[1], frontRight[0], frontRight[1]);
  top.addColorStop(0, "#fff0b8");
  top.addColorStop(0.38, "#efd88e");
  top.addColorStop(1, "#c8a957");
  context.fillStyle = top;
  context.beginPath();
  context.moveTo(backLeft[0] + 5 * ratio, backLeft[1]);
  context.quadraticCurveTo(backLeft[0], backLeft[1], backLeft[0], backLeft[1] + 5 * ratio);
  context.lineTo(frontLeft[0], frontLeft[1] - 4 * ratio);
  context.quadraticCurveTo(frontLeft[0], frontLeft[1], frontLeft[0] + 6 * ratio, frontLeft[1]);
  context.lineTo(frontRight[0] - 5 * ratio, frontRight[1]);
  context.quadraticCurveTo(frontRight[0], frontRight[1], frontRight[0] + 2 * ratio, frontRight[1] - 5 * ratio);
  context.lineTo(backRight[0], backRight[1] + 5 * ratio);
  context.quadraticCurveTo(backRight[0] + 1 * ratio, backRight[1], backRight[0] - 6 * ratio, backRight[1]);
  context.closePath();
  context.fill();

  context.save();
  context.clip();
  context.lineWidth = 0.55 * ratio;
  for (let mark = 0; mark < 15; mark += 1) {
    const offset = (-118 + mark * 17) * ratio;
    context.strokeStyle = `rgba(116,91,42,${0.025 + (mark % 4) * 0.008})`;
    context.beginPath();
    context.moveTo(centerX + offset, centerY - 43 * ratio);
    context.bezierCurveTo(centerX + offset + 7 * ratio, centerY - 26 * ratio, centerX + offset + 4 * ratio, centerY - 8 * ratio, centerX + offset + 13 * ratio, centerY + 5 * ratio);
    context.stroke();
  }
  context.fillStyle = "rgba(113,88,40,.10)";
  for (let pore = 0; pore < 46; pore += 1) {
    const x = centerX + (-121 + ((pore * 71) % 238)) * ratio;
    const y = centerY + (-38 + ((pore * 29) % 34)) * ratio;
    context.fillRect(x, y, 0.7 * ratio, 0.7 * ratio);
  }
  context.restore();

  context.strokeStyle = "rgba(255,247,210,.46)";
  context.lineWidth = 0.8 * ratio;
  context.beginPath();
  context.moveTo(frontLeft[0] + 5 * ratio, frontLeft[1]);
  context.lineTo(frontRight[0] - 6 * ratio, frontRight[1]);
  context.stroke();
}

function frame(now) {
  if (!engine || fatalError) return;
  try {
    const delta = Math.min((now - lastFrameTime) / 1000, 0.05);
    lastFrameTime = now;
    if (playing && !document.hidden) engine.advance(delta, timeScale);
    engine.render();
    $("#sim-clock").textContent = formatSimulationClock(engine.simulationTime);

    framesSinceCounter += 1;
    if (now - frameCounterTime >= 1000) {
      currentFps = Math.round((framesSinceCounter * 1000) / (now - frameCounterTime));
      framesSinceCounter = 0;
      frameCounterTime = now;
      updateEngineBadge();
    }
    if (now - lastTelemetryTime > 700 && !engine.readbackPending && !engine.reconfiguring) {
      lastTelemetryTime = now;
      requestTelemetry();
    }
  } catch (error) {
    console.error(error);
    showFatalError(error);
    return;
  }
  requestAnimationFrame(frame);
}

function bindUi() {
  Object.values(controls).forEach((input) => {
    input.addEventListener("input", () => {
      updateControlReadouts();
      syncEnvironment();
      clearPresetSelection();
      if (input === controls.heaterPower) engine?.setHeater({ power: Number(input.value) * 1000 });
    });
  });
  $$(".preset").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  $$(".material").forEach((button) => button.addEventListener("click", () => {
    selectMaterial(button.dataset.material);
    clearPresetSelection();
  }));
  $$(".quality").forEach((button) => button.addEventListener("click", async () => {
    const profileName = button.dataset.quality;
    if (!engine || engine.readbackPending || engine.reconfiguring) return;
    telemetryGeneration += 1;
    try {
      const changed = await engine.setQuality(profileName);
      if (!changed) return;
      $$(".quality").forEach((item) => {
        const selected = item === button;
        item.classList.toggle("is-active", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
      qualityCopy.textContent = `${QUALITY_PROFILES[profileName].label} · tuned for 6 GB VRAM`;
      history = [];
      initialFootprint = 0;
    } catch (error) {
      console.error(error);
      if (engine?.deviceLostError) showFatalError(engine.deviceLostError);
      else qualityCopy.textContent = `Quality unchanged · ${error.message}`;
    }
  }));

  $("#heater-toggle").addEventListener("click", (event) => {
    const button = event.currentTarget;
    const active = button.getAttribute("aria-checked") !== "true";
    button.setAttribute("aria-checked", String(active));
    controls.heaterPower.disabled = !active;
    document.body.classList.toggle("heater-active", active);
    engine?.setHeater({ active });
  });
  $("#view-camera").addEventListener("click", () => setViewMode(0));
  $("#view-thermal").addEventListener("click", () => setViewMode(1));
  $("#view-structure").addEventListener("click", () => setViewMode(2));
  $("#photo-mode").addEventListener("click", () => setPhotoMode(!photoMode));
  $("#play-button").addEventListener("click", () => setPlaying(!playing));
  $("#reset-button").addEventListener("click", resetExperiment);
  $("#time-scale").addEventListener("change", (event) => { timeScale = Number(event.target.value); });
  $("#science-button").addEventListener("click", () => $("#science-dialog").showModal());
  $("#retry-button").addEventListener("click", () => location.reload());

  canvas.addEventListener("pointerdown", (event) => {
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    pointerLast = [event.clientX, event.clientY];
    const heaterActive = $("#heater-toggle").getAttribute("aria-checked") === "true";
    pointerMode = heaterActive ? "heater" : "orbit";
    if (pointerMode === "heater") {
      const point = engine?.screenToCounter(event.clientX, event.clientY);
      if (point) engine.setHeater(point);
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!pointerMode || !pointerLast || !engine) return;
    if (pointerMode === "orbit") engine.orbit(event.clientX - pointerLast[0], event.clientY - pointerLast[1]);
    else {
      const point = engine.screenToCounter(event.clientX, event.clientY);
      if (point) engine.setHeater(point);
    }
    pointerLast = [event.clientX, event.clientY];
  });
  const releasePointer = () => { pointerMode = null; pointerLast = null; };
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);
  canvas.addEventListener("lostpointercapture", releasePointer);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    engine?.zoom(event.deltaY);
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    const orbitStep = 18;
    const isArrow = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key);
    if (event.shiftKey && isArrow && engine) {
      if ($("#heater-toggle").getAttribute("aria-checked") !== "true") $("#heater-toggle").click();
      const step = 0.005;
      const xDelta = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
      const zDelta = event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0;
      engine.setHeater({
        x: Math.max(-DOMAIN.width / 2, Math.min(DOMAIN.width / 2, engine.heater.x + xDelta)),
        z: Math.max(-DOMAIN.depth / 2, Math.min(DOMAIN.depth / 2, engine.heater.z + zDelta)),
      });
    } else if (event.key === "ArrowLeft") engine?.orbit(-orbitStep, 0);
    else if (event.key === "ArrowRight") engine?.orbit(orbitStep, 0);
    else if (event.key === "ArrowUp") engine?.orbit(0, -orbitStep);
    else if (event.key === "ArrowDown") engine?.orbit(0, orbitStep);
    else if (event.key === "+" || event.key === "=") engine?.zoom(-80);
    else if (event.key === "-" || event.key === "_") engine?.zoom(80);
    else if (event.key.toLowerCase() === "h" && !event.repeat) $("#heater-toggle").click();
    else if (event.code === "Space" && !event.repeat) setPlaying(!playing);
    else return;
    event.preventDefault();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && photoMode) setPhotoMode(false);
    if (event.code === "Space" && event.target === document.body) { event.preventDefault(); setPlaying(!playing); }
  });
  window.addEventListener("resize", () => { engine?.resize(true); drawHistory(); });
  document.addEventListener("visibilitychange", () => { lastFrameTime = performance.now(); });
}

function showFatalError(error) {
  if (fatalError) return;
  fatalError = true;
  telemetryGeneration += 1;
  playing = false;
  bootOverlay.classList.add("is-hidden");
  errorPanel.hidden = false;
  engineBadge.classList.remove("is-ready");
  engineBadge.innerHTML = "<i></i> GPU stopped";
  const message = error instanceof Error ? error.message : String(error);
  errorCopy.textContent = `${message} This PC has a compatible GTX 1660 Ti and current driver, so also confirm that Chrome hardware acceleration is enabled.`;
  requestAnimationFrame(() => errorPanel.focus());
}

async function initialize() {
  updateControlReadouts();
  bindUi();
  if (new URLSearchParams(location.search).has("ui-preview")) {
    drawInterfacePreview();
    bootOverlay.classList.add("is-hidden");
    bootOverlay.style.display = "none";
    engineBadge.classList.add("is-ready");
    engineBadge.innerHTML = "<i></i> Interface preview";
    qualityCopy.textContent = "Balanced · 40,960 thermodynamic columns";
    window.addEventListener("resize", drawInterfacePreview);
    return;
  }
  try {
    engine = await WebGPUButterEngine.create(canvas, (message, progress) => {
      bootStatus.textContent = message;
      bootProgress.style.width = `${Math.round(progress * 100)}%`;
    });
    engine.setDeviceLostHandler((error) => showFatalError(error));
    if (fatalError || engine.deviceLostError) return;
    bootStatus.textContent = "Material state ready.";
    bootProgress.style.width = "100%";
    qualityCopy.textContent = `${QUALITY_PROFILES.balanced.label} · ${engine.adapterDescription}`;
    syncEnvironment();
    engine.setMaterial("marble");
    engine.setHeater({ power: Number(controls.heaterPower.value) * 1000 });
    updateEngineBadge();
    setTimeout(() => bootOverlay.classList.add("is-hidden"), 280);
    lastFrameTime = performance.now();
    requestAnimationFrame(frame);
  } catch (error) {
    console.error(error);
    showFatalError(error);
  }
}

initialize();
