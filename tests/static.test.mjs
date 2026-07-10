import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const html = await readFile(resolve(root, "index.html"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
assert.deepEqual(duplicates, [], `duplicate HTML ids: ${duplicates.join(", ")}`);
assert.match(html, /id="lab-canvas"[\s\S]*?tabindex="0"/);
assert.match(html, /id="heater-toggle"[\s\S]*?aria-label="Localized heater"/);
assert.match(html, /id="error-panel"[\s\S]*?role="alert"/);
for (const group of ["view-camera", "photo-mode"]) {
  assert.match(html, new RegExp(`id="${group}"[^>]*aria-pressed=`));
}

for (const file of [
  "styles.css",
  "src/app.js",
  "src/webgpu-engine.js",
  "src/physics-model.js",
  "src/shaders.js",
  "launch.ps1",
  "Launch Butter Lab.cmd",
  "README.md",
  "SOURCES.md",
]) {
  await access(resolve(root, file));
}

for (const id of [
  "lab-canvas",
  "engine-badge",
  "ambient-temp",
  "surface-temp",
  "heater-toggle",
  "history-canvas",
  "boot-overlay",
  "error-panel",
]) {
  assert.ok(ids.includes(id), `missing required interface id: ${id}`);
}

const engine = await readFile(resolve(root, "src", "webgpu-engine.js"), "utf8");
assert.match(engine, /requestAdapter\(\{ powerPreference: "high-performance" \}\)/);
assert.match(engine, /dispatchWorkgroups/);
assert.match(engine, /GPUMapMode\.READ/);
assert.match(engine, /flowStepScale/);
assert.match(engine, /binding: 3/);
assert.match(engine, /wallPipeline/);
assert.match(engine, /multisample: \{ count: 4 \}/);

const shaders = await readFile(resolve(root, "src", "shaders.js"), "utf8");
assert.match(shaders, /@compute @workgroup_size\(8, 8, 1\)/);
assert.match(shaders, /specific enthalpy/);
assert.match(shaders, /yieldedProfile/);
assert.match(shaders, /radiationFlux/);
assert.match(shaders, /fn computeRawFaces/);
assert.match(shaders, /fn computeDonorScale/);
assert.match(shaders, /fn computeReceiverScale/);
assert.match(shaders, /fn horizontalHeatFace/);
assert.match(shaders, /fn contactTemperatureResidual/);
assert.match(shaders, /fn mobileLayerState/);
assert.match(shaders, /8\.0 \* qRight \* rightH/);
assert.match(shaders, /fn reconstructedHeightAt/);
assert.match(shaders, /fn butterWallVertex/);
assert.match(shaders, /cell\.geom\.w > 0\.12/);
assert.doesNotMatch(shaders, /h - neighborH > 4\.5e-4/);
assert.doesNotMatch(shaders, /SURFACE_TENSION/);
assert.doesNotMatch(shaders, /newHeight = clamp/);

const launcher = await readFile(resolve(root, "launch.ps1"), "utf8");
assert.match(launcher, /\$RootPrefix/);
assert.match(launcher, /Content-Security-Policy/);
assert.match(launcher, /Forbidden host/);
assert.match(launcher, /ReceiveTimeout = 5000/);

const styles = await readFile(resolve(root, "styles.css"), "utf8");
assert.match(styles, /:focus-visible/);
assert.match(styles, /prefers-reduced-motion/);
assert.match(styles, /max-height: 599px/);

console.log(`Static contract checks passed for ${ids.length} unique interface ids.`);
