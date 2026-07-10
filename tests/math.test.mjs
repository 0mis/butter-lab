import assert from "node:assert/strict";
import {
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4PerspectiveZO,
  transformPoint,
} from "../src/webgpu-engine.js";

const eye = [0.15, 0.13, 0.26];
const target = [0, 0.006, 0];
const view = mat4LookAt(eye, target);
const projection = mat4PerspectiveZO(37 * Math.PI / 180, 16 / 9, 0.01, 3);
const viewProjection = mat4Multiply(projection, view);
const inverse = mat4Invert(viewProjection);
const identity = mat4Multiply(viewProjection, inverse);

for (let column = 0; column < 4; column += 1) {
  for (let row = 0; row < 4; row += 1) {
    const expected = row === column ? 1 : 0;
    assert.ok(Math.abs(identity[column * 4 + row] - expected) < 1e-4, "view-projection inverse is inconsistent");
  }
}

const targetClip = transformPoint(viewProjection, [...target, 1]);
assert.ok(Math.abs(targetClip[0] / targetClip[3]) < 1e-5, "look-at target should be horizontally centered");
assert.ok(Math.abs(targetClip[1] / targetClip[3]) < 1e-5, "look-at target should be vertically centered");
assert.ok(targetClip[2] / targetClip[3] > 0 && targetClip[2] / targetClip[3] < 1, "target depth must use WebGPU's 0..1 clip range");

console.log("Camera and WebGPU clip-space matrix checks passed.");
