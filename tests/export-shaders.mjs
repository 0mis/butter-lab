import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { backgroundShader, butterShader, computeShader } from "../src/shaders.js";

const output = resolve(import.meta.dirname, "..", "artifacts", "wgsl");
await mkdir(output, { recursive: true });
await Promise.all([
  writeFile(resolve(output, "background.wgsl"), backgroundShader),
  writeFile(resolve(output, "butter.wgsl"), butterShader),
  writeFile(resolve(output, "compute.wgsl"), computeShader),
]);

console.log("Exported validated WGSL snapshots.");
