import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { backgroundShader, butterShader, computeShader } from "../src/shaders.js";

const directory = resolve("artifacts", "wgsl");
await mkdir(directory, { recursive: true });
await Promise.all([
  writeFile(resolve(directory, "compute.wgsl"), computeShader, "utf8"),
  writeFile(resolve(directory, "background.wgsl"), backgroundShader, "utf8"),
  writeFile(resolve(directory, "butter.wgsl"), butterShader, "utf8"),
]);
console.log(`Exported WGSL validation artifacts to ${directory}`);
