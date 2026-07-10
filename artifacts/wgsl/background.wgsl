


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

fn reconstructedHeightAt(x: i32, z: i32) -> f32 {
  let center = max(heightAt(x, z), 0.0) * 4.0;
  let axial = (max(heightAt(x - 1, z), 0.0) + max(heightAt(x + 1, z), 0.0)
    + max(heightAt(x, z - 1), 0.0) + max(heightAt(x, z + 1), 0.0)) * 2.0;
  let diagonal = max(heightAt(x - 1, z - 1), 0.0) + max(heightAt(x + 1, z - 1), 0.0)
    + max(heightAt(x - 1, z + 1), 0.0) + max(heightAt(x + 1, z + 1), 0.0);
  return (center + axial + diagonal) * 0.0625;
}

fn reconstructedHeightNearest(position: vec2f) -> f32 {
  let coordinate = worldToGrid(position);
  return reconstructedHeightAt(i32(round(coordinate.x)), i32(round(coordinate.y)));
}

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
    let blocker = reconstructedHeightNearest(samplePosition);
    let rayHeight = position.y + verticalPerMeter * distance;
    occlusion += smoothstep(-0.0002, 0.0010, blocker - rayHeight - 0.00035);
  }
  return clamp(exp(-0.24 * occlusion), 0.16, 1.0);
}


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
    let centerHeight = reconstructedHeightAt(gridX, gridZ);
    let neighborHeight = max(max(reconstructedHeightAt(gridX - 1, gridZ), reconstructedHeightAt(gridX + 1, gridZ)),
      max(reconstructedHeightAt(gridX, gridZ - 1), reconstructedHeightAt(gridX, gridZ + 1)));
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
