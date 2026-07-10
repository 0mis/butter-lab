import {
  DOMAIN,
  MATERIALS,
  QUALITY_PROFILES,
  buildInitialState,
  temperatureToEnthalpy,
} from "./physics-model.js?v=0.7.0";
import { backgroundShader, butterShader, computeShader } from "./shaders.js?v=0.7.0";

const STATE_STRIDE_FLOATS = 12;
const STATE_STRIDE_BYTES = STATE_STRIDE_FLOATS * 4;
const FLOW_STRIDE_FLOATS = 4;
const FLOW_STRIDE_BYTES = FLOW_STRIDE_FLOATS * 4;
const DONOR_SCALE_BYTES = 8;
const PARAM_BYTES = 7 * 16;
const SCENE_BYTES = 12 * 16;

const GPU_BUFFER = globalThis.GPUBufferUsage;
const GPU_SHADER = globalThis.GPUShaderStage;

export function shouldUseMobileSafeMode(navigatorLike = globalThis.navigator, viewportWidth = globalThis.innerWidth) {
  const userAgent = navigatorLike?.userAgent || "";
  const platform = navigatorLike?.platform || "";
  const touchPoints = Number(navigatorLike?.maxTouchPoints || 0);
  const appleTouchDevice = /iPhone|iPad|iPod/i.test(userAgent) || (platform === "MacIntel" && touchPoints > 1);
  const otherMobileDevice = /Android|Mobile/i.test(userAgent) || (touchPoints > 0 && Number(viewportWidth) <= 820);
  return appleTouchDevice || otherMobileDevice;
}

function withTimeout(promise, milliseconds, message) {
  let timer = 0;
  const expiry = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), milliseconds);
  });
  return Promise.race([promise, expiry]).finally(() => clearTimeout(timer));
}

function normalize3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function mat4PerspectiveZO(fieldOfView, aspect, near, far) {
  const output = new Float32Array(16);
  const f = 1 / Math.tan(fieldOfView / 2);
  output[0] = f / aspect;
  output[5] = f;
  output[10] = far / (near - far);
  output[11] = -1;
  output[14] = (far * near) / (near - far);
  return output;
}

function mat4LookAt(eye, target, up = [0, 1, 0]) {
  const output = new Float32Array(16);
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let length = Math.hypot(zx, zy, zz) || 1;
  zx /= length; zy /= length; zz /= length;
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  length = Math.hypot(xx, xy, xz) || 1;
  xx /= length; xy /= length; xz /= length;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  output[0] = xx; output[1] = yx; output[2] = zx; output[3] = 0;
  output[4] = xy; output[5] = yy; output[6] = zy; output[7] = 0;
  output[8] = xz; output[9] = yz; output[10] = zz; output[11] = 0;
  output[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  output[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  output[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  output[15] = 1;
  return output;
}

function mat4Multiply(a, b) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      output[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3];
    }
  }
  return output;
}

function mat4Invert(matrix) {
  const output = new Float32Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = matrix;
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let determinant = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!determinant) return output;
  determinant = 1 / determinant;
  output[0] = (a11 * b11 - a12 * b10 + a13 * b09) * determinant;
  output[1] = (a02 * b10 - a01 * b11 - a03 * b09) * determinant;
  output[2] = (a31 * b05 - a32 * b04 + a33 * b03) * determinant;
  output[3] = (a22 * b04 - a21 * b05 - a23 * b03) * determinant;
  output[4] = (a12 * b08 - a10 * b11 - a13 * b07) * determinant;
  output[5] = (a00 * b11 - a02 * b08 + a03 * b07) * determinant;
  output[6] = (a32 * b02 - a30 * b05 - a33 * b01) * determinant;
  output[7] = (a20 * b05 - a22 * b02 + a23 * b01) * determinant;
  output[8] = (a10 * b10 - a11 * b08 + a13 * b06) * determinant;
  output[9] = (a01 * b08 - a00 * b10 - a03 * b06) * determinant;
  output[10] = (a30 * b04 - a31 * b02 + a33 * b00) * determinant;
  output[11] = (a21 * b02 - a20 * b04 - a23 * b00) * determinant;
  output[12] = (a11 * b07 - a10 * b09 - a12 * b06) * determinant;
  output[13] = (a00 * b09 - a01 * b07 + a02 * b06) * determinant;
  output[14] = (a31 * b01 - a30 * b03 - a32 * b00) * determinant;
  output[15] = (a20 * b03 - a21 * b01 + a22 * b00) * determinant;
  return output;
}

function transformPoint(matrix, point) {
  const x = point[0], y = point[1], z = point[2], w = point[3];
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w,
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w,
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w,
    matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w,
  ];
}

async function assertShader(device, module, label) {
  if (!module.getCompilationInfo) return;
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) {
    const detail = errors.map((message) => `${label}:${message.lineNum}:${message.linePos} ${message.message}`).join("\n");
    throw new Error(detail);
  }
}

function makeIndexData(width, height) {
  const indices = new Uint32Array((width - 1) * (height - 1) * 6);
  let cursor = 0;
  for (let z = 0; z < height - 1; z += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      const a = z * width + x;
      const b = a + 1;
      const c = a + width;
      const d = c + 1;
      indices[cursor++] = a; indices[cursor++] = c; indices[cursor++] = b;
      indices[cursor++] = b; indices[cursor++] = c; indices[cursor++] = d;
    }
  }
  return indices;
}

function makeInitialDonorScales(width, height) {
  return new Float32Array(width * height * 2).fill(1);
}

export { mat4PerspectiveZO, mat4LookAt, mat4Multiply, mat4Invert, transformPoint };

export class WebGPUButterEngine {
  static async create(canvas, onProgress = () => {}) {
    if (!navigator.gpu) throw new Error("navigator.gpu is not available. Enable hardware acceleration in a current Chrome or Edge browser.");
    onProgress("Selecting the discrete GPU…", 0.13);
    let adapter = null;
    try {
      adapter = await withTimeout(
        navigator.gpu.requestAdapter({ powerPreference: "high-performance" }),
        3500,
        "The high-performance WebGPU adapter request timed out.",
      );
    } catch (error) {
      console.warn(error.message);
    }
    // Headless validation and hybrid-laptop routing can decline or delay a
    // power preference even when a conformant adapter exists. The normal
    // request is a safe fallback; the simulation also autotunes its grid.
    if (!adapter) {
      adapter = await withTimeout(
        navigator.gpu.requestAdapter(),
        6500,
        "WebGPU adapter selection timed out. Restart Chrome or Edge and confirm hardware acceleration is enabled.",
      );
    }
    if (!adapter) throw new Error("No compatible WebGPU adapter was returned by the browser.");
    const devicePromise = adapter.requestDevice();
    let device;
    try {
      device = await withTimeout(
        devicePromise,
        8000,
        "The browser found the GPU but did not create a WebGPU device in time.",
      );
    } catch (error) {
      devicePromise.then((lateDevice) => lateDevice.destroy(), () => {});
      throw error;
    }
    const engine = new WebGPUButterEngine(canvas, adapter, device);
    onProgress("Compiling thermodynamic and rendering kernels…", 0.27);
    try {
      const lossFailure = engine.deviceLossSignal.then((lossError) => {
        if (lossError) throw lossError;
        return new Promise(() => {});
      });
      await withTimeout(
        Promise.race([engine.initialize(onProgress), lossFailure]),
        30000,
        "GPU shader or pipeline initialization timed out. Restart the browser before retrying.",
      );
    } catch (error) {
      engine.destroy();
      throw error;
    }
    return engine;
  }

  constructor(canvas, adapter, device) {
    this.canvas = canvas;
    this.adapter = adapter;
    this.device = device;
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.mobileSafeMode = shouldUseMobileSafeMode();
    this.profileName = this.mobileSafeMode ? "efficient" : "balanced";
    this.profile = QUALITY_PROFILES[this.profileName];
    this.maximumSubsteps = this.mobileSafeMode ? 4 : 24;
    this.telemetryInterval = this.mobileSafeMode ? 1100 : 700;
    this.mobileQueuePending = false;
    this.materialName = "marble";
    this.environment = { ambient: 22, surface: 22, sunlight: 0, airflow: 0.2, tilt: 0 };
    this.heater = { active: false, x: 0, z: 0, power: 3000, radius: 0.018 };
    this.viewMode = 0;
    this.photoMode = false;
    this.simulationTime = 0;
    this.currentState = 0;
    this.readbackPending = false;
    this.reconfiguring = false;
    this.flowStepScale = 1;
    this.initialMass = 0;
    this.initialFootprint = 0;
    this.camera = { yaw: -0.44, pitch: 0.40, distance: 0.25, target: [-0.018, 0.006, 0] };
    this.lastInverseViewProjection = new Float32Array(16);
    this.lastCameraPosition = [0, 0, 0];
    this.depthTexture = null;
    this.msaaColorTexture = null;
    this.resourceGeneration = 0;
    this.deviceLostError = null;
    this.deviceLostHandler = null;
    this.destroyed = false;
    this.reportDeviceError = (error) => {
      if (this.destroyed || this.deviceLostError) return this.deviceLostError;
      this.deviceLostError = error instanceof Error ? error : new Error(String(error));
      console.error("Butter Lab WebGPU failure:", this.deviceLostError);
      this.deviceLostHandler?.(this.deviceLostError);
      return this.deviceLostError;
    };
    this.uncapturedErrorHandler = (event) => {
      event.preventDefault();
      const category = event.error?.constructor?.name || "GPU error";
      const detail = event.error?.message || "An asynchronous WebGPU operation failed.";
      this.reportDeviceError(new Error(`${category}: ${detail}`));
    };
    this.device.addEventListener("uncapturederror", this.uncapturedErrorHandler);
    this.deviceLossSignal = this.device.lost.then((info) => {
      if (this.destroyed || info.reason === "destroyed") return null;
      const detail = info.message ? ` ${info.message}` : "";
      return this.reportDeviceError(new Error(`The WebGPU device was lost (${info.reason || "unknown"}).${detail}`));
    });
  }

  setDeviceLostHandler(handler) {
    this.deviceLostHandler = handler;
    if (this.deviceLostError) handler(this.deviceLostError);
  }

  async initialize(onProgress) {
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.paramsBuffer = this.device.createBuffer({ label: "BL05 parameters", size: PARAM_BYTES, usage: GPU_BUFFER.UNIFORM | GPU_BUFFER.COPY_DST });
    this.sceneBuffer = this.device.createBuffer({ label: "BL05 scene", size: SCENE_BYTES, usage: GPU_BUFFER.UNIFORM | GPU_BUFFER.COPY_DST });

    this.computeLayout = this.device.createBindGroupLayout({
      label: "BL05 compute bind layout",
      entries: [
        { binding: 0, visibility: GPU_SHADER.COMPUTE, buffer: { type: "uniform", minBindingSize: PARAM_BYTES } },
        { binding: 1, visibility: GPU_SHADER.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPU_SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPU_SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPU_SHADER.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.renderLayout = this.device.createBindGroupLayout({
      label: "BL05 render bind layout",
      entries: [
        { binding: 0, visibility: GPU_SHADER.VERTEX | GPU_SHADER.FRAGMENT, buffer: { type: "uniform", minBindingSize: SCENE_BYTES } },
        { binding: 1, visibility: GPU_SHADER.VERTEX | GPU_SHADER.FRAGMENT, buffer: { type: "uniform", minBindingSize: PARAM_BYTES } },
        { binding: 2, visibility: GPU_SHADER.VERTEX | GPU_SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });

    const computeModule = this.device.createShaderModule({ label: "BL05 enthalpy-viscoplastic solver", code: computeShader });
    const backgroundModule = this.device.createShaderModule({ label: "BL05 analytical studio", code: backgroundShader });
    const butterModule = this.device.createShaderModule({ label: "BL05 butter dielectric", code: butterShader });
    await Promise.all([
      assertShader(this.device, computeModule, "compute"),
      assertShader(this.device, backgroundModule, "background"),
      assertShader(this.device, butterModule, "butter"),
    ]);

    onProgress("Creating asynchronous GPU pipelines…", 0.50);
    const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    [this.rawFluxPipeline, this.donorScalePipeline, this.receiverScalePipeline, this.computePipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: "BL05 raw face flux pipeline",
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: "computeRawFaces" },
      }),
      this.device.createComputePipelineAsync({
        label: "BL05 donor scale pipeline",
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: "computeDonorScale" },
      }),
      this.device.createComputePipelineAsync({
        label: "BL05 receiver scale pipeline",
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: "computeReceiverScale" },
      }),
      this.device.createComputePipelineAsync({
        label: "BL05 conservative solver pipeline",
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: "simulate" },
      }),
    ]);
    const renderPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] });
    this.backgroundPipeline = await this.device.createRenderPipelineAsync({
      label: "BL05 countertop pipeline",
      layout: renderPipelineLayout,
      vertex: { module: backgroundModule, entryPoint: "backgroundVertex" },
      fragment: { module: backgroundModule, entryPoint: "backgroundFragment", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
    });
    [this.butterPipeline, this.wallPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        label: "BL05 butter surface pipeline",
        layout: renderPipelineLayout,
        vertex: { module: butterModule, entryPoint: "butterVertex" },
        fragment: { module: butterModule, entryPoint: "butterFragment", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      }),
      this.device.createRenderPipelineAsync({
        label: "BL05 butter cut-wall pipeline",
        layout: renderPipelineLayout,
        vertex: { module: butterModule, entryPoint: "butterWallVertex" },
        fragment: { module: butterModule, entryPoint: "butterFragment", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      }),
    ]);
    [this.backgroundPhotoPipeline, this.butterPhotoPipeline, this.wallPhotoPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        label: "BL05 photo countertop pipeline",
        layout: renderPipelineLayout,
        vertex: { module: backgroundModule, entryPoint: "backgroundVertex" },
        fragment: { module: backgroundModule, entryPoint: "backgroundFragment", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
        multisample: { count: 4 },
      }),
      this.device.createRenderPipelineAsync({
        label: "BL05 photo butter surface pipeline",
        layout: renderPipelineLayout,
        vertex: { module: butterModule, entryPoint: "butterVertex" },
        fragment: { module: butterModule, entryPoint: "butterFragment", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
        multisample: { count: 4 },
      }),
      this.device.createRenderPipelineAsync({
        label: "BL05 photo butter cut-wall pipeline",
        layout: renderPipelineLayout,
        vertex: { module: butterModule, entryPoint: "butterWallVertex" },
        fragment: { module: butterModule, entryPoint: "butterFragment", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
        multisample: { count: 4 },
      }),
    ]);

    onProgress("Allocating conserved material fields…", 0.76);
    await this.createGridResources(this.profile);
    this.resize(true);
    this.updateParameterBuffer(0.02);
    onProgress("Calibrating the initial crystal state…", 0.94);
    await this.device.queue.onSubmittedWorkDone();
  }

  get adapterDescription() {
    const info = this.adapter.info || {};
    return info.description || info.device || "High-performance WebGPU adapter";
  }

  destroyResourceSet(resources) {
    for (const buffer of resources.stateBuffers || []) buffer?.destroy();
    resources.flowBuffer?.destroy();
    resources.donorScaleBuffer?.destroy();
    resources.indexBuffer?.destroy();
    resources.readbackBuffer?.destroy();
  }

  async createGridResources(profile) {
    if (this.readbackPending) throw new Error("Cannot replace the simulation grid while telemetry is being read.");
    const { width, height } = profile;
    const byteLength = width * height * STATE_STRIDE_BYTES;
    const flowByteLength = width * height * FLOW_STRIDE_BYTES;
    const donorByteLength = width * height * DONOR_SCALE_BYTES;
    const initial = buildInitialState(width, height, 7, this.environment.surface);
    const indices = makeIndexData(width, height);
    const next = { stateBuffers: [], flowBuffer: null, donorScaleBuffer: null, indexBuffer: null, readbackBuffer: null, computeBindGroups: [], renderBindGroups: [] };

    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    let creationError = null;
    try {
      next.stateBuffers = [0, 1].map((index) => this.device.createBuffer({
        label: `BL05 state ${index}`,
        size: byteLength,
        usage: GPU_BUFFER.STORAGE | GPU_BUFFER.COPY_DST | GPU_BUFFER.COPY_SRC,
      }));
      this.device.queue.writeBuffer(next.stateBuffers[0], 0, initial.data);
      this.device.queue.writeBuffer(next.stateBuffers[1], 0, initial.data);
      next.flowBuffer = this.device.createBuffer({
        label: "BL05 conservative face flux scratch",
        size: flowByteLength,
        usage: GPU_BUFFER.STORAGE,
      });
      next.donorScaleBuffer = this.device.createBuffer({
        label: "BL05 aggregate donor scales",
        size: donorByteLength,
        usage: GPU_BUFFER.STORAGE | GPU_BUFFER.COPY_SRC | GPU_BUFFER.COPY_DST,
      });
      this.device.queue.writeBuffer(next.donorScaleBuffer, 0, makeInitialDonorScales(width, height));
      next.indexBuffer = this.device.createBuffer({
        label: "BL05 surface topology",
        size: indices.byteLength,
        usage: GPU_BUFFER.INDEX | GPU_BUFFER.COPY_DST,
      });
      this.device.queue.writeBuffer(next.indexBuffer, 0, indices);
      next.readbackBuffer = this.device.createBuffer({
        label: "BL05 telemetry readback",
        size: byteLength + donorByteLength,
        usage: GPU_BUFFER.COPY_DST | GPU_BUFFER.MAP_READ,
      });
      next.computeBindGroups = [
        this.device.createBindGroup({
          label: "BL05 compute A to B",
          layout: this.computeLayout,
          entries: [
            { binding: 0, resource: { buffer: this.paramsBuffer } },
            { binding: 1, resource: { buffer: next.stateBuffers[0] } },
            { binding: 2, resource: { buffer: next.stateBuffers[1] } },
            { binding: 3, resource: { buffer: next.flowBuffer } },
            { binding: 4, resource: { buffer: next.donorScaleBuffer } },
          ],
        }),
        this.device.createBindGroup({
          label: "BL05 compute B to A",
          layout: this.computeLayout,
          entries: [
            { binding: 0, resource: { buffer: this.paramsBuffer } },
            { binding: 1, resource: { buffer: next.stateBuffers[1] } },
            { binding: 2, resource: { buffer: next.stateBuffers[0] } },
            { binding: 3, resource: { buffer: next.flowBuffer } },
            { binding: 4, resource: { buffer: next.donorScaleBuffer } },
          ],
        }),
      ];
      next.renderBindGroups = next.stateBuffers.map((buffer, index) => this.device.createBindGroup({
        label: `BL05 render state ${index}`,
        layout: this.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuffer } },
          { binding: 1, resource: { buffer: this.paramsBuffer } },
          { binding: 2, resource: { buffer } },
        ],
      }));
    } catch (error) {
      creationError = error;
    }

    const validationPromise = this.device.popErrorScope();
    const memoryPromise = this.device.popErrorScope();
    let validationError = null;
    let memoryError = null;
    try {
      [validationError, memoryError] = await Promise.all([validationPromise, memoryPromise]);
    } catch (error) {
      creationError ||= error;
    }
    if (creationError || validationError || memoryError) {
      this.destroyResourceSet(next);
      const gpuError = validationError || memoryError;
      throw creationError || new Error(`WebGPU grid allocation failed: ${gpuError.message}`);
    }
    if (this.destroyed || this.deviceLostError) {
      this.destroyResourceSet(next);
      throw this.deviceLostError || new Error("WebGPU initialization was cancelled.");
    }

    const previous = {
      stateBuffers: this.stateBuffers,
      flowBuffer: this.flowBuffer,
      donorScaleBuffer: this.donorScaleBuffer,
      indexBuffer: this.indexBuffer,
      readbackBuffer: this.readbackBuffer,
    };
    this.stateBuffers = next.stateBuffers;
    this.flowBuffer = next.flowBuffer;
    this.donorScaleBuffer = next.donorScaleBuffer;
    this.indexBuffer = next.indexBuffer;
    this.readbackBuffer = next.readbackBuffer;
    this.computeBindGroups = next.computeBindGroups;
    this.renderBindGroups = next.renderBindGroups;
    this.indexCount = indices.length;
    this.dx = initial.dx;
    this.dz = initial.dz;
    this.initialMass = initial.modeledMass;
    this.initialFootprint = initial.initialFootprint;
    this.flowStepScale = 1;
    this.resourceGeneration += 1;
    this.currentState = 0;
    this.simulationTime = 0;
    this.destroyResourceSet(previous);
  }

  destroyGridResources() {
    this.destroyResourceSet({
      stateBuffers: this.stateBuffers,
      flowBuffer: this.flowBuffer,
      donorScaleBuffer: this.donorScaleBuffer,
      indexBuffer: this.indexBuffer,
      readbackBuffer: this.readbackPending ? null : this.readbackBuffer,
    });
  }

  async setQuality(profileName) {
    if (!QUALITY_PROFILES[profileName] || profileName === this.profileName || this.reconfiguring || this.readbackPending) return false;
    if (this.deviceLostError) throw this.deviceLostError;
    this.reconfiguring = true;
    try {
      // Stop new submissions, then let every command that references the old
      // state buffers retire before destroying and replacing them.
      await withTimeout(
        this.device.queue.onSubmittedWorkDone(),
        8000,
        "The GPU did not finish its previous work before the quality change.",
      );
      const nextProfile = QUALITY_PROFILES[profileName];
      await this.createGridResources(nextProfile);
      this.profileName = profileName;
      this.profile = nextProfile;
      this.resize(true);
      this.updateParameterBuffer(0.02);
      return true;
    } finally {
      this.reconfiguring = false;
    }
  }

  reset() {
    if (this.reconfiguring || this.deviceLostError || this.destroyed) return false;
    const initial = buildInitialState(this.profile.width, this.profile.height, 7, this.environment.surface);
    this.device.queue.writeBuffer(this.stateBuffers[0], 0, initial.data);
    this.device.queue.writeBuffer(this.stateBuffers[1], 0, initial.data);
    this.device.queue.writeBuffer(this.donorScaleBuffer, 0, makeInitialDonorScales(this.profile.width, this.profile.height));
    this.initialMass = initial.modeledMass;
    this.initialFootprint = initial.initialFootprint;
    this.flowStepScale = 1;
    this.currentState = 0;
    this.simulationTime = 0;
    this.updateParameterBuffer(0.02);
    return true;
  }

  setEnvironment(values) { Object.assign(this.environment, values); }
  setMaterial(name) { if (MATERIALS[name]) this.materialName = name; }
  setViewMode(mode) { this.viewMode = mode; }
  setPhotoMode(active) { this.photoMode = active; this.resize(true); }
  setHeater(values) { Object.assign(this.heater, values); }
  updateFlowLimiter(thetaMin, limitedFraction) {
    if (limitedFraction > 0.01 && Number.isFinite(thetaMin)) {
      this.flowStepScale = Math.max(0.025, Math.min(this.flowStepScale, thetaMin * 0.8));
    } else {
      this.flowStepScale = Math.min(1, this.flowStepScale * 1.18 + 0.02);
    }
  }

  updateParameterBuffer(dt) {
    const material = MATERIALS[this.materialName];
    const values = new Float32Array(28);
    values.set([this.profile.width, this.profile.height, this.dx, this.dz], 0);
    values.set([dt, DOMAIN.width, DOMAIN.depth, this.simulationTime], 4);
    values.set([this.environment.ambient, this.environment.surface, this.environment.sunlight, this.environment.airflow], 8);
    values.set([material.contactConductance, material.substrateDiffusivity, material.substrateCapacity, material.surfaceMobility], 12);
    values.set([material.setpointRelaxation, material.solarAbsorptivity, material.id, this.viewMode], 16);
    const tiltRadians = this.environment.tilt * Math.PI / 180;
    values.set([Math.sin(tiltRadians), 0, this.heater.x, this.heater.z], 20);
    values.set([this.heater.power, this.heater.radius, this.heater.active ? 1 : 0, temperatureToEnthalpy(this.environment.ambient)], 24);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, values);
  }

  advance(realSeconds, timeScale) {
    if (this.deviceLostError) throw this.deviceLostError;
    if (this.reconfiguring) return 0;
    if (this.mobileSafeMode && this.mobileQueuePending) return 0;
    const desired = Math.min(Math.max(realSeconds, 0) * timeScale, 0.96);
    const targetDt = 0.04 * this.flowStepScale;
    const requested = Math.min(desired, targetDt * this.maximumSubsteps);
    if (requested <= 0) return 0;
    const substepCount = Math.max(1, Math.min(this.maximumSubsteps, Math.ceil(requested / targetDt)));
    const dt = requested / substepCount;
    this.simulationTime += requested;
    this.updateParameterBuffer(dt);
    const encoder = this.device.createCommandEncoder({ label: "BL05 simulation frame" });
    for (let step = 0; step < substepCount; step += 1) {
      const bindGroup = this.computeBindGroups[this.currentState];
      for (const [label, pipeline] of [
        ["raw faces", this.rawFluxPipeline],
        ["donor scale", this.donorScalePipeline],
        ["receiver scale", this.receiverScalePipeline],
        ["material update", this.computePipeline],
      ]) {
        const pass = encoder.beginComputePass({ label: `BL05 ${label} ${step}` });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.profile.width / 8), Math.ceil(this.profile.height / 8));
        pass.end();
      }
      this.currentState = 1 - this.currentState;
    }
    this.device.queue.submit([encoder.finish()]);
    return requested;
  }

  resize(force = false) {
    const baseScale = this.photoMode ? 1.06 : ({ efficient: 0.70, balanced: 0.84, high: 1.0 }[this.profileName] || 0.84);
    const ratioLimit = this.mobileSafeMode ? 1.0 : 1.25;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, ratioLimit) * baseScale;
    const width = Math.max(2, Math.floor(this.canvas.clientWidth * pixelRatio));
    const height = Math.max(2, Math.floor(this.canvas.clientHeight * pixelRatio));
    if (!force && this.canvas.width === width && this.canvas.height === height) return false;
    const sampleCount = this.photoMode ? 4 : 1;
    const nextDepthTexture = this.device.createTexture({
      label: "BL05 depth",
      size: [width, height],
      format: "depth24plus",
      sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const nextMsaaColorTexture = this.photoMode ? this.device.createTexture({
      label: "BL05 photo MSAA color",
      size: [width, height],
      format: this.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    }) : null;
    const previousDepthTexture = this.depthTexture;
    const previousMsaaColorTexture = this.msaaColorTexture;
    this.canvas.width = width;
    this.canvas.height = height;
    this.depthTexture = nextDepthTexture;
    this.msaaColorTexture = nextMsaaColorTexture;
    previousDepthTexture?.destroy();
    previousMsaaColorTexture?.destroy();
    return true;
  }

  cameraPosition() {
    const { yaw, pitch, distance, target } = this.camera;
    const cosPitch = Math.cos(pitch);
    return [
      target[0] + Math.sin(yaw) * cosPitch * distance,
      target[1] + Math.sin(pitch) * distance,
      target[2] + Math.cos(yaw) * cosPitch * distance,
    ];
  }

  updateSceneBuffer() {
    const eye = this.cameraPosition();
    const aspect = this.canvas.width / this.canvas.height;
    const projection = mat4PerspectiveZO(37 * Math.PI / 180, aspect, 0.01, 3.0);
    const view = mat4LookAt(eye, this.camera.target);
    const viewProjection = mat4Multiply(projection, view);
    const inverse = mat4Invert(viewProjection);
    const light = normalize3([-0.42, 0.78, 0.53]);
    const values = new Float32Array(48);
    values.set(viewProjection, 0);
    values.set(inverse, 16);
    values.set([...eye, 1], 32);
    values.set([...light, 0], 36);
    values.set([this.canvas.width, this.canvas.height, performance.now() / 1000, this.viewMode], 40);
    const material = MATERIALS[this.materialName];
    values.set([material.id, this.photoMode ? -0.04 : -0.14, this.photoMode ? 1 : 0, 0], 44);
    this.device.queue.writeBuffer(this.sceneBuffer, 0, values);
    this.lastInverseViewProjection = inverse;
    this.lastCameraPosition = eye;
  }

  render() {
    if (this.deviceLostError) throw this.deviceLostError;
    if (this.reconfiguring || (this.mobileSafeMode && this.mobileQueuePending)) return false;
    this.resize();
    this.updateParameterBuffer(0.02);
    this.updateSceneBuffer();
    const encoder = this.device.createCommandEncoder({ label: "BL05 render frame" });
    const colorView = this.context.getCurrentTexture().createView();
    const renderView = this.msaaColorTexture?.createView() || colorView;
    const pass = encoder.beginRenderPass({
      label: "BL05 studio pass",
      colorAttachments: [{
        view: renderView,
        resolveTarget: this.msaaColorTexture ? colorView : undefined,
        clearValue: { r: 0.04, g: 0.04, b: 0.035, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.photoMode ? this.backgroundPhotoPipeline : this.backgroundPipeline);
    pass.setBindGroup(0, this.renderBindGroups[this.currentState]);
    pass.draw(3);
    pass.setPipeline(this.photoMode ? this.wallPhotoPipeline : this.wallPipeline);
    pass.setBindGroup(0, this.renderBindGroups[this.currentState]);
    pass.draw(6, this.profile.width * this.profile.height * 4);
    pass.setPipeline(this.photoMode ? this.butterPhotoPipeline : this.butterPipeline);
    pass.setBindGroup(0, this.renderBindGroups[this.currentState]);
    pass.setIndexBuffer(this.indexBuffer, "uint32");
    pass.drawIndexed(this.indexCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    if (this.mobileSafeMode) {
      this.mobileQueuePending = true;
      this.device.queue.onSubmittedWorkDone().then(
        () => { this.mobileQueuePending = false; },
        (error) => {
          this.mobileQueuePending = false;
          this.reportDeviceError(error);
        },
      );
    }
    return true;
  }

  orbit(deltaX, deltaY) {
    this.camera.yaw -= deltaX * 0.0042;
    this.camera.pitch = Math.max(0.18, Math.min(1.15, this.camera.pitch + deltaY * 0.0033));
  }

  zoom(delta) {
    this.camera.distance = Math.max(0.19, Math.min(0.52, this.camera.distance * Math.exp(delta * 0.0011)));
  }

  screenToCounter(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = (1 - (clientY - rect.top) / rect.height) * 2 - 1;
    let near = transformPoint(this.lastInverseViewProjection, [ndcX, ndcY, 0, 1]);
    let far = transformPoint(this.lastInverseViewProjection, [ndcX, ndcY, 1, 1]);
    near = near.map((value, index) => index < 3 ? value / near[3] : value);
    far = far.map((value, index) => index < 3 ? value / far[3] : value);
    const direction = normalize3([far[0] - near[0], far[1] - near[1], far[2] - near[2]]);
    if (Math.abs(direction[1]) < 1e-6) return null;
    const distance = -this.lastCameraPosition[1] / direction[1];
    if (distance <= 0) return null;
    const x = this.lastCameraPosition[0] + direction[0] * distance;
    const z = this.lastCameraPosition[2] + direction[2] * distance;
    if (Math.abs(x) > DOMAIN.width / 2 || Math.abs(z) > DOMAIN.depth / 2) return null;
    return { x, z };
  }

  async readState() {
    if (this.readbackPending || this.reconfiguring) return null;
    if (this.deviceLostError) throw this.deviceLostError;
    this.readbackPending = true;
    const generation = this.resourceGeneration;
    const readbackBuffer = this.readbackBuffer;
    const stateBuffer = this.stateBuffers[this.currentState];
    const byteLength = this.profile.width * this.profile.height * STATE_STRIDE_BYTES;
    const donorByteLength = this.profile.width * this.profile.height * DONOR_SCALE_BYTES;
    try {
      const encoder = this.device.createCommandEncoder({ label: "BL05 telemetry copy" });
      encoder.copyBufferToBuffer(stateBuffer, 0, readbackBuffer, 0, byteLength);
      encoder.copyBufferToBuffer(this.donorScaleBuffer, 0, readbackBuffer, byteLength, donorByteLength);
      this.device.queue.submit([encoder.finish()]);
      await withTimeout(
        readbackBuffer.mapAsync(GPUMapMode.READ),
        4000,
        "GPU telemetry readback timed out.",
      );
      if (generation !== this.resourceGeneration) {
        readbackBuffer.unmap();
        return null;
      }
      const mapped = new Float32Array(readbackBuffer.getMappedRange());
      const state = mapped.slice(0, byteLength / 4);
      const packedScales = mapped.slice(byteLength / 4, (byteLength + donorByteLength) / 4);
      const donorScale = new Float32Array(this.profile.width * this.profile.height);
      for (let index = 0; index < donorScale.length; index += 1) {
        donorScale[index] = Math.min(packedScales[index * 2], packedScales[index * 2 + 1]);
      }
      readbackBuffer.unmap();
      return { state, donorScale };
    } catch (error) {
      try { readbackBuffer.unmap(); } catch { /* The buffer was never mapped. */ }
      throw error;
    } finally {
      this.readbackPending = false;
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.destroyGridResources();
    this.paramsBuffer?.destroy();
    this.sceneBuffer?.destroy();
    this.depthTexture?.destroy();
    this.msaaColorTexture?.destroy();
    this.context?.unconfigure();
    this.device?.removeEventListener("uncapturederror", this.uncapturedErrorHandler);
    this.device?.destroy();
  }
}
