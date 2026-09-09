/**
 * SIH26054 3D Aero Piston Engine Digital Twin Viewport (Three.js + GLSL)
 *
 * Implements:
 *   - Procedural aero piston engine CAD assembly (Cylinder Head, Block, Piston, Conrod, Crankshaft, Exhaust)
 *   - Slider-crank reciprocating kinematic animation synchronized to live RPM
 *   - Custom GLSL Volumetric Thermal Heatmap Shader (CHT & EGT gradient mapping)
 *   - Dynamic Exploded Assembly Slider (0-100%) with internal degradation visualization
 *   - Wear marks on Piston Crown, Journal Bearings, Cylinder Walls
 */

class AeroEngine3D {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.explodedFactor = 0.0;
    this.crankAngle = 0.0;
    this.rpm = 4800.0;
    this.cht = 148.9;   // Default live CAN-bus CHT: 148.9°C
    this.egt = 666.1;   // Default live CAN-bus EGT: 666.1°C
    this.heatmapEnabled = true;
    this.clock = new THREE.Clock();

    this.parts = {};
    this.initialPositions = {};
    this.degradationMarkers = [];

    this.initScene();
    this.initThermalShaders();
    this.buildEngineAssembly();
    this.setupLighting();
    this.setupDegradationCallouts();
    this.animate();

    window.addEventListener('resize', () => this.onResize());
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b14);

    const width = this.canvas.clientWidth || 800;
    const height = this.canvas.clientHeight || 500;

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(3.8, 2.6, 4.8);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 0.4, 0);
    this.controls.maxDistance = 14;
    this.controls.minDistance = 1.5;
  }

  initThermalShaders() {
    // Custom GLSL Volumetric Thermal Shader for CHT & EGT
    const vertexShader = `
      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        vPosition = position;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `;

    const fragmentShader = `
      uniform float uCht;              // Cylinder Head Temp in °C (e.g. 148.9)
      uniform float uEgt;              // Exhaust Gas Temp in °C (e.g. 666.1)
      uniform float uHeatmapEnabled;  // 1.0 = on, 0.0 = off
      uniform float uTime;
      uniform int uPartType;          // 0 = Head, 1 = Block/Fins, 2 = Exhaust

      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      // Smooth multi-stage volumetric thermal colormap: Blue -> Cyan -> Green -> Amber -> Orange -> Crimson -> White-Hot
      vec3 getThermalColor(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.06, 0.22, 0.72); // Blue (Cold/Sump ~70°C)
        vec3 c1 = vec3(0.08, 0.65, 0.88); // Cyan (~100°C)
        vec3 c2 = vec3(0.12, 0.82, 0.38); // Green (~130°C Nominal)
        vec3 c3 = vec3(0.96, 0.82, 0.12); // Yellow/Amber (~150°C CHT)
        vec3 c4 = vec3(0.98, 0.44, 0.06); // Orange (~180°C High)
        vec3 c5 = vec3(0.92, 0.12, 0.08); // Crimson Red (~210°C Redline)
        vec3 c6 = vec3(1.00, 0.94, 0.82); // White-Hot (>600°C Exhaust)

        if (t < 0.18) return mix(c0, c1, t / 0.18);
        if (t < 0.36) return mix(c1, c2, (t - 0.18) / 0.18);
        if (t < 0.54) return mix(c2, c3, (t - 0.36) / 0.18);
        if (t < 0.72) return mix(c3, c4, (t - 0.54) / 0.18);
        if (t < 0.90) return mix(c4, c5, (t - 0.72) / 0.18);
        return mix(c5, c6, (t - 0.90) / 0.10);
      }

      void main() {
        vec3 lightDir = normalize(vec3(0.8, 1.2, 0.7));
        float diff = max(dot(vNormal, lightDir), 0.0) * 0.65 + 0.35;
        vec3 viewDir = normalize(-vWorldPosition);
        float rim = 1.0 - max(dot(vNormal, viewDir), 0.0);
        rim = pow(rim, 3.0) * 0.4;

        if (uHeatmapEnabled > 0.5) {
          float localTemp = 80.0;

          if (uPartType == 0) {
            // Cylinder Head: heavily governed by CHT with hot core center
            float radial = clamp(length(vPosition.xz) / 0.85, 0.0, 1.0);
            localTemp = uCht + (1.0 - radial) * 15.0;
          } else if (uPartType == 1) {
            // Cylinder Block: temperature rises with height towards combustion chamber
            float hFactor = clamp((vPosition.y + 0.7) / 1.4, 0.0, 1.0);
            localTemp = 85.0 + hFactor * (uCht - 85.0);
          } else if (uPartType == 2) {
            // Exhaust Manifold: governed by EGT with combustion gas pulse
            float pulse = sin(uTime * 5.0 + vPosition.x * 4.0) * 8.0;
            localTemp = uEgt * 0.35 + pulse; // scaled for thermal visualization
          }

          // Normalize: 60°C = 0.0, 220°C = 1.0
          float normT = clamp((localTemp - 60.0) / 160.0, 0.0, 1.0);
          vec3 heatColor = getThermalColor(normT);

          // Emissive glow for extreme heat (>180°C CHT or hot exhaust)
          float glow = smoothstep(0.65, 1.0, normT);
          vec3 emissive = heatColor * glow * 0.75;

          gl_FragColor = vec4(heatColor * diff + emissive + rim * vec3(0.3, 0.6, 1.0), 1.0);
        } else {
          // Standard aerospace metal
          vec3 baseColor = (uPartType == 0) ? vec3(0.65, 0.70, 0.75) :
                           (uPartType == 1) ? vec3(0.40, 0.46, 0.54) : vec3(0.35, 0.40, 0.45);
          gl_FragColor = vec4(baseColor * diff + rim * 0.35, 1.0);
        }
      }
    `;

    this.thermalUniforms = {
      uCht: { value: this.cht },
      uEgt: { value: this.egt },
      uHeatmapEnabled: { value: this.heatmapEnabled ? 1.0 : 0.0 },
      uTime: { value: 0.0 },
      uPartType: { value: 0 }
    };

    this.createThermalMaterial = (partType) => {
      const uniforms = {
        uCht: this.thermalUniforms.uCht,
        uEgt: this.thermalUniforms.uEgt,
        uHeatmapEnabled: this.thermalUniforms.uHeatmapEnabled,
        uTime: this.thermalUniforms.uTime,
        uPartType: { value: partType }
      };
      return new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms,
        side: THREE.DoubleSide
      });
    };
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    this.scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x38bdf8, 2.0);
    mainLight.position.set(5, 8, 4);
    mainLight.castShadow = true;
    this.scene.add(mainLight);

    const rimLight = new THREE.DirectionalLight(0x06b6d4, 1.2);
    rimLight.position.set(-5, 4, -4);
    this.scene.add(rimLight);

    this.thermalPointLight = new THREE.PointLight(0xf59e0b, 1.8, 7);
    this.thermalPointLight.position.set(0, 1.3, 0);
    this.scene.add(this.thermalPointLight);

    const gridHelper = new THREE.GridHelper(10, 20, 0x0284c7, 0x1e293b);
    gridHelper.position.y = -1.6;
    this.scene.add(gridHelper);
  }

  buildEngineAssembly() {
    this.engineGroup = new THREE.Group();
    this.scene.add(this.engineGroup);

    // Standard structural materials
    const metalMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      metalness: 0.85,
      roughness: 0.25
    });

    const darkSteelMat = new THREE.MeshStandardMaterial({
      color: 0x243042,
      metalness: 0.92,
      roughness: 0.28
    });

    // 1. Crankcase / Oil Sump (Base)
    const crankcaseGeo = new THREE.BoxGeometry(1.6, 1.1, 1.8);
    this.parts.crankcase = new THREE.Mesh(crankcaseGeo, darkSteelMat);
    this.parts.crankcase.position.set(0, -0.9, 0);
    this.engineGroup.add(this.parts.crankcase);

    // Internal oil gallery sump inspection plane inside crankcase
    const sumpInspectionGeo = new THREE.PlaneGeometry(1.4, 1.5);
    const sumpInspectionMat = new THREE.MeshStandardMaterial({
      color: 0x78350f, // oil varnish / sludge degradation
      roughness: 0.8,
      metalness: 0.1,
      side: THREE.DoubleSide
    });
    const sumpPlane = new THREE.Mesh(sumpInspectionGeo, sumpInspectionMat);
    sumpPlane.rotation.x = Math.PI / 2;
    sumpPlane.position.y = 0.52;
    this.parts.crankcase.add(sumpPlane);

    // 2. Cylinder Block with Cooling Fins (Equipped with Thermal GLSL Shader)
    const blockGroup = new THREE.Group();
    const cylinderGeo = new THREE.CylinderGeometry(0.7, 0.7, 1.4, 32);
    this.blockShaderMat = this.createThermalMaterial(1);
    const cylinderMesh = new THREE.Mesh(cylinderGeo, this.blockShaderMat);
    blockGroup.add(cylinderMesh);

    // Add 6 radial cooling fin discs
    for (let i = 0; i < 6; i++) {
      const finGeo = new THREE.CylinderGeometry(0.95 - (i * 0.03), 0.95 - (i * 0.03), 0.04, 32);
      const finMesh = new THREE.Mesh(finGeo, this.blockShaderMat);
      finMesh.position.y = -0.5 + i * 0.2;
      blockGroup.add(finMesh);
    }
    blockGroup.position.set(0, 0.4, 0);
    this.parts.cylinderBlock = blockGroup;
    this.engineGroup.add(this.parts.cylinderBlock);

    // 3. Cylinder Head (Thermal Monitoring Core with CHT GLSL Shader)
    const headGeo = new THREE.CylinderGeometry(0.85, 0.75, 0.5, 32);
    this.headShaderMat = this.createThermalMaterial(0);
    this.parts.cylinderHead = new THREE.Mesh(headGeo, this.headShaderMat);
    this.parts.cylinderHead.position.set(0, 1.35, 0);
    this.engineGroup.add(this.parts.cylinderHead);

    // Spark plug & valves on head
    const plugGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.35, 16);
    const plugMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.1 });
    const plug = new THREE.Mesh(plugGeo, plugMat);
    plug.position.set(0, 0.35, 0);
    this.parts.cylinderHead.add(plug);

    // Intake & Exhaust Valves
    const valveGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.25, 16);
    const valveMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8 });
    const valveIn = new THREE.Mesh(valveGeo, valveMat);
    valveIn.position.set(0.3, 0.28, 0);
    const valveEx = new THREE.Mesh(valveGeo, valveMat);
    valveEx.position.set(-0.3, 0.28, 0);
    this.parts.cylinderHead.add(valveIn);
    this.parts.cylinderHead.add(valveEx);

    // 4. Piston with Internal Degradation Meshes
    const pistonGroup = new THREE.Group();
    const pistonGeo = new THREE.CylinderGeometry(0.66, 0.66, 0.6, 32);
    const pistonMat = new THREE.MeshStandardMaterial({
      color: 0xcfd8dc,
      metalness: 0.85,
      roughness: 0.2
    });
    const pistonBody = new THREE.Mesh(pistonGeo, pistonMat);
    pistonGroup.add(pistonBody);

    // Piston Crown Carbon Scoring (Degradation Feature)
    const crownGeo = new THREE.CircleGeometry(0.64, 32);
    const crownMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b, // Dark carbon deposits
      roughness: 0.9,
      metalness: 0.1
    });
    const crownScuff = new THREE.Mesh(crownGeo, crownMat);
    crownScuff.rotation.x = -Math.PI / 2;
    crownScuff.position.y = 0.301;
    pistonGroup.add(crownScuff);

    // Piston Rings (Compression Ring, Scraper Ring, Oil Control Ring)
    const ringMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.9, roughness: 0.3 });
    for (let r = 0; r < 3; r++) {
      const ringGeo = new THREE.TorusGeometry(0.662, 0.012, 8, 32);
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = Math.PI / 2;
      ringMesh.position.y = 0.18 - r * 0.08;
      pistonGroup.add(ringMesh);
    }

    // Wristpin
    const pinGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.6, 16);
    const pinMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.95 });
    const wristPin = new THREE.Mesh(pinGeo, pinMat);
    wristPin.rotation.z = Math.PI / 2;
    wristPin.position.y = -0.05;
    pistonGroup.add(wristPin);

    pistonGroup.position.set(0, 0.4, 0);
    this.parts.piston = pistonGroup;
    this.engineGroup.add(this.parts.piston);

    // 5. Connecting Rod
    const conrodGroup = new THREE.Group();
    const rodGeo = new THREE.BoxGeometry(0.12, 1.2, 0.16);
    const rodMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.85, roughness: 0.2 });
    const rodMesh = new THREE.Mesh(rodGeo, rodMat);
    rodMesh.position.y = 0.6;
    conrodGroup.add(rodMesh);
    conrodGroup.position.set(0, -0.6, 0);
    this.parts.conrod = conrodGroup;
    this.engineGroup.add(this.parts.conrod);

    // 6. Crankshaft & Journal Bearing (Degradation Feature)
    const crankGroup = new THREE.Group();
    const crankPinGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.4, 16);
    const crankPin = new THREE.Mesh(crankPinGeo, metalMat);
    crankPin.rotation.x = Math.PI / 2;
    crankPin.position.set(0, 0.35, 0);
    crankGroup.add(crankPin);

    // Bearing Shell Wear Band (Bronze-colored thermal distress scoring)
    const bearingShellGeo = new THREE.CylinderGeometry(0.128, 0.128, 0.2, 16, 1, true);
    const bearingShellMat = new THREE.MeshStandardMaterial({
      color: 0xd97706, // Bronze/copper wear layer exposed
      metalness: 0.8,
      roughness: 0.4
    });
    const bearingShell = new THREE.Mesh(bearingShellGeo, bearingShellMat);
    bearingShell.rotation.x = Math.PI / 2;
    bearingShell.position.set(0, 0.35, 0);
    crankGroup.add(bearingShell);

    const webGeo = new THREE.BoxGeometry(0.4, 0.65, 0.14);
    const webMesh = new THREE.Mesh(webGeo, darkSteelMat);
    webMesh.position.set(0, 0.1, -0.15);
    crankGroup.add(webMesh);

    crankGroup.position.set(0, -0.85, 0);
    this.parts.crankshaft = crankGroup;
    this.engineGroup.add(this.parts.crankshaft);

    // 7. Exhaust Manifold (Equipped with EGT Thermal Shader)
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.5, 1.3, 0),
      new THREE.Vector3(1.2, 1.2, 0.3),
      new THREE.Vector3(1.6, 0.6, 0.5),
      new THREE.Vector3(1.7, -0.2, 0.6)
    ]);
    const exhaustGeo = new THREE.TubeGeometry(curve, 32, 0.14, 16, false);
    this.exhaustShaderMat = this.createThermalMaterial(2);
    this.parts.exhaust = new THREE.Mesh(exhaustGeo, this.exhaustShaderMat);
    this.engineGroup.add(this.parts.exhaust);

    // Store base initial local positions for exploded-view calculation
    for (const [key, part] of Object.entries(this.parts)) {
      this.initialPositions[key] = part.position.clone();
    }
  }

  setupDegradationCallouts() {
    // Floating 3D degradation marker sprite badges that reveal when exploded
    const createMarker = (text, pos, color = 0x38bdf8) => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.roundRect(4, 4, 248, 56, 8);
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.roundRect(4, 4, 248, 56, 8);
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 18px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, 128, 38);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.0 });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      sprite.scale.set(1.4, 0.35, 1);
      this.scene.add(sprite);
      return sprite;
    };

    this.degradationSprites = [
      { sprite: createMarker('Piston Crown Carbon', new THREE.Vector3(1.3, 1.0, 0)), minExplode: 0.25 },
      { sprite: createMarker('Bearing Clearance 0.048mm', new THREE.Vector3(-1.4, -1.2, 0)), minExplode: 0.35 },
      { sprite: createMarker('Cylinder Fin Dissipation', new THREE.Vector3(1.5, 0.5, 0)), minExplode: 0.20 }
    ];
  }

  /**
   * Set exploded assembly view percentage (0% to 100% or 0.0 to 1.0)
   */
  setExplodedView(input) {
    let factor = (input > 1.0) ? (input / 100.0) : input;
    this.explodedFactor = Math.max(0.0, Math.min(1.0, factor));
    const ef = this.explodedFactor;

    // Dynamically offset along physical disassembly vectors
    if (this.parts.cylinderHead) {
      this.parts.cylinderHead.position.y = this.initialPositions.cylinderHead.y + ef * 2.2;
    }
    if (this.parts.cylinderBlock) {
      this.parts.cylinderBlock.position.y = this.initialPositions.cylinderBlock.y + ef * 1.1;
    }
    if (this.parts.exhaust) {
      this.parts.exhaust.position.x = this.initialPositions.exhaust.x + ef * 1.8;
      this.parts.exhaust.position.z = this.initialPositions.exhaust.z + ef * 0.9;
    }
    if (this.parts.crankcase) {
      this.parts.crankcase.position.y = this.initialPositions.crankcase.y - ef * 1.5;
    }
    if (this.parts.crankshaft) {
      this.parts.crankshaft.position.y = this.initialPositions.crankshaft.y - ef * 0.85;
    }

    // Fade in degradation sprite callouts when exploded > 20%
    if (this.degradationSprites) {
      this.degradationSprites.forEach(item => {
        if (ef >= item.minExplode) {
          item.sprite.material.opacity = Math.min(1.0, (ef - item.minExplode) * 3.5);
        } else {
          item.sprite.material.opacity = 0.0;
        }
      });
    }
  }

  /**
   * Updates real-time CAN-bus telemetry inputs (CHT, EGT, RPM)
   */
  updateTelemetryState(data) {
    if (data.telemetry) {
      this.rpm = data.telemetry.rpm || 4800.0;
      this.cht = data.telemetry.cht || 148.9;
      this.egt = data.telemetry.egt || 666.1;

      // Update GLSL shader uniforms in real-time
      if (this.thermalUniforms) {
        this.thermalUniforms.uCht.value = this.cht;
        this.thermalUniforms.uEgt.value = this.egt;
      }

      // Point light intensity adjusts with thermodynamic heat
      if (this.thermalPointLight) {
        const heatNorm = Math.max(0.0, Math.min(1.0, (this.cht - 130.0) / 80.0));
        this.thermalPointLight.intensity = 1.0 + heatNorm * 2.5;
        this.thermalPointLight.color.setHex(this.cht > 185.0 ? 0xef4444 : 0xf59e0b);
      }
    }
  }

  toggleHeatmap(forceState) {
    if (typeof forceState === 'boolean') {
      this.heatmapEnabled = forceState;
    } else {
      this.heatmapEnabled = !this.heatmapEnabled;
    }
    if (this.thermalUniforms) {
      this.thermalUniforms.uHeatmapEnabled.value = this.heatmapEnabled ? 1.0 : 0.0;
    }
    return this.heatmapEnabled;
  }

  resetView() {
    this.camera.position.set(3.8, 2.6, 4.8);
    this.controls.target.set(0, 0.4, 0);
    this.controls.update();
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const elapsedTime = this.clock.getElapsedTime();
    if (this.thermalUniforms) {
      this.thermalUniforms.uTime.value = elapsedTime;
    }

    // Kinematic crankshaft & reciprocating piston cycle
    const crankSpeed = (this.rpm / 60.0) * (2 * Math.PI) * 0.003;
    this.crankAngle += crankSpeed;

    if (this.parts.crankshaft) {
      this.parts.crankshaft.rotation.z = this.crankAngle;
    }

    // Reciprocating slider-crank kinematics
    const r = 0.35;
    const l = 1.1;
    const s = r * Math.cos(this.crankAngle) + Math.sqrt(Math.max(0.01, l * l - r * r * Math.sin(this.crankAngle) * Math.sin(this.crankAngle)));
    const pistonY = s - 0.7 + (this.explodedFactor * 0.6);

    if (this.parts.piston) {
      this.parts.piston.position.y = pistonY;
    }

    if (this.parts.conrod) {
      const rodAngle = Math.asin((-r * Math.sin(this.crankAngle)) / l);
      this.parts.conrod.rotation.z = rodAngle;
      this.parts.conrod.position.x = -r * Math.sin(this.crankAngle) * 0.4;
      this.parts.conrod.position.y = this.initialPositions.conrod.y + (pistonY - 0.4) * 0.5;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}
