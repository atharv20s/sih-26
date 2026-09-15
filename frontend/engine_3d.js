/**
 * SIH26054 3D Command & Control Visualization Hub (Three.js + GLSL)
 *
 * Implements:
 *   1. Complete TAPAS-BH-201 MALE UAV Airframe in semi-transparent X-Ray style
 *      with internal avionics, wiring harness, twin tail booms, and pusher propeller.
 *   2. Detailed 4-Cylinder Aero Piston Engine (Rotax 914 F) with GLSL thermal heatmap.
 *   3. Textured 3D ALT/BATTERY PACK with charge indicator (100%).
 *   4. Schematic animated 3D FUEL & OIL SYSTEM LOOP with circulating fluid particles.
 *   5. Pinned 3D dynamic data overlays (Airframe Vibration, RPM/CHT/EGT, Battery Charge).
 *   6. View Manager camera focus modes: AIRCRAFT OVERVIEW, PROPULSION UNIT, SYSTEM DIAGRAMS.
 */

class AeroEngine3D {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.explodedFactor = 0.0;
    this.crankAngle = 0.0;
    this.propellerAngle = 0.0;
    this.rpm = 4535.4;
    this.cht = 148.6;
    this.egt = 667.7;
    this.heatmapEnabled = true;
    this.clock = new THREE.Clock();
    this.viewMode = 'overview';
    this.isTransitioning = false;

    this.parts = {};
    this.initialPositions = {};
    this.fluidParticles = [];
    this.calloutSprites = [];

    // Raycasted hotspot interaction state
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.proxyMeshes = [];

    this.targetCameraPos = new THREE.Vector3(0, 3.4, 7.2);
    this.targetLookAt = new THREE.Vector3(0.1, 0.1, 0);

    // Native clipping planes for internal cross-section inspection
    this.clipPlanes = {
      x: new THREE.Plane(new THREE.Vector3(1, 0, 0), 4.0),
      y: new THREE.Plane(new THREE.Vector3(0, 1, 0), 4.0),
      z: new THREE.Plane(new THREE.Vector3(0, 0, 1), 4.0),
    };
    this.clipEnabled = { x: false, y: false, z: false };

    this.initScene();
    this.buildEnvironmentMap();
    this.initPostProcessing();
    this.initThermalShaders();
    this.buildTapasAirframe();
    this.buildEngineAssembly();
    this.buildBatteryPack();
    this.buildFuelOilLoop();
    this.setupLighting();
    this.applyShadowCasters();
    this.setupSensorNodes();
    this.setupHotspots();
    this.setupPinnedCallouts();
    this.animate();

    window.addEventListener('resize', () => this.onResize());
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060911);

    const width = this.canvas.clientWidth || 900;
    const height = this.canvas.clientHeight || 550;

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.copy(this.targetCameraPos);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.localClippingEnabled = true;
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(this.targetLookAt);
    this.controls.maxDistance = 25;
    this.controls.minDistance = 0.5;
    this.controls.enableZoom = true;
    this.controls.zoomSpeed = 1.4;
    this.controls.enableRotate = true;
    this.controls.rotateSpeed = 0.8;
    this.controls.enablePan = true;
    this.controls.panSpeed = 0.8;

    // Immediately stop automated camera lerp as soon as user drags or zooms
    const cancelTransition = () => {
      this.isTransitioning = false;
    };
    this.controls.addEventListener('start', cancelTransition);
    this.canvas.addEventListener('wheel', cancelTransition, { passive: true });
    this.canvas.addEventListener('pointerdown', cancelTransition, { passive: true });
    this.canvas.addEventListener('touchstart', cancelTransition, { passive: true });
  }

  initThermalShaders() {
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
      uniform float uCht;              // Cylinder Head Temp in °C (148.6)
      uniform float uEgt;              // Exhaust Gas Temp in °C (667.7)
      uniform float uHeatmapEnabled;
      uniform float uTime;
      uniform int uPartType;

      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      vec3 getThermalColor(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.06, 0.22, 0.72); // Blue (Sump ~70°C)
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
          float localTemp = 85.0;

          if (uPartType == 0) {
            float radial = clamp(length(vPosition.xz) / 0.85, 0.0, 1.0);
            localTemp = uCht + (1.0 - radial) * 16.0;
          } else if (uPartType == 1) {
            float hFactor = clamp((vPosition.y + 0.7) / 1.4, 0.0, 1.0);
            localTemp = 85.0 + hFactor * (uCht - 85.0);
          } else if (uPartType == 2) {
            float pulse = sin(uTime * 5.0 + vPosition.x * 4.0) * 8.0;
            localTemp = uEgt * 0.35 + pulse;
          }

          float normT = clamp((localTemp - 60.0) / 160.0, 0.0, 1.0);
          vec3 heatColor = getThermalColor(normT);
          float glow = smoothstep(0.65, 1.0, normT);
          vec3 emissive = heatColor * glow * 0.75;

          gl_FragColor = vec4(heatColor * diff + emissive + rim * vec3(0.3, 0.6, 1.0), 1.0);
        } else {
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
      return new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          uCht: this.thermalUniforms.uCht,
          uEgt: this.thermalUniforms.uEgt,
          uHeatmapEnabled: this.thermalUniforms.uHeatmapEnabled,
          uTime: this.thermalUniforms.uTime,
          uPartType: { value: partType }
        },
        side: THREE.DoubleSide,
        clipping: true
      });
    };
  }

  // -------------------------------------------------------------------------
  // 1. Semi-Transparent X-Ray TAPAS MALE UAV Airframe
  // -------------------------------------------------------------------------
  buildTapasAirframe() {
    this.uavGroup = new THREE.Group();
    this.uavGroup.position.set(0.1, 0.95, -0.6);
    this.uavGroup.rotation.y = -0.32;
    this.scene.add(this.uavGroup);

    // Translucent X-ray fuselage material
    const xrayMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.32,
      roughness: 0.12,
      metalness: 0.88,
      transmission: 0.72,
      ior: 1.25,
      side: THREE.DoubleSide
    });

    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x0ea5e9,
      wireframe: true,
      transparent: true,
      opacity: 0.25
    });

    // Aerodynamic TAPAS BH-201 Fuselage Body Curve
    const bodyCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-3.2, 0.05, 0),    // FLIR Sensor Pod Chin
      new THREE.Vector3(-2.4, 0.28, 0),    // Forward Radome Transition
      new THREE.Vector3(-1.6, 0.42, 0),    // SATCOM Bulbous Upper Bulge
      new THREE.Vector3(0.0, 0.32, 0),     // Mid Fuselage Payload/Fuel Bay
      new THREE.Vector3(1.6, 0.22, 0),     // Aft Engine Bay
      new THREE.Vector3(2.4, 0.08, 0)      // Pusher Propeller Mount
    ]);
    const bodyGeo = new THREE.TubeGeometry(bodyCurve, 40, 0.42, 20, false);
    const bodyMesh = new THREE.Mesh(bodyGeo, xrayMat);
    const bodyWire = new THREE.Mesh(bodyGeo, wireMat);
    this.uavGroup.add(bodyMesh);
    this.uavGroup.add(bodyWire);

    // Forward EO/FLIR Turret Ball under nose
    const flirTurret = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9, roughness: 0.2 })
    );
    flirTurret.position.set(-2.8, -0.18, 0);
    this.uavGroup.add(flirTurret);

    // Internal Avionics Stack & Glowing Telemetry Modules
    const avionicsGroup = new THREE.Group();
    const rackGeo = new THREE.BoxGeometry(0.9, 0.24, 0.28);
    const rackMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x059669,
      emissiveIntensity: 0.5
    });
    const avionicsRack = new THREE.Mesh(rackGeo, rackMat);
    avionicsRack.position.set(-1.1, 0.26, 0);
    avionicsGroup.add(avionicsRack);

    // Glowing telemetry LED modules
    for (let i = 0; i < 4; i++) {
      const ledGeo = new THREE.BoxGeometry(0.12, 0.12, 0.05);
      const ledMat = new THREE.MeshBasicMaterial({ color: i % 2 === 0 ? 0x06b6d4 : 0xec4899 });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(-1.4 + i * 0.2, 0.32, 0.14);
      avionicsGroup.add(led);
    }

    // Mid-Fuselage Internal Fuel Cell (Yellow/Amber glow)
    const fuelTankGeo = new THREE.CylinderGeometry(0.24, 0.24, 1.1, 16);
    const fuelTankMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 0.35,
      transparent: true,
      opacity: 0.7
    });
    const fuelTank = new THREE.Mesh(fuelTankGeo, fuelTankMat);
    fuelTank.rotation.z = Math.PI / 2;
    fuelTank.position.set(0.3, 0.25, 0);
    avionicsGroup.add(fuelTank);

    // Golden Wiring Harness along fuselage spine
    const wireHarnessGeo = new THREE.CylinderGeometry(0.02, 0.02, 3.4, 8);
    const harnessMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
    const harness1 = new THREE.Mesh(wireHarnessGeo, harnessMat);
    harness1.rotation.z = Math.PI / 2;
    harness1.position.set(-0.2, 0.38, 0.08);
    avionicsGroup.add(harness1);

    this.uavGroup.add(avionicsGroup);

    // High-Aspect Ratio Wings with Dihedral and Winglets
    const wingShape = new THREE.Shape();
    wingShape.moveTo(-0.35, 0);
    wingShape.lineTo(0.35, 0);
    wingShape.lineTo(0.14, 4.6);
    wingShape.lineTo(-0.16, 4.6);
    wingShape.closePath();

    const wingExtrudeSettings = { depth: 0.05, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.02, bevelThickness: 0.02 };
    const wingGeo = new THREE.ExtrudeGeometry(wingShape, wingExtrudeSettings);

    // Left Wing
    const leftWing = new THREE.Mesh(wingGeo, xrayMat);
    leftWing.rotation.x = Math.PI / 2;
    leftWing.rotation.y = -0.04;
    leftWing.position.set(-0.1, 0.38, 0);
    this.uavGroup.add(leftWing);

    // Right Wing
    const rightWing = new THREE.Mesh(wingGeo, xrayMat);
    rightWing.rotation.x = -Math.PI / 2;
    rightWing.rotation.y = 0.04;
    rightWing.position.set(-0.1, 0.38, 0);
    this.uavGroup.add(rightWing);

    // Winglets
    const wingletGeo = new THREE.BoxGeometry(0.18, 0.45, 0.03);
    const leftWinglet = new THREE.Mesh(wingletGeo, xrayMat);
    leftWinglet.position.set(-0.05, 0.58, 4.6);
    leftWinglet.rotation.x = 0.25;
    this.uavGroup.add(leftWinglet);

    const rightWinglet = new THREE.Mesh(wingletGeo, xrayMat);
    rightWinglet.position.set(-0.05, 0.58, -4.6);
    rightWinglet.rotation.x = -0.25;
    this.uavGroup.add(rightWinglet);

    // Twin Tail Booms & Inverted V-Tail
    const boomGeo = new THREE.CylinderGeometry(0.048, 0.04, 2.8, 12);
    const boomMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.85 });

    const leftBoom = new THREE.Mesh(boomGeo, boomMat);
    leftBoom.rotation.z = Math.PI / 2;
    leftBoom.position.set(1.3, 0.26, 1.1);
    this.uavGroup.add(leftBoom);

    const rightBoom = new THREE.Mesh(boomGeo, boomMat);
    rightBoom.rotation.z = Math.PI / 2;
    rightBoom.position.set(1.3, 0.26, -1.1);
    this.uavGroup.add(rightBoom);

    // V-Tail Fins
    const finGeo = new THREE.BoxGeometry(0.42, 0.78, 0.03);
    const leftFin = new THREE.Mesh(finGeo, xrayMat);
    leftFin.rotation.x = 0.52;
    leftFin.position.set(2.65, 0.52, 1.1);
    this.uavGroup.add(leftFin);

    const rightFin = new THREE.Mesh(finGeo, xrayMat);
    rightFin.rotation.x = -0.52;
    rightFin.position.set(2.65, 0.52, -1.1);
    this.uavGroup.add(rightFin);

    // Stabilizer crossbar
    const crossbar = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 2.2, 8), boomMat);
    crossbar.rotation.x = Math.PI / 2;
    crossbar.position.set(2.65, 0.52, 0);
    this.uavGroup.add(crossbar);

    // Pusher Propeller at Tail
    this.propellerGroup = new THREE.Group();
    const hubGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.14, 16);
    const hubMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9 });
    const hub = new THREE.Mesh(hubGeo, hubMat);
    hub.rotation.z = Math.PI / 2;
    this.propellerGroup.add(hub);

    const bladeGeo = new THREE.BoxGeometry(0.04, 0.85, 0.09);
    const bladeMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.95 });
    for (let b = 0; b < 3; b++) {
      const blade = new THREE.Mesh(bladeGeo, bladeMat);
      blade.rotation.x = (b * Math.PI * 2) / 3;
      blade.position.y = Math.cos((b * Math.PI * 2) / 3) * 0.42;
      blade.position.z = Math.sin((b * Math.PI * 2) / 3) * 0.42;
      this.propellerGroup.add(blade);
    }
    this.propellerGroup.position.set(2.45, 0.08, 0);
    this.uavGroup.add(this.propellerGroup);

    // Landing Gear (Tricycle)
    const strutMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.9 });
    const wheelGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.09, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.95 });

    // Nose gear
    const noseStrut = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.55, 8), strutMat);
    noseStrut.position.set(-2.0, -0.22, 0);
    const noseWheel = new THREE.Mesh(wheelGeo, wheelMat);
    noseWheel.rotation.x = Math.PI / 2;
    noseWheel.position.y = -0.28;
    noseStrut.add(noseWheel);
    this.uavGroup.add(noseStrut);

    // Main gear (left and right under booms)
    const mainGearL = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.55, 8), strutMat);
    mainGearL.position.set(0.4, -0.22, 1.1);
    const wheelL = new THREE.Mesh(wheelGeo, wheelMat);
    wheelL.rotation.x = Math.PI / 2;
    wheelL.position.y = -0.28;
    mainGearL.add(wheelL);
    this.uavGroup.add(mainGearL);

    const mainGearR = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.55, 8), strutMat);
    mainGearR.position.set(0.4, -0.22, -1.1);
    const wheelR = new THREE.Mesh(wheelGeo, wheelMat);
    wheelR.rotation.x = Math.PI / 2;
    wheelR.position.y = -0.28;
    mainGearR.add(wheelR);
    this.uavGroup.add(mainGearR);
  }

  // -------------------------------------------------------------------------
  // 2. Detailed 4-Cylinder Aero Piston Engine (Rotax 914 F Boxer)
  // -------------------------------------------------------------------------
  buildEngineAssembly() {
    this.engineGroup = new THREE.Group();
    this.engineGroup.position.set(-0.85, -0.45, 0.95);
    this.engineGroup.scale.set(0.72, 0.72, 0.72);
    this.engineGroup.rotation.y = 0.35;
    this.scene.add(this.engineGroup);

    const metalMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.88, roughness: 0.25 });
    const darkSteelMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.92, roughness: 0.3 });

    // 1. Central Crankcase Block
    const crankcaseGeo = new THREE.BoxGeometry(1.2, 1.0, 1.5);
    this.parts.crankcase = new THREE.Mesh(crankcaseGeo, darkSteelMat);
    this.parts.crankcase.position.set(0, -0.2, 0);
    this.engineGroup.add(this.parts.crankcase);

    // Front accessory gearbox and drive pulley
    const pulleyGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.15, 24);
    const pulleyMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.95 });
    const pulley = new THREE.Mesh(pulleyGeo, pulleyMat);
    pulley.rotation.x = Math.PI / 2;
    pulley.position.set(0, -0.15, 0.82);
    this.parts.crankcase.add(pulley);

    // Alternator belt
    const beltGeo = new THREE.TorusGeometry(0.32, 0.03, 8, 24);
    const belt = new THREE.Mesh(beltGeo, darkSteelMat);
    belt.position.set(0, -0.15, 0.82);
    this.parts.crankcase.add(belt);

    // 2. Left Cylinder Bank (2 Cylinders along -X)
    this.headShaderMat = this.createThermalMaterial(0);
    this.blockShaderMat = this.createThermalMaterial(1);

    const leftBankGroup = new THREE.Group();
    [-0.38, 0.38].forEach((zPos, idx) => {
      // Cylinder barrel with cooling fins
      const barrelGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.65, 24);
      const barrelMesh = new THREE.Mesh(barrelGeo, this.blockShaderMat);
      barrelMesh.rotation.z = Math.PI / 2;
      barrelMesh.position.set(-0.65, 0, zPos);
      leftBankGroup.add(barrelMesh);

      // Cooling fins
      for (let f = 0; f < 5; f++) {
        const finGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.025, 24);
        const finMesh = new THREE.Mesh(finGeo, this.blockShaderMat);
        finMesh.rotation.z = Math.PI / 2;
        finMesh.position.set(-0.45 - f * 0.08, 0, zPos);
        leftBankGroup.add(finMesh);
      }

      // Cylinder Head
      const headGeo = new THREE.BoxGeometry(0.35, 0.68, 0.68);
      const headMesh = new THREE.Mesh(headGeo, this.headShaderMat);
      headMesh.position.set(-1.05, 0, zPos);
      leftBankGroup.add(headMesh);

      // Valve rocker cover
      const rockerGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.62, 16);
      const rocker = new THREE.Mesh(rockerGeo, darkSteelMat);
      rocker.rotation.x = Math.PI / 2;
      rocker.position.set(-1.24, 0, zPos);
      leftBankGroup.add(rocker);
    });
    leftBankGroup.position.set(0, 0, 0);
    this.parts.leftCylinderBank = leftBankGroup;
    this.engineGroup.add(this.parts.leftCylinderBank);

    // 3. Right Cylinder Bank (2 Cylinders along +X)
    const rightBankGroup = new THREE.Group();
    [-0.38, 0.38].forEach((zPos, idx) => {
      // Cylinder barrel with cooling fins
      const barrelGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.65, 24);
      const barrelMesh = new THREE.Mesh(barrelGeo, this.blockShaderMat);
      barrelMesh.rotation.z = -Math.PI / 2;
      barrelMesh.position.set(0.65, 0, zPos);
      rightBankGroup.add(barrelMesh);

      // Cooling fins
      for (let f = 0; f < 5; f++) {
        const finGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.025, 24);
        const finMesh = new THREE.Mesh(finGeo, this.blockShaderMat);
        finMesh.rotation.z = -Math.PI / 2;
        finMesh.position.set(0.45 + f * 0.08, 0, zPos);
        rightBankGroup.add(finMesh);
      }

      // Cylinder Head
      const headGeo = new THREE.BoxGeometry(0.35, 0.68, 0.68);
      const headMesh = new THREE.Mesh(headGeo, this.headShaderMat);
      headMesh.position.set(1.05, 0, zPos);
      rightBankGroup.add(headMesh);

      // Valve rocker cover
      const rockerGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.62, 16);
      const rocker = new THREE.Mesh(rockerGeo, darkSteelMat);
      rocker.rotation.x = Math.PI / 2;
      rocker.position.set(1.24, 0, zPos);
      rightBankGroup.add(rocker);
    });
    rightBankGroup.position.set(0, 0, 0);
    this.parts.rightCylinderBank = rightBankGroup;
    this.engineGroup.add(this.parts.rightCylinderBank);

    // 4. Overhead Intake Manifold Pipes (Top)
    const intakeGroup = new THREE.Group();
    const plenumGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.8, 16);
    const plenum = new THREE.Mesh(plenumGeo, metalMat);
    plenum.rotation.x = Math.PI / 2;
    plenum.position.set(0, 0.62, 0);
    intakeGroup.add(plenum);

    // Intake runners to left and right
    [-0.38, 0.38].forEach(zPos => {
      const runnerL = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.85, 12), metalMat);
      runnerL.rotation.z = 0.55;
      runnerL.position.set(-0.45, 0.45, zPos);
      intakeGroup.add(runnerL);

      const runnerR = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.85, 12), metalMat);
      runnerR.rotation.z = -0.55;
      runnerR.position.set(0.45, 0.45, zPos);
      intakeGroup.add(runnerR);
    });
    this.parts.intakeManifold = intakeGroup;
    this.engineGroup.add(this.parts.intakeManifold);

    // 5. Lower Exhaust Manifold Pipes (EGT Thermal Glow)
    this.exhaustShaderMat = this.createThermalMaterial(2);
    const exhaustGroup = new THREE.Group();
    const exCurveL = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.9, -0.35, -0.38),
      new THREE.Vector3(-0.5, -0.75, -0.1),
      new THREE.Vector3(0, -0.85, 0.2),
      new THREE.Vector3(0.5, -0.9, 0.5),
      new THREE.Vector3(0.9, -0.95, 0.85)
    ]);
    const exGeo = new THREE.TubeGeometry(exCurveL, 32, 0.09, 12, false);
    const exMesh = new THREE.Mesh(exGeo, this.exhaustShaderMat);
    exhaustGroup.add(exMesh);

    // Turbocharger turbine housing
    const turboGeo = new THREE.TorusGeometry(0.22, 0.08, 12, 24);
    const turbo = new THREE.Mesh(turboGeo, this.exhaustShaderMat);
    turbo.position.set(0.5, -0.85, 0.4);
    exhaustGroup.add(turbo);

    this.parts.exhaust = exhaustGroup;
    this.engineGroup.add(this.parts.exhaust);

    // 6. Reciprocating Piston Kinematics inside block
    const pistonLGroup = new THREE.Group();
    const pistonGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.35, 24);
    const pistonMat = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, metalness: 0.9, roughness: 0.2 });
    const pistonLMesh = new THREE.Mesh(pistonGeo, pistonMat);
    pistonLMesh.rotation.z = Math.PI / 2;
    pistonLGroup.add(pistonLMesh);
    this.parts.pistonL = pistonLGroup;
    this.engineGroup.add(this.parts.pistonL);

    // Crankshaft inside crankcase
    const crankGroup = new THREE.Group();
    const crankPin = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.7, 16), metalMat);
    crankPin.rotation.x = Math.PI / 2;
    crankGroup.add(crankPin);
    this.parts.crankshaft = crankGroup;
    this.engineGroup.add(this.parts.crankshaft);

    // Cache initial positions for exploded view
    for (const [key, part] of Object.entries(this.parts)) {
      this.initialPositions[key] = part.position.clone();
    }
  }

  // -------------------------------------------------------------------------
  // 3. 3D ALT / Battery Pack Module
  // -------------------------------------------------------------------------
  buildBatteryPack() {
    this.batteryGroup = new THREE.Group();
    this.batteryGroup.position.set(2.05, -0.38, 0.35);
    this.batteryGroup.scale.set(0.68, 0.68, 0.68);
    this.scene.add(this.batteryGroup);

    // Battery Main Casing (Two-tone dark alloy and military emerald)
    const packGeo = new THREE.BoxGeometry(1.3, 0.9, 1.0);
    const packMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.8,
      roughness: 0.3
    });
    const packMesh = new THREE.Mesh(packGeo, packMat);
    this.batteryGroup.add(packMesh);

    // Heat Sink Cooling Ribs on Side
    for (let r = 0; r < 5; r++) {
      const ribGeo = new THREE.BoxGeometry(0.04, 0.7, 0.9);
      const ribMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9 });
      const rib = new THREE.Mesh(ribGeo, ribMat);
      rib.position.set(-0.66 + r * 0.02, 0, 0);
      this.batteryGroup.add(rib);
    }

    // Top Emerald Header
    const topCapGeo = new THREE.BoxGeometry(1.24, 0.16, 0.94);
    const topCapMat = new THREE.MeshStandardMaterial({
      color: 0x059669,
      emissive: 0x047857,
      emissiveIntensity: 0.35
    });
    const topCap = new THREE.Mesh(topCapGeo, topCapMat);
    topCap.position.y = 0.52;
    this.batteryGroup.add(topCap);

    // Terminal Posts (+ Red, - Black)
    const termPos = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.18, 16), new THREE.MeshStandardMaterial({ color: 0xef4444 }));
    termPos.position.set(-0.35, 0.66, 0.25);
    this.batteryGroup.add(termPos);

    const termNeg = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.18, 16), new THREE.MeshStandardMaterial({ color: 0x0f172a }));
    termNeg.position.set(0.35, 0.66, 0.25);
    this.batteryGroup.add(termNeg);

    // Glowing 100% Charge LED Bar
    const ledBarGeo = new THREE.BoxGeometry(0.65, 0.06, 0.04);
    const ledBarMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const ledBar = new THREE.Mesh(ledBarGeo, ledBarMat);
    ledBar.position.set(0, 0.25, 0.52);
    this.batteryGroup.add(ledBar);
  }

  // -------------------------------------------------------------------------
  // 4. Schematic Animated 3D Fuel & Oil System Loop
  // -------------------------------------------------------------------------
  buildFuelOilLoop() {
    this.loopGroup = new THREE.Group();
    this.loopGroup.position.set(0.75, -0.62, 0.8);
    this.loopGroup.scale.set(0.65, 0.65, 0.65);
    this.scene.add(this.loopGroup);

    // Oil Reservoir Tank (Vertical translucent cylinder)
    const tankGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.85, 24);
    const tankMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.7,
      metalness: 0.85,
      roughness: 0.15
    });
    const tank = new THREE.Mesh(tankGeo, tankMat);
    tank.position.set(0, 0.38, 0);
    this.loopGroup.add(tank);

    // Tank top pressure cap
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.08, 16), new THREE.MeshStandardMaterial({ color: 0xf59e0b }));
    cap.position.set(0, 0.84, 0);
    this.loopGroup.add(cap);

    // Oil Filter Module
    const filterGeo = new THREE.BoxGeometry(0.75, 0.38, 0.42);
    const filterMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9 });
    const filter = new THREE.Mesh(filterGeo, filterMat);
    filter.position.set(0, -0.32, 0);
    this.loopGroup.add(filter);

    // Illuminated Closed Circulation Piping Loop
    const pipePoints = [
      new THREE.Vector3(0, 0.78, 0),
      new THREE.Vector3(0.72, 0.78, 0),
      new THREE.Vector3(0.72, -0.32, 0),
      new THREE.Vector3(0.38, -0.32, 0),
      new THREE.Vector3(-0.38, -0.32, 0),
      new THREE.Vector3(-0.72, -0.32, 0),
      new THREE.Vector3(-0.72, 0.38, 0),
      new THREE.Vector3(-0.3, 0.38, 0)
    ];
    this.loopCurve = new THREE.CatmullRomCurve3(pipePoints, true);
    const pipeGeo = new THREE.TubeGeometry(this.loopCurve, 54, 0.038, 10, true);
    const pipeMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      emissive: 0x0284c7,
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.8
    });
    const pipeMesh = new THREE.Mesh(pipeGeo, pipeMat);
    this.loopGroup.add(pipeMesh);

    // Circulating Fluid Particle Beacons
    this.fluidParticles = [];
    const particleGeo = new THREE.SphereGeometry(0.048, 8, 8);
    const particleMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    for (let p = 0; p < 10; p++) {
      const pMesh = new THREE.Mesh(particleGeo, particleMat);
      this.loopGroup.add(pMesh);
      this.fluidParticles.push({ mesh: pMesh, offset: p / 10 });
    }
  }

  setupLighting() {
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(this.ambientLight);

    // Key light — casts shadows, defines primary form
    this.dirLight1 = new THREE.DirectionalLight(0x9fc9ff, 1.6);
    this.dirLight1.position.set(5, 8, 5);
    this.dirLight1.castShadow = true;
    this.dirLight1.shadow.mapSize.set(2048, 2048);
    this.dirLight1.shadow.camera.near = 0.5;
    this.dirLight1.shadow.camera.far = 30;
    this.dirLight1.shadow.camera.left = -8;
    this.dirLight1.shadow.camera.right = 8;
    this.dirLight1.shadow.camera.top = 8;
    this.dirLight1.shadow.camera.bottom = -8;
    this.dirLight1.shadow.bias = -0.0015;
    this.scene.add(this.dirLight1);

    // Warm fill/back light — separates silhouette, no shadows (perf)
    this.dirLight2 = new THREE.DirectionalLight(0xf59e0b, 0.55);
    this.dirLight2.position.set(-5, -3, -5);
    this.scene.add(this.dirLight2);

    // Cool rim light from below-behind for edge definition
    this.rimLight = new THREE.DirectionalLight(0x38bdf8, 0.5);
    this.rimLight.position.set(-4, 2, -6);
    this.scene.add(this.rimLight);

    this.gridHelper = new THREE.GridHelper(20, 20, 0x1e293b, 0x0f172a);
    this.gridHelper.position.y = -1.2;
    this.scene.add(this.gridHelper);

    // Shadow-catcher ground plane (invisible, receives soft contact shadow only)
    const groundGeo = new THREE.PlaneGeometry(24, 24);
    const groundMat = new THREE.ShadowMaterial({ opacity: 0.28 });
    this.shadowGround = new THREE.Mesh(groundGeo, groundMat);
    this.shadowGround.rotation.x = -Math.PI / 2;
    this.shadowGround.position.y = -1.19;
    this.shadowGround.receiveShadow = true;
    this.scene.add(this.shadowGround);
  }

  // -------------------------------------------------------------------------
  // Procedural environment map (PMREM) — gives metallic parts realistic
  // reflections without loading an external HDRI file.
  // -------------------------------------------------------------------------
  buildEnvironmentMap() {
    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    pmremGenerator.compileEquirectangularShader();

    const envScene = new THREE.Scene();
    const roomGeo = new THREE.BoxGeometry();
    roomGeo.deleteAttribute('uv');
    const roomMat = new THREE.MeshStandardMaterial({ color: 0x0a0e14, side: THREE.BackSide, roughness: 1.0 });
    const room = new THREE.Mesh(roomGeo, roomMat);
    room.scale.setScalar(10);
    envScene.add(room);

    const coolLight = new THREE.PointLight(0x8ecfff, 22, 20);
    coolLight.position.set(3, 5, -3);
    envScene.add(coolLight);

    const warmLight = new THREE.PointLight(0xffae66, 14, 15);
    warmLight.position.set(-4, -2, 4);
    envScene.add(warmLight);

    const topLight = new THREE.PointLight(0xbfe4ff, 10, 15);
    topLight.position.set(0, 6, 0);
    envScene.add(topLight);

    this.scene.environment = pmremGenerator.fromScene(envScene, 0.045).texture;
    pmremGenerator.dispose();
  }

  // -------------------------------------------------------------------------
  // Post-processing — subtle bloom reserved for thermal redlines / critical
  // alarm glow, consistent with "color reserved for anomalies" HUD design.
  // -------------------------------------------------------------------------
  initPostProcessing() {
    if (!THREE.EffectComposer || !THREE.RenderPass || !THREE.UnrealBloomPass) {
      this.composer = null;
      return;
    }
    const width = this.canvas.clientWidth || 900;
    const height = this.canvas.clientHeight || 550;

    this.composer = new THREE.EffectComposer(this.renderer);
    this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));

    this.bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(width, height), 0.55, 0.35, 0.86);
    this.composer.addPass(this.bloomPass);
  }

  // Enable shadow casting/receiving on all solid meshes built above (called
  // once after every buildX() has populated the scene).
  applyShadowCasters() {
    this.scene.traverse((obj) => {
      if (obj.isMesh && obj !== this.shadowGround) {
        obj.castShadow = true;
        obj.receiveShadow = true;
      }
    });
  }

  setTheme(theme) {
    if (theme === 'day' || theme === 'light') {
      this.scene.background.setHex(0xe2e8f0);
      if (this.ambientLight) this.ambientLight.intensity = 1.35;
      if (this.dirLight1) this.dirLight1.intensity = 1.6;
      if (this.gridHelper) this.gridHelper.material.color.setHex(0x94a3b8);
    } else {
      this.scene.background.setHex(0x060911);
      if (this.ambientLight) this.ambientLight.intensity = 0.9;
      if (this.dirLight1) this.dirLight1.intensity = 1.3;
      if (this.gridHelper) this.gridHelper.material.color.setHex(0x1e293b);
    }
  }

  // -------------------------------------------------------------------------
  // 5. Native 3D Sensor Node Markers & Clipping Control
  // -------------------------------------------------------------------------
  setupSensorNodes() {
    this.sensorNodes = {};
    const sensorDefs = [
      { id: 'cht', name: 'CHT Sensor', pos: new THREE.Vector3(-0.55, 0.42, 0.0), defaultColor: 0x3fb950 },
      { id: 'egt', name: 'EGT Probe', pos: new THREE.Vector3(0.90, -0.10, 0.0), defaultColor: 0x3fb950 },
      { id: 'oil_pressure', name: 'Oil Pressure', pos: new THREE.Vector3(0.0, -0.55, 0.25), defaultColor: 0x3fb950 },
      { id: 'vibration_rms', name: 'Vibration RMS', pos: new THREE.Vector3(-0.30, 0.10, 0.55), defaultColor: 0x3fb950 },
      { id: 'oil_temp', name: 'Oil Temp', pos: new THREE.Vector3(0.0, -0.55, -0.25), defaultColor: 0x3fb950 },
    ];

    const group = new THREE.Group();
    group.name = 'sensor_nodes_group';

    sensorDefs.forEach(def => {
      const geom = new THREE.SphereGeometry(0.05, 16, 16);
      const mat = new THREE.MeshStandardMaterial({
        color: def.defaultColor,
        emissive: def.defaultColor,
        emissiveIntensity: 0.8,
        roughness: 0.3,
        metalness: 0.5
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(def.pos);

      // Beacon halo ring
      const ringGeom = new THREE.RingGeometry(0.065, 0.085, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: def.defaultColor,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.rotation.x = Math.PI / 2;
      mesh.add(ring);

      group.add(mesh);
      this.sensorNodes[def.id] = { mesh, ring, mat, ringMat, basePos: def.pos.clone(), state: 0 };
    });

    this.scene.add(group);
  }

  updateSensorNodeStates(telemetry, alarmMap = null) {
    if (!this.sensorNodes || !telemetry) return;

    const evalChannel = (ch, val) => {
      if (alarmMap && alarmMap[ch] !== undefined) return alarmMap[ch];
      if (ch === 'cht') return val >= 210 ? 3 : (val >= 185 ? 1 : 0);
      if (ch === 'egt') return val >= 800 ? 3 : (val >= 740 ? 1 : 0);
      if (ch === 'vibration_rms') return val >= 3.5 ? 3 : (val >= 2.5 ? 1 : 0);
      if (ch === 'oil_pressure') return val <= 180 ? 3 : (val <= 220 ? 1 : 0);
      if (ch === 'oil_temp') return val >= 140 ? 3 : (val >= 120 ? 1 : 0);
      return 0;
    };

    const colorForState = (st) => {
      if (st >= 3) return 0xf85149; // Critical red
      if (st >= 1) return 0xd29922; // Caution amber
      return 0x3fb950; // Nominal green/grey
    };

    for (const [id, node] of Object.entries(this.sensorNodes)) {
      const val = telemetry[id];
      if (val === undefined || Number.isNaN(val)) continue;
      const st = evalChannel(id, val);
      node.state = st;
      const hex = colorForState(st);
      node.mat.color.setHex(hex);
      node.mat.emissive.setHex(hex);
      node.mat.emissiveIntensity = st >= 3 ? 1.5 : (st >= 1 ? 1.0 : 0.6);
      node.ringMat.color.setHex(hex);
      node.ringMat.opacity = st >= 3 ? 0.95 : (st >= 1 ? 0.75 : 0.4);
    }
  }

  setClippingPlane(axis, enabled, constant) {
    if (!this.clipPlanes[axis]) return;
    this.clipEnabled[axis] = enabled;
    this.clipPlanes[axis].constant = constant;
    this.applyClipping();
  }

  applyClipping() {
    const active = [];
    if (this.clipEnabled.x) active.push(this.clipPlanes.x);
    if (this.clipEnabled.y) active.push(this.clipPlanes.y);
    if (this.clipEnabled.z) active.push(this.clipPlanes.z);

    this.scene.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => {
            m.clippingPlanes = active;
            m.clipShadows = true;
          });
        } else {
          obj.material.clippingPlanes = active;
          obj.material.clipShadows = true;
        }
      }
    });
  }

  applyHistoricalFrame(frame) {
    if (!frame || !frame.telemetry) return;
    const tel = frame.telemetry;
    this.rpm = tel.rpm || 4535.4;
    this.cht = tel.cht || 148.6;
    this.egt = tel.egt || 667.7;
    if (this.thermalUniforms) {
      this.thermalUniforms.uCht.value = this.cht;
      this.thermalUniforms.uEgt.value = this.egt;
    }
    this.updateSensorNodeStates(tel);
  }

  // -------------------------------------------------------------------------
  // Docked leader-line callouts — replaces the old always-facing-camera
  // billboard sprites with a proper CAD/Open MCT style annotation: a fixed
  // 3D anchor point, a thin leader line, and a label box docked to the
  // nearest viewport edge so text stays upright and readable at all times.
  // -------------------------------------------------------------------------
  setupPinnedCallouts() {
    this.callouts = {};
    this.showCallouts = true;

    this.leaderCanvas = document.getElementById('leader-line-canvas');
    this.leaderCtx = this.leaderCanvas ? this.leaderCanvas.getContext('2d') : null;
    if (this.leaderCanvas) {
      this.leaderCanvas.width = this.canvas.clientWidth || 900;
      this.leaderCanvas.height = this.canvas.clientHeight || 550;
    }

    const define = (id, title, lines, color, anchor, dockSide) => {
      this.callouts[id] = { title, color, lines, anchor: anchor.clone(), dockSide };
    };

    define('uav', 'AIRFRAME VIBRATION', [
      'VIB: 1.41 g RMS',
      'ALT: 12,500 ft',
      'SUSTAIN: NOMINAL'
    ], '#38bdf8', new THREE.Vector3(0.1, 2.05, -0.6), 'right');

    define('engine', 'ROTAX 914 F TURBO', [
      'RPM 4535  CHT 148.6°',
      'EGT 667.7°  x1.00',
      'NOMINAL'
    ], '#f59e0b', new THREE.Vector3(-0.85, 0.55, 0.95), 'right');

    define('battery', 'ALT / BATTERY PACK', [
      'BUS 28.4V  13.5A',
      'CHARGE 100%'
    ], '#10b981', new THREE.Vector3(2.05, 0.42, 0.35), 'right');

    define('loop', 'FUEL & OIL SYSTEM LOOP', [
      'FLOW 8.3L/h',
      'OIL P 282  T 96°',
      'CIRCULATION OK'
    ], '#06b6d4', new THREE.Vector3(0.75, 0.28, 0.8), 'right');
  }

  updateCalloutData(id, lines, newColor = null) {
    const entry = this.callouts[id];
    if (!entry) return;
    entry.lines = lines;
    if (newColor) entry.color = newColor;
  }

  // Projects each callout's 3D anchor to screen space, docks its label box
  // to the nearest viewport edge (stacked vertically per side), and draws a
  // leader line connecting the two. Runs once per animation frame.
  drawLeaderLines() {
    if (!this.showCallouts || !this.leaderCtx || !this.leaderCanvas) return;
    const ctx = this.leaderCtx;
    const w = this.leaderCanvas.width;
    const h = this.leaderCanvas.height;
    ctx.clearRect(0, 0, w, h);

    // All callouts dock into a single 2x2 grid block tucked into the
    // viewport's right side — this keeps the ENTIRE left/center clear for
    // the model (that's the point: no boxes drift over the airframe/engine)
    // and needs only 2 stacked rows instead of 4, so it still fits short
    // canvases. Still clears the top-right View Manager (topClear) and the
    // bottom-right clipping-controls/heatmap-legend overlays (bottomClear).
    const topClear = 118;
    const bottomClear = 168;
    const margin = 14;
    const gap = 8;
    const availH = Math.max(90, h - topClear - bottomClear);
    const boxH = Math.max(52, Math.min(64, (availH - gap) / 2));
    const boxW = Math.max(140, Math.min(210, w * 0.30));
    const colXOuter = w - boxW - margin;         // rightmost column (hugs the edge)
    const colXInner = colXOuter - boxW - gap;    // column just inside it
    const rowYTop = topClear;
    const rowYBottom = topClear + boxH + gap;
    const gridSlots = [
      { x: colXInner, y: rowYTop },
      { x: colXOuter, y: rowYTop },
      { x: colXInner, y: rowYBottom },
      { x: colXOuter, y: rowYBottom },
    ];

    Object.values(this.callouts).forEach((entry, i) => {
      const projected = entry.anchor.clone().project(this.camera);
      if (projected.z > 1) return; // behind camera
      const anchorX = (projected.x * 0.5 + 0.5) * w;
      const anchorY = (-projected.y * 0.5 + 0.5) * h;

      const slot = gridSlots[i % gridSlots.length];
      const boxX = slot.x;
      const boxY = slot.y;

      const dockEdgeX = boxX + boxW / 2;
      const dockEdgeY = boxY;

      // Leader line: anchor -> elbow -> top-center of its box
      const elbowY = Math.min(anchorY + 20, dockEdgeY - 14);
      ctx.strokeStyle = entry.color;
      ctx.globalAlpha = 0.75;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(anchorX, anchorY);
      ctx.lineTo(anchorX, elbowY);
      ctx.lineTo(dockEdgeX, elbowY);
      ctx.lineTo(dockEdgeX, dockEdgeY);
      ctx.stroke();

      // Anchor dot
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = entry.color;
      ctx.beginPath();
      ctx.arc(anchorX, anchorY, 3, 0, Math.PI * 2);
      ctx.fill();

      // Docked label box
      ctx.fillStyle = 'rgba(8, 14, 28, 0.92)';
      ctx.strokeStyle = entry.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(boxX, boxY, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.save();
      ctx.beginPath();
      ctx.rect(boxX, boxY, boxW, boxH);
      ctx.clip();

      ctx.fillStyle = entry.color;
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.fillText(entry.title, boxX + 8, boxY + 14, boxW - 16);

      ctx.fillStyle = '#e6edf3';
      ctx.font = '8.5px "JetBrains Mono", monospace';
      entry.lines.forEach((line, idx) => {
        ctx.fillText(line, boxX + 8, boxY + 26 + idx * 10, boxW - 16);
      });
      ctx.restore();
    });
  }

  // -------------------------------------------------------------------------
  // 6. View Manager Camera Modes
  // -------------------------------------------------------------------------
  setViewMode(mode) {
    this.viewMode = mode;
    this.isTransitioning = true;
    if (mode === 'overview') {
      this.targetCameraPos.set(0, 3.4, 7.2);
      this.targetLookAt.set(0.1, 0.1, 0);
    } else if (mode === 'propulsion') {
      this.targetCameraPos.set(-0.85, 0.7, 3.2);
      this.targetLookAt.set(-0.85, -0.4, 0.95);
    } else if (mode === 'systems') {
      this.targetCameraPos.set(1.4, 0.7, 3.2);
      this.targetLookAt.set(1.4, -0.4, 0.6);
    }
  }

  zoomIn(fraction = 0.25) {
    this.isTransitioning = false;
    const target = this.controls.target;
    const offset = new THREE.Vector3().subVectors(this.camera.position, target);
    const curDist = offset.length();
    const newDist = Math.max(this.controls.minDistance + 0.2, curDist * (1.0 - fraction));
    offset.setLength(newDist);
    this.camera.position.copy(target).add(offset);
    this.controls.update();
  }

  zoomOut(fraction = 0.25) {
    this.isTransitioning = false;
    const target = this.controls.target;
    const offset = new THREE.Vector3().subVectors(this.camera.position, target);
    const curDist = offset.length();
    const newDist = Math.min(this.controls.maxDistance - 0.5, curDist * (1.0 + fraction));
    offset.setLength(newDist);
    this.camera.position.copy(target).add(offset);
    this.controls.update();
  }

  setExplodedView(input) {
    let factor = (input > 1.0) ? (input / 100.0) : input;
    this.explodedFactor = Math.max(0.0, Math.min(1.0, factor));
    const ef = this.explodedFactor;

    if (this.parts.leftCylinderBank) {
      this.parts.leftCylinderBank.position.x = this.initialPositions.leftCylinderBank.x - ef * 1.5;
    }
    if (this.parts.rightCylinderBank) {
      this.parts.rightCylinderBank.position.x = this.initialPositions.rightCylinderBank.x + ef * 1.5;
    }
    if (this.parts.intakeManifold) {
      this.parts.intakeManifold.position.y = this.initialPositions.intakeManifold.y + ef * 1.4;
    }
    if (this.parts.exhaust) {
      this.parts.exhaust.position.y = this.initialPositions.exhaust.y - ef * 1.2;
    }
    if (this.parts.crankcase) {
      this.parts.crankcase.position.y = this.initialPositions.crankcase.y - ef * 0.8;
    }
  }

  updateTelemetryState(data) {
    if (data.telemetry) {
      const tel = data.telemetry;
      this.lastTelemetry = tel;
      this.lastComponentHealth = data.component_health || this.lastComponentHealth;
      this.rpm = tel.rpm || 4535.4;
      this.cht = isNaN(tel.cht) ? 210.0 : (tel.cht || 148.6);
      this.egt = tel.egt || 667.7;
      const vib = isNaN(tel.vibration_rms) ? 1.41 : tel.vibration_rms;
      const oilP = (tel.oil_pressure && tel.oil_pressure > 0) ? tel.oil_pressure : 110.0;
      const oilTemp = tel.oil_temp || 96.4;
      const fuelFlow = tel.fuel_flow || 8.30;
      const alt = tel.altitude !== undefined ? tel.altitude : 12500;
      const ambTemp = tel.ambient_temp !== undefined ? tel.ambient_temp : 15.0;
      const coolingFactor = tel.cooling_factor !== undefined ? tel.cooling_factor : 1.0;

      if (this.thermalUniforms) {
        this.thermalUniforms.uCht.value = this.cht;
        this.thermalUniforms.uEgt.value = this.egt;
      }

      // Update 3D Physical Sensor Nodes
      this.updateSensorNodeStates(tel);

      // Update 3D Pinned Callouts (if enabled)
      if (this.showCallouts && this.callouts) {
        // 1. Airframe Vibration Callout
        const vibColor = vib >= 3.5 ? '#f43f5e' : (vib >= 2.4 ? '#f59e0b' : '#38bdf8');
        const freqHz = Math.round(this.rpm / 60.0);
        const currentRul = data.adjusted_rul || data.rul_cycles || 485.0;
        let sustainTag = data.sustain_flight_str;
        if (!sustainTag) {
          const totH = currentRul * 0.1;
          const h = Math.floor(totH);
          const m = Math.round((totH - h) * 60);
          sustainTag = `${h}h ${m.toString().padStart(2, '0')}m`;
        }
        const envelopeTag = currentRul < 50 ? `EMERGENCY RTB: ${sustainTag}` : `SUSTAIN: ${sustainTag}`;

        this.updateCalloutData('uav', [
          `VIB: ${vib.toFixed(2)} g RMS`,
          `ALT: ${Math.round(alt).toLocaleString()} ft`,
          envelopeTag
        ], vibColor);

        // 2. Engine Callout
        const engColor = this.cht >= 200.0 ? '#f43f5e' : (this.cht >= 175.0 ? '#f59e0b' : '#f59e0b');
        const engStatus = this.cht >= 200.0 ? 'THERMAL ALERT' : (data.fault_archetype === 'oil_starvation' ? 'OIL STARVED' : 'NOMINAL');
        this.updateCalloutData('engine', [
          `RPM ${Math.round(this.rpm)}  CHT ${this.cht.toFixed(1)}°`,
          `EGT ${this.egt.toFixed(1)}°  x${coolingFactor}`,
          engStatus
        ], engColor);

        // 3. Battery Callout
        const loadA = (12.0 + (alt / 10000.0) * 1.5).toFixed(1);
        this.updateCalloutData('battery', [
          `BUS 28.4V  ${loadA}A`,
          `CHARGE 100%`
        ], '#10b981');

        // 4. Fluid Loop Callout
        const loopColor = oilP < 200.0 ? '#f43f5e' : '#06b6d4';
        const loopStatus = oilP < 200.0 ? 'LOW PRESSURE' : 'CIRCULATION OK';
        this.updateCalloutData('loop', [
          `FLOW ${fuelFlow.toFixed(1)}L/h`,
          `OIL P ${oilP.toFixed(0)}  T ${oilTemp.toFixed(0)}°`,
          loopStatus
        ], loopColor);
      }
    }
  }

  updateEnvironment(altFt, ambC) {
    // Immediate response to slider movement before next WebSocket frame arrives
    const densityRatio = Math.pow(Math.max(0.01, 1 - 2.25577e-5 * altFt), 4.25588);
    const coolingFactor = Math.max(0.4, (1.0 + (15.0 - ambC) * 0.012) * Math.sqrt(densityRatio)).toFixed(2);
    const tempDelta = ambC - 15.0;
    const estCht = Math.max(110.0, 148.6 + tempDelta * 0.75 + (1.0 - coolingFactor) * 35.0);
    const estEgt = Math.max(520.0, 667.7 + tempDelta * 0.45 + (1.0 - coolingFactor) * 20.0);
    const estVib = Math.max(0.9, 1.41 + (altFt / 10000.0) * 0.28 + Math.max(0.0, (estCht - 148.0) * 0.014));
    const estOilTemp = Math.max(65.0, 85.0 + tempDelta * 0.45 + (1.0 - coolingFactor) * 15.0);
    const estOilP = Math.max(120.0, 281.8 - (estOilTemp - 85.0) * 1.3 - (altFt / 10000.0) * 8.0);
    const estFuelFlow = Math.max(4.2, 8.30 * densityRatio);

    // Compute dynamic physical Arrhenius & mechanical RUL
    let estRul = 485.0;
    if (estCht > 155.0) {
      const excess = (estCht - 155.0) / 22.0;
      estRul *= Math.max(0.028, Math.exp(-excess * 0.95));
    }
    if (estOilP < 220.0) {
      const pLoss = Math.max(0.0, 220.0 - estOilP) / 120.0;
      estRul *= Math.max(0.045, 1.0 - pLoss * 0.92);
    }
    if (estVib > 2.2) {
      const vExcess = Math.max(0.0, estVib - 2.2) / 1.5;
      estRul *= Math.max(0.05, 1.0 - vExcess * 0.88);
    }
    estRul = Math.max(12.0, Math.min(520.0, estRul));
    const totH = estRul * 0.1;
    const h = Math.floor(totH);
    const m = Math.round((totH - h) * 60);
    const estSustainStr = `${h}h ${m.toString().padStart(2, '0')}m`;

    this.updateTelemetryState({
      telemetry: {
        rpm: this.rpm,
        cht: estCht,
        egt: estEgt,
        vibration_rms: estVib,
        oil_temp: estOilTemp,
        oil_pressure: estOilP,
        fuel_flow: estFuelFlow,
        altitude: altFt,
        ambient_temp: ambC,
        cooling_factor: coolingFactor
      },
      adjusted_rul: estRul,
      sustain_flight_str: estSustainStr
    });
  }

  toggleHeatmap(forceState) {
    this.heatmapEnabled = (typeof forceState === 'boolean') ? forceState : !this.heatmapEnabled;
    if (this.thermalUniforms) {
      this.thermalUniforms.uHeatmapEnabled.value = this.heatmapEnabled ? 1.0 : 0.0;
    }
    return this.heatmapEnabled;
  }

  resetView() {
    this.setViewMode('overview');
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const elapsedTime = this.clock.getElapsedTime();
    if (this.thermalUniforms) {
      this.thermalUniforms.uTime.value = elapsedTime;
    }

    // Smooth camera transition toward View Manager target ONLY during active transition
    if (this.isTransitioning) {
      this.camera.position.lerp(this.targetCameraPos, 0.06);
      this.controls.target.lerp(this.targetLookAt, 0.06);
      if (this.camera.position.distanceTo(this.targetCameraPos) < 0.04) {
        this.camera.position.copy(this.targetCameraPos);
        this.controls.target.copy(this.targetLookAt);
        this.isTransitioning = false;
      }
    }

    // Rotate UAV pusher propeller
    this.propellerAngle += (this.rpm / 60.0) * (2 * Math.PI) * 0.004;
    if (this.propellerGroup) {
      this.propellerGroup.rotation.x = this.propellerAngle;
    }

    // Engine crankshaft rotation
    const crankSpeed = (this.rpm / 60.0) * (2 * Math.PI) * 0.003;
    this.crankAngle += crankSpeed;

    if (this.parts.crankshaft) {
      this.parts.crankshaft.rotation.z = this.crankAngle;
    }

    // Reciprocating piston kinematics along X
    const r = 0.22;
    const l = 0.75;
    const s = r * Math.cos(this.crankAngle) + Math.sqrt(Math.max(0.01, l * l - r * r * Math.sin(this.crankAngle) * Math.sin(this.crankAngle)));
    const pistonX = -0.65 - (s - 0.5) - (this.explodedFactor * 0.8);

    if (this.parts.pistonL) {
      this.parts.pistonL.position.x = pistonX;
    }

    // Animate fluid circulation particles in fuel/oil loop
    if (this.loopCurve && this.fluidParticles.length > 0) {
      const loopTime = (elapsedTime * 0.3) % 1.0;
      this.fluidParticles.forEach(p => {
        const t = (loopTime + p.offset) % 1.0;
        const pos = this.loopCurve.getPointAt(t);
        p.mesh.position.copy(pos);
      });
    }

    // Animate sensor beacon rings pulse
    if (this.sensorNodes) {
      const pulse = 1.0 + 0.22 * Math.sin(elapsedTime * 4.5);
      for (const node of Object.values(this.sensorNodes)) {
        if (node.ring) {
          const s = node.state >= 3 ? (1.0 + 0.45 * Math.sin(elapsedTime * 8.0)) : pulse;
          node.ring.scale.set(s, s, 1);
        }
      }
    }

    this.controls.update();
    this.drawLeaderLines();

    if (this.composer) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  onResize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    if (this.composer) this.composer.setSize(width, height);
    if (this.bloomPass) this.bloomPass.resolution.set(width, height);
    if (this.leaderCanvas) {
      this.leaderCanvas.width = width;
      this.leaderCanvas.height = height;
    }
  }

  // -------------------------------------------------------------------------
  // 7. Raycasted Component Hotspots (click a component -> floating metric card)
  // -------------------------------------------------------------------------
  setupHotspots() {
    const proxyDefs = [
      { id: 'cylinder_head', label: 'Cylinder Head', size: [0.6, 0.5, 1.2], center: [-0.55, 0.38, 0.0], fields: ['cht', 'egt'], healthKey: 'cylinder_head' },
      { id: 'crankshaft', label: 'Crankshaft', size: [0.25, 0.25, 1.6], center: [0.0, 0.0, 0.0], fields: ['vibration_rms', 'rpm'], healthKey: 'crankshaft' },
      { id: 'oil_sump', label: 'Oil Sump / Lube', size: [0.9, 0.35, 1.1], center: [0.0, -0.55, 0.0], fields: ['oil_pressure', 'oil_temp'], healthKey: 'lubrication_system' },
      { id: 'exhaust_manifold', label: 'Exhaust Manifold', size: [0.25, 0.3, 1.1], center: [0.88, -0.08, 0.0], fields: ['egt', 'fuel_flow'], healthKey: 'exhaust_manifold' },
    ];

    this.hotspotDefs = {};
    proxyDefs.forEach((def) => {
      const geo = new THREE.BoxGeometry(...def.size);
      const mat = new THREE.MeshBasicMaterial({ visible: false });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...def.center);
      mesh.userData.hotspotId = def.id;
      this.scene.add(mesh);
      this.proxyMeshes.push(mesh);
      this.hotspotDefs[def.id] = def;
    });

    this.hotspotCard = document.createElement('div');
    this.hotspotCard.id = 'hotspot-card';
    Object.assign(this.hotspotCard.style, {
      position: 'absolute', display: 'none', background: 'rgba(6, 12, 26, 0.94)',
      border: '1px solid rgba(88, 166, 255, 0.4)', borderRadius: '8px', padding: '10px 14px',
      minWidth: '180px', color: '#c8d1dc', fontFamily: 'Inter, sans-serif', fontSize: '11px',
      lineHeight: '1.7', boxShadow: '0 4px 24px rgba(0,0,0,0.5)', zIndex: '50', pointerEvents: 'none',
    });
    document.body.appendChild(this.hotspotCard);

    this.canvas.addEventListener('click', (evt) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
      this.raycaster.setFromCamera(this.mouse, this.camera);
      const hits = this.raycaster.intersectObjects(this.proxyMeshes);
      if (hits.length > 0) {
        this.showHotspotCard(hits[0].object.userData.hotspotId, evt.clientX, evt.clientY);
      } else {
        this.hotspotCard.style.display = 'none';
      }
    });
  }

  showHotspotCard(id, screenX, screenY) {
    const def = this.hotspotDefs && this.hotspotDefs[id];
    if (!def) return;
    const tel = this.lastTelemetry || {};
    const health = (this.lastComponentHealth || {})[def.healthKey];

    let html = `<div style="font-weight:700;font-size:12px;margin-bottom:5px;color:#58a6ff">${def.label}</div>`;
    def.fields.forEach((f) => {
      const v = tel[f];
      const label = f.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
      const vStr = (v !== undefined && v !== null && !Number.isNaN(v)) ? v.toFixed(1) : '—';
      html += `<div><span style="color:#6b7785">${label}:</span> <b>${vStr}</b></div>`;
    });
    if (health !== undefined) {
      const pct = Math.round(health * 100);
      const col = health > 0.75 ? '#3fb950' : health > 0.45 ? '#d29922' : '#f85149';
      html += `<div style="margin-top:4px"><span style="color:#6b7785">Component health:</span> <b style="color:${col}">${pct}%</b></div>`;
    }
    this.hotspotCard.innerHTML = html;
    this.hotspotCard.style.display = 'block';

    const margin = 14;
    let x = screenX + margin;
    let y = screenY + margin;
    const rect = this.hotspotCard.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = screenX - rect.width - margin;
    if (y + rect.height > window.innerHeight) y = screenY - rect.height - margin;
    this.hotspotCard.style.left = `${x}px`;
    this.hotspotCard.style.top = `${y}px`;
  }
}
