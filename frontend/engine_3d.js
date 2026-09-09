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

    this.targetCameraPos = new THREE.Vector3(0, 3.4, 7.2);
    this.targetLookAt = new THREE.Vector3(0.1, 0.1, 0);

    this.initScene();
    this.initThermalShaders();
    this.buildTapasAirframe();
    this.buildEngineAssembly();
    this.buildBatteryPack();
    this.buildFuelOilLoop();
    this.setupLighting();
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
        side: THREE.DoubleSide
      });
    };
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x38bdf8, 2.2);
    mainLight.position.set(6, 10, 5);
    mainLight.castShadow = true;
    this.scene.add(mainLight);

    const rimLight = new THREE.DirectionalLight(0x06b6d4, 1.4);
    rimLight.position.set(-6, 5, -5);
    this.scene.add(rimLight);

    // Warm underside glow for engine and fluid loops
    this.thermalPointLight = new THREE.PointLight(0xf59e0b, 2.0, 8);
    this.thermalPointLight.position.set(-0.8, 0.4, 0.8);
    this.scene.add(this.thermalPointLight);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(14, 28, 0x0284c7, 0x1e293b);
    gridHelper.position.y = -1.6;
    this.scene.add(gridHelper);
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

  // -------------------------------------------------------------------------
  // 5. Pinned 3D Holographic Data Callouts
  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------
  // 5. Pinned 3D Holographic Data Callouts (Dynamic Real-Time Canvases)
  // -------------------------------------------------------------------------
  setupPinnedCallouts() {
    this.callouts = {};

    const createDynamicCallout = (id, title, lines, color = '#38bdf8') => {
      const canvas = document.createElement('canvas');
      canvas.width = 440;
      canvas.height = 140;
      const ctx = canvas.getContext('2d');
      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.scale.set(1.5, 0.48, 1);
      this.scene.add(sprite);

      this.callouts[id] = { canvas, ctx, texture, sprite, title, color, lines };
      this.drawCallout(id);
      return sprite;
    };

    // 1. UAV Callout
    this.calloutUAV = createDynamicCallout('uav', 'AIRFRAME VIBRATION', [
      'VIBRATION: 1.41 G (RMS)',
      'FREQ: 720 Hz | AMB: +15.0°C | ALT: 12,500 ft'
    ], '#38bdf8');
    this.calloutUAV.position.set(0.1, 2.05, -0.6);

    // 2. Engine Callout
    this.calloutEngine = createDynamicCallout('engine', 'ROTAX 914 F TURBO', [
      'RPM: 4535 | CHT: 148.6°C | EGT: 667.7°C',
      'COOLING: 1.00x | CRUISE ENVELOPE'
    ], '#f59e0b');
    this.calloutEngine.position.set(-0.85, 0.55, 0.95);

    // 3. Battery Callout
    this.calloutBattery = createDynamicCallout('battery', 'ALT / BATTERY PACK', [
      'BUS: 28.4 V | LOAD: 13.5 A',
      'CHARGE: 100% | CELL BAL: OPTIMAL'
    ], '#10b981');
    this.calloutBattery.position.set(2.05, 0.42, 0.35);

    // 4. Fluid Loop Callout
    this.calloutLoop = createDynamicCallout('loop', 'FUEL & OIL SYSTEM LOOP', [
      'FLOW: 8.30 L/h | OIL P: 281.8 kPa',
      'OIL TEMP: 96.4°C | NOMINAL CIRCULATION'
    ], '#06b6d4');
    this.calloutLoop.position.set(0.75, 0.28, 0.8);
  }

  drawCallout(id) {
    const entry = this.callouts[id];
    if (!entry) return;
    const { canvas, ctx, texture, title, color, lines } = entry;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Card background
    ctx.fillStyle = 'rgba(8, 14, 28, 0.94)';
    ctx.beginPath();
    ctx.roundRect(6, 6, 428, 128, 10);
    ctx.fill();

    // Border & header line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(6, 6, 428, 128, 10);
    ctx.stroke();

    // Header title
    ctx.fillStyle = color;
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.fillText(title, 20, 38);

    // Value lines
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px "JetBrains Mono", monospace';
    lines.forEach((line, idx) => {
      ctx.fillText(line, 20, 72 + idx * 28);
    });

    texture.needsUpdate = true;
  }

  updateCalloutData(id, lines, newColor = null) {
    const entry = this.callouts[id];
    if (!entry) return;
    entry.lines = lines;
    if (newColor) entry.color = newColor;
    this.drawCallout(id);
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

      // Update 3D Pinned Callouts in Real-Time
      if (this.callouts) {
        // 1. Airframe Vibration Callout
        const vibColor = vib >= 3.5 ? '#f43f5e' : (vib >= 2.4 ? '#f59e0b' : '#38bdf8');
        const freqHz = Math.round(this.rpm / 60.0);
        this.updateCalloutData('uav', [
          `VIBRATION: ${vib.toFixed(2)} G (RMS)`,
          `FREQ: ${freqHz} Hz | AMB: ${ambTemp > 0 ? '+' : ''}${ambTemp.toFixed(1)}°C | ALT: ${Math.round(alt).toLocaleString()} ft`
        ], vibColor);

        // 2. Engine Callout
        const engColor = this.cht >= 200.0 ? '#f43f5e' : (this.cht >= 175.0 ? '#f59e0b' : '#f59e0b');
        const engStatus = this.cht >= 200.0 ? 'THERMAL ALERT' : (data.fault_archetype === 'oil_starvation' ? 'OIL STARVED' : 'CRUISE NOMINAL');
        this.updateCalloutData('engine', [
          `RPM: ${Math.round(this.rpm)} | CHT: ${this.cht.toFixed(1)}°C | EGT: ${this.egt.toFixed(1)}°C`,
          `COOLING: ${coolingFactor}x | STATUS: ${engStatus}`
        ], engColor);

        // 3. Battery Callout
        const loadA = (12.0 + (alt / 10000.0) * 1.5).toFixed(1);
        this.updateCalloutData('battery', [
          `BUS: 28.4 V | LOAD: ${loadA} A`,
          `CHARGE: 100% | CELL BAL: OPTIMAL`
        ], '#10b981');

        // 4. Fluid Loop Callout
        const loopColor = oilP < 200.0 ? '#f43f5e' : '#06b6d4';
        const loopStatus = oilP < 200.0 ? 'LOW PRESSURE ALERT' : 'CIRCULATION OK';
        this.updateCalloutData('loop', [
          `FLOW: ${fuelFlow.toFixed(2)} L/h | OIL P: ${oilP.toFixed(1)} kPa`,
          `OIL TEMP: ${oilTemp.toFixed(1)}°C | ${loopStatus}`
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
      }
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
