/**
 * SIH26054 3D Aero Piston Engine Digital Twin Viewport (Three.js)
 *
 * Implements:
 *   - Procedural aero piston engine CAD assembly (Cylinder Head, Block, Piston, Conrod, Crankshaft, Exhaust)
 *   - Slider-crank reciprocating kinematic animation synchronized to live RPM
 *   - Real-time thermal heatmap shaders/emissive glow on CHT & EGT
 *   - Interactive exploded assembly slider
 *   - Component RUL color coding
 */

class AeroEngine3D {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.explodedFactor = 0.0;
    this.crankAngle = 0.0;
    this.rpm = 4800.0;
    this.cht = 150.0;
    this.egt = 650.0;
    this.heatmapEnabled = true;

    this.parts = {};
    this.initialPositions = {};

    this.initScene();
    this.buildEngineAssembly();
    this.setupLighting();
    this.animate();

    window.addEventListener('resize', () => this.onResize());
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b14);

    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;

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
    this.renderer.toneMappingExposure = 1.2;

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 0.4, 0);
    this.controls.maxDistance = 14;
    this.controls.minDistance = 1.5;
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x38bdf8, 1.8);
    mainLight.position.set(5, 8, 4);
    mainLight.castShadow = true;
    this.scene.add(mainLight);

    const rimLight = new THREE.DirectionalLight(0x06b6d4, 1.0);
    rimLight.position.set(-5, 4, -4);
    this.scene.add(rimLight);

    // Warm underside glow for crankcase/thermal reflection
    this.thermalPointLight = new THREE.PointLight(0xf59e0b, 1.5, 6);
    this.thermalPointLight.position.set(0, 1.2, 0);
    this.scene.add(this.thermalPointLight);

    // Ground shadow plane with aerospace grid
    const gridHelper = new THREE.GridHelper(10, 20, 0x0284c7, 0x1e293b);
    gridHelper.position.y = -1.6;
    this.scene.add(gridHelper);
  }

  buildEngineAssembly() {
    this.engineGroup = new THREE.Group();
    this.scene.add(this.engineGroup);

    // Common Materials
    const metalMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      metalness: 0.85,
      roughness: 0.25
    });

    const darkSteelMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.9,
      roughness: 0.3
    });

    // 1. Crankcase / Oil Sump (Base)
    const crankcaseGeo = new THREE.BoxGeometry(1.6, 1.1, 1.8);
    this.parts.crankcase = new THREE.Mesh(crankcaseGeo, darkSteelMat);
    this.parts.crankcase.position.set(0, -0.9, 0);
    this.engineGroup.add(this.parts.crankcase);

    // 2. Cylinder Block with Cooling Fins
    const blockGroup = new THREE.Group();
    const cylinderGeo = new THREE.CylinderGeometry(0.7, 0.7, 1.4, 32);
    const cylinderMesh = new THREE.Mesh(cylinderGeo, metalMat);
    blockGroup.add(cylinderMesh);

    // Add 6 radial cooling fin discs
    const finMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 });
    for (let i = 0; i < 6; i++) {
      const finGeo = new THREE.CylinderGeometry(0.95 - (i * 0.03), 0.95 - (i * 0.03), 0.04, 32);
      const finMesh = new THREE.Mesh(finGeo, finMat);
      finMesh.position.y = -0.5 + i * 0.2;
      blockGroup.add(finMesh);
    }
    blockGroup.position.set(0, 0.4, 0);
    this.parts.cylinderBlock = blockGroup;
    this.engineGroup.add(this.parts.cylinderBlock);

    // 3. Cylinder Head (Thermal Monitoring Core)
    const headGeo = new THREE.CylinderGeometry(0.85, 0.75, 0.5, 32);
    this.headMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.6,
      roughness: 0.4,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0.0
    });
    this.parts.cylinderHead = new THREE.Mesh(headGeo, this.headMat);
    this.parts.cylinderHead.position.set(0, 1.35, 0);
    this.engineGroup.add(this.parts.cylinderHead);

    // Spark plug & valves on head
    const plugGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.35, 16);
    const plugMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.1 });
    const plug = new THREE.Mesh(plugGeo, plugMat);
    plug.position.set(0, 0.35, 0);
    this.parts.cylinderHead.add(plug);

    // 4. Piston
    const pistonGeo = new THREE.CylinderGeometry(0.66, 0.66, 0.6, 32);
    const pistonMat = new THREE.MeshStandardMaterial({
      color: 0xcfd8dc,
      metalness: 0.8,
      roughness: 0.2
    });
    this.parts.piston = new THREE.Mesh(pistonGeo, pistonMat);
    this.parts.piston.position.set(0, 0.4, 0);
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

    // 6. Crankshaft & Web Counterweight
    const crankGroup = new THREE.Group();
    const crankPinGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.4, 16);
    const crankPin = new THREE.Mesh(crankPinGeo, metalMat);
    crankPin.rotation.x = Math.PI / 2;
    crankPin.position.set(0, 0.35, 0);
    crankGroup.add(crankPin);

    const webGeo = new THREE.BoxGeometry(0.4, 0.65, 0.14);
    const webMesh = new THREE.Mesh(webGeo, darkSteelMat);
    webMesh.position.set(0, 0.1, -0.15);
    crankGroup.add(webMesh);

    crankGroup.position.set(0, -0.85, 0);
    this.parts.crankshaft = crankGroup;
    this.engineGroup.add(this.parts.crankshaft);

    // 7. Exhaust Manifold (Curved glowing pipe)
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.5, 1.3, 0),
      new THREE.Vector3(1.2, 1.2, 0.3),
      new THREE.Vector3(1.6, 0.6, 0.5),
      new THREE.Vector3(1.7, -0.2, 0.6)
    ]);
    const exhaustGeo = new THREE.TubeGeometry(curve, 32, 0.14, 16, false);
    this.exhaustMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      metalness: 0.9,
      roughness: 0.3,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0.0
    });
    this.parts.exhaust = new THREE.Mesh(exhaustGeo, this.exhaustMat);
    this.engineGroup.add(this.parts.exhaust);

    // Store base initial local positions for exploded-view calculation
    for (const [key, part] of Object.entries(this.parts)) {
      this.initialPositions[key] = part.position.clone();
    }
  }

  setExplodedView(factor) {
    this.explodedFactor = Math.max(0.0, Math.min(1.0, factor));
    const ef = this.explodedFactor;

    // Explode parts along their primary physical disassembly axes
    if (this.parts.cylinderHead) {
      this.parts.cylinderHead.position.y = this.initialPositions.cylinderHead.y + ef * 1.5;
    }
    if (this.parts.cylinderBlock) {
      this.parts.cylinderBlock.position.y = this.initialPositions.cylinderBlock.y + ef * 0.7;
    }
    if (this.parts.exhaust) {
      this.parts.exhaust.position.x = this.initialPositions.exhaust.x + ef * 1.2;
      this.parts.exhaust.position.z = this.initialPositions.exhaust.z + ef * 0.6;
    }
    if (this.parts.crankcase) {
      this.parts.crankcase.position.y = this.initialPositions.crankcase.y - ef * 0.9;
    }
    if (this.parts.crankshaft) {
      this.parts.crankshaft.position.y = this.initialPositions.crankshaft.y - ef * 0.7;
    }
  }

  updateTelemetryState(data) {
    if (data.telemetry) {
      this.rpm = data.telemetry.rpm || 4800.0;
      this.cht = data.telemetry.cht || 150.0;
      this.egt = data.telemetry.egt || 650.0;
    }

    if (this.heatmapEnabled) {
      this.updateThermalHeatmap();
    }
  }

  updateThermalHeatmap() {
    // CHT Thermal mapping on Cylinder Head
    // Nominal: 140-160°C (cool metal) -> Warning: >185°C (amber glow) -> Critical: >210°C (incandescent red)
    const chtFactor = Math.max(0.0, Math.min(1.0, (this.cht - 130.0) / 80.0));
    const chtColor = new THREE.Color();

    if (chtFactor < 0.5) {
      chtColor.lerpColors(new THREE.Color(0x94a3b8), new THREE.Color(0xf59e0b), chtFactor * 2.0);
    } else {
      chtColor.lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), (chtFactor - 0.5) * 2.0);
    }

    this.headMat.color.copy(chtColor);
    if (this.cht > 180.0) {
      this.headMat.emissive.setHex(this.cht > 205.0 ? 0xdc2626 : 0xd97706);
      this.headMat.emissiveIntensity = THREE.MathUtils.lerp(0.2, 0.9, (this.cht - 180.0) / 35.0);
      this.thermalPointLight.intensity = 1.5 + this.headMat.emissiveIntensity * 2.5;
    } else {
      this.headMat.emissiveIntensity = 0.0;
      this.thermalPointLight.intensity = 1.0;
    }

    // EGT Thermal mapping on Exhaust Manifold
    // Nominal: 600-660°C -> Critical: >780°C
    const egtFactor = Math.max(0.0, Math.min(1.0, (this.egt - 580.0) / 220.0));
    const egtColor = new THREE.Color();
    egtColor.lerpColors(new THREE.Color(0x475569), new THREE.Color(0xf97316), egtFactor);
    this.exhaustMat.color.copy(egtColor);

    if (this.egt > 720.0) {
      this.exhaustMat.emissive.setHex(0xea580c);
      this.exhaustMat.emissiveIntensity = THREE.MathUtils.lerp(0.1, 0.8, (this.egt - 720.0) / 100.0);
    } else {
      this.exhaustMat.emissiveIntensity = 0.0;
    }
  }

  setComponentHealthColors(healthData) {
    if (this.heatmapEnabled) return; // Thermal mode takes precedence if active

    const colorFromHealth = (h) => {
      if (h > 0.7) return 0x10b981; // Green
      if (h > 0.4) return 0xf59e0b; // Amber
      return 0xef4444;              // Red
    };

    if (healthData.cylinder_head && this.headMat) {
      this.headMat.color.setHex(colorFromHealth(healthData.cylinder_head));
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    // Kinematic crankshaft & reciprocating piston cycle
    const crankSpeed = (this.rpm / 60.0) * (2 * Math.PI) * 0.003;
    this.crankAngle += crankSpeed;

    if (this.parts.crankshaft) {
      this.parts.crankshaft.rotation.z = this.crankAngle;
    }

    // Reciprocating slider-crank kinematics:
    // r = crank radius (0.35), l = rod length (1.1)
    const r = 0.35;
    const l = 1.1;
    const s = r * Math.cos(this.crankAngle) + Math.sqrt(Math.max(0.01, l * l - r * r * Math.sin(this.crankAngle) * Math.sin(this.crankAngle)));
    const pistonY = s - 0.7 + (this.explodedFactor * 0.4);

    if (this.parts.piston) {
      this.parts.piston.position.y = pistonY;
    }

    if (this.parts.conrod) {
      const rodAngle = Math.asin((-r * Math.sin(this.crankAngle)) / l);
      this.parts.conrod.rotation.z = rodAngle;
      this.parts.conrod.position.x = -r * Math.sin(this.crankAngle) * 0.4;
      this.parts.conrod.position.y = this.initialPositions.conrod.y + (pistonY - 0.4) * 0.5;
    }

    // Slow orbital camera rotation if idle
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}
