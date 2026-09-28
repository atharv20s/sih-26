// PRAHARI Mission Map — 3D Flight Profile panel. A small, separate Three.js
// scene (deliberately not bolted onto engine_3d.js, which is the engine
// twin): the trajectory as an altitude-as-height ribbon over a flat India
// bounding-box plane, animated in sync with the 2D map via
// window.missionMapClock (set by mission-map.js's tick loop).

(function () {
  const { TRAJECTORY, DRDO_STATIONS, COMM_STATIONS } = window.MISSION_MAP_DATA;

  const container = document.getElementById('flight-profile-3d');
  if (!container || typeof THREE === 'undefined') return;

  // Same linear projection as mission-map.js/build_india_map.py, but
  // centered at origin and scaled down to a manageable 3D scene size.
  const BOUNDS = { lng_min: 68.0, lng_max: 97.5, lat_min: 6.5, lat_max: 36.0 };
  const SCALE = 8; // world units across the full bounding box
  const ALT_SCALE = 0.00035; // ft -> world units (keeps altitude readable, not absurd)

  function project3(lng, lat, alt_ft) {
    const nx = (lng - BOUNDS.lng_min) / (BOUNDS.lng_max - BOUNDS.lng_min) - 0.5;
    const nz = (lat - BOUNDS.lat_min) / (BOUNDS.lat_max - BOUNDS.lat_min) - 0.5;
    return new THREE.Vector3(nx * SCALE, (alt_ft || 0) * ALT_SCALE, -nz * SCALE);
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060911);

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 280;
  const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  camera.position.set(4.2, 3.6, 5.2);
  camera.lookAt(0, 0.6, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dir = new THREE.DirectionalLight(0x9fc9ff, 1.0);
  dir.position.set(4, 6, 3);
  scene.add(dir);

  // Ground plane (India's bounding box, subtle grid) + a lat/lng grid feel
  const groundGeo = new THREE.PlaneGeometry(SCALE, SCALE, 12, 12);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x0d1420, wireframe: true, transparent: true, opacity: 0.35 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const groundFill = new THREE.Mesh(
    new THREE.PlaneGeometry(SCALE, SCALE),
    new THREE.MeshBasicMaterial({ color: 0x0a0e14, transparent: true, opacity: 0.6 })
  );
  groundFill.rotation.x = -Math.PI / 2;
  groundFill.position.y = -0.01;
  scene.add(groundFill);

  // Trajectory ribbon
  const trajPoints = TRAJECTORY.map(wp => project3(wp.lng, wp.lat, wp.alt_ft));
  const curve = new THREE.CatmullRomCurve3(trajPoints, false);
  const tubeGeo = new THREE.TubeGeometry(curve, 200, 0.025, 8, false);
  const tubeMat = new THREE.MeshStandardMaterial({ color: 0x58a6ff, emissive: 0x1a4d7a, emissiveIntensity: 0.6 });
  scene.add(new THREE.Mesh(tubeGeo, tubeMat));

  // Drop-lines from each waypoint down to the ground (altitude readability)
  TRAJECTORY.forEach((wp) => {
    const top = project3(wp.lng, wp.lat, wp.alt_ft);
    const bottom = project3(wp.lng, wp.lat, 0);
    const lineGeo = new THREE.BufferGeometry().setFromPoints([top, bottom]);
    const lineMat = new THREE.LineBasicMaterial({ color: wp.isThreatZone ? 0xf85149 : 0x30363d, transparent: true, opacity: 0.5 });
    scene.add(new THREE.Line(lineGeo, lineMat));

    const dotColor = wp.isThreatZone ? 0xf85149 : 0x58a6ff;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(wp.isThreatZone ? 0.05 : 0.03, 12, 12),
      new THREE.MeshStandardMaterial({ color: dotColor, emissive: dotColor, emissiveIntensity: 0.6 }));
    dot.position.copy(top);
    scene.add(dot);
  });

  // Ground markers for DRDO/comm stations
  function addGroundMarkers(list, color, size) {
    list.forEach((s) => {
      const pos = project3(s.lng, s.lat, 0);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 }));
      dot.position.copy(pos);
      dot.position.y = 0.01;
      scene.add(dot);
    });
  }
  addGroundMarkers(DRDO_STATIONS, 0xf59e0b, 0.035);
  addGroundMarkers(COMM_STATIONS, 0x10b981, 0.03);

  // Animated aircraft marker, driven by the shared clock from mission-map.js
  const aircraft = new THREE.Mesh(
    new THREE.ConeGeometry(0.05, 0.14, 8),
    new THREE.MeshStandardMaterial({ color: 0x3fb950, emissive: 0x1a5c2e, emissiveIntensity: 0.8 })
  );
  scene.add(aircraft);

  // -------------------------------------------------------------------------
  // Regional Globe scene — India highlighted among neighboring countries for
  // geographic context ONLY. No cross-border lines, strikes, or activity of
  // any kind are drawn toward neighboring markers — they're plain reference
  // labels, same treatment as a city label on any atlas.
  // -------------------------------------------------------------------------
  const globeScene = new THREE.Scene();
  globeScene.background = new THREE.Color(0x060911);
  const GLOBE_R = 2.2;

  function latLngToSphere(lat, lng, r) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lng + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    );
  }

  globeScene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const globeDir = new THREE.DirectionalLight(0x9fc9ff, 1.0);
  globeDir.position.set(4, 4, 4);
  globeScene.add(globeDir);

  const globeSphere = new THREE.Mesh(
    new THREE.SphereGeometry(GLOBE_R, 48, 48),
    new THREE.MeshStandardMaterial({ color: 0x0d1626, roughness: 0.85 })
  );
  globeScene.add(globeSphere);
  const globeWire = new THREE.Mesh(
    new THREE.SphereGeometry(GLOBE_R * 1.002, 24, 24),
    new THREE.MeshBasicMaterial({ color: 0x1e2733, wireframe: true, transparent: true, opacity: 0.5 })
  );
  globeScene.add(globeWire);

  // India region highlight — a soft glowing patch at India's centroid
  const indiaCentroid = { lat: 22.0, lng: 79.0 };
  const indiaPos = latLngToSphere(indiaCentroid.lat, indiaCentroid.lng, GLOBE_R);
  const indiaGlow = new THREE.Mesh(
    new THREE.SphereGeometry(0.55, 20, 20),
    new THREE.MeshBasicMaterial({ color: 0x58c8ff, transparent: true, opacity: 0.5 })
  );
  indiaGlow.position.copy(indiaPos);
  globeScene.add(indiaGlow);

  // India outline draped on the globe surface (reuses TRAJECTORY + station
  // points as a light scatter, plus the mission trajectory arc)
  const trajGlobePts = TRAJECTORY.map(wp => latLngToSphere(wp.lat, wp.lng, GLOBE_R * 1.01));
  const globeCurve = new THREE.CatmullRomCurve3(trajGlobePts, false);
  const globeTube = new THREE.TubeGeometry(globeCurve, 100, 0.025, 8, false);
  globeScene.add(new THREE.Mesh(globeTube, new THREE.MeshBasicMaterial({ color: 0x7dd3ff })));

  addGlobeMarkers(globeScene, DRDO_STATIONS.map(s => ({ lat: s.lat, lng: s.lng })), 0xf59e0b, 0.045, GLOBE_R);

  // Neighboring country reference labels — geographic context only, plain
  // dots with no connecting lines, no styling that implies activity/threat.
  const NEIGHBOR_REFERENCES = [
    { name: 'Islamabad', lat: 33.6844, lng: 73.0479 },
    { name: 'Kathmandu', lat: 27.7172, lng: 85.3240 },
    { name: 'Dhaka', lat: 23.8103, lng: 90.4125 },
    { name: 'Colombo', lat: 6.9271, lng: 79.8612 },
    { name: 'Naypyidaw', lat: 19.7633, lng: 96.0785 },
    { name: 'Lhasa', lat: 29.6500, lng: 91.1000 },
  ];
  addGlobeMarkers(globeScene, NEIGHBOR_REFERENCES, 0x8b98a5, 0.03, GLOBE_R);

  function addGlobeMarkers(sceneRef, list, color, size, r) {
    list.forEach((p) => {
      const pos = latLngToSphere(p.lat, p.lng, r * 1.005);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(size, 8, 8), new THREE.MeshBasicMaterial({ color }));
      dot.position.copy(pos);
      sceneRef.add(dot);
    });
  }

  const globeAircraft = new THREE.Mesh(
    new THREE.ConeGeometry(0.035, 0.09, 8),
    new THREE.MeshStandardMaterial({ color: 0x3fb950, emissive: 0x1a5c2e, emissiveIntensity: 0.8 })
  );
  globeScene.add(globeAircraft);

  const globeCamera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);

  // -------------------------------------------------------------------------
  // Shared render loop — swaps between the flight-profile scene and the
  // regional-globe scene based on `viewMode`, driven by the toggle buttons.
  // -------------------------------------------------------------------------
  let viewMode = 'profile';
  let angle = 0;
  // Start the globe already facing India instead of the Greenwich meridian
  // (which would leave a viewer staring at empty ocean for ~15s on load).
  let globeAngle = Math.atan2(indiaPos.x, indiaPos.z);

  function animate() {
    requestAnimationFrame(animate);

    const clock = window.missionMapClock;

    if (viewMode === 'profile') {
      if (clock && typeof clock.t === 'number') {
        aircraft.position.copy(project3(clock.lng, clock.lat, clock.alt_ft));
      }
      angle += 0.0025;
      camera.position.set(Math.sin(angle) * 6.2, 3.6, Math.cos(angle) * 6.2);
      camera.lookAt(0, 0.6, 0);
      renderer.render(scene, camera);
    } else {
      if (clock && typeof clock.t === 'number') {
        globeAircraft.position.copy(latLngToSphere(clock.lat, clock.lng, GLOBE_R * 1.02));
      }
      globeAngle += 0.0018;
      globeCamera.position.set(Math.sin(globeAngle) * 5.5, 2.0, Math.cos(globeAngle) * 5.5);
      globeCamera.lookAt(0, 0, 0);
      renderer.render(globeScene, globeCamera);
    }
  }
  animate();

  function resize() {
    const w = container.clientWidth || 600;
    const h = container.clientHeight || 280;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    globeCamera.aspect = w / h;
    globeCamera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);

  const btnProfile = document.getElementById('btn-view-profile');
  const btnGlobe = document.getElementById('btn-view-globe');
  function setViewMode(mode) {
    viewMode = mode;
    if (btnProfile) btnProfile.classList.toggle('btn-primary', mode === 'profile');
    if (btnGlobe) btnGlobe.classList.toggle('btn-primary', mode === 'globe');
  }
  if (btnProfile) btnProfile.addEventListener('click', () => setViewMode('profile'));
  if (btnGlobe) btnGlobe.addEventListener('click', () => setViewMode('globe'));
  setViewMode('profile');
})();
