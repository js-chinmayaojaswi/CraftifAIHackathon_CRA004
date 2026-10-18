const assert = require('assert');

// Exact geometry code as implemented in index.html
const hazardZone = [
  { x: 500, y: 180 },
  { x: 1120, y: 180 },
  { x: 1120, y: 640 },
  { x: 500, y: 640 }
];

function isPointInsidePolygon(px, py, poly) {
  if (typeof px !== 'number' || typeof py !== 'number' || isNaN(px) || isNaN(py) || !poly || poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > py) !== (yj > py)) &&
                      (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function doSegmentsIntersect(p1, p2, p3, p4) {
  function ccw(a, b, c) {
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
  }
  return (ccw(p1, p3, p4) !== ccw(p2, p3, p4)) && (ccw(p1, p2, p3) !== ccw(p1, p2, p4));
}

function evaluateZoneSafety(obj, poly) {
  poly = poly || hazardZone;
  if (!obj || typeof obj.x !== 'number' || typeof obj.y !== 'number' ||
      isNaN(obj.x) || isNaN(obj.y) || isNaN(obj.w) || isNaN(obj.h) ||
      obj.w <= 0 || obj.h <= 0 || !poly || poly.length < 3) {
    return {
      state: 'UNKNOWN',
      inZone: false,
      distanceMeters: null,
      distancePx: null,
      bbox: null,
      zoneBounds: null
    };
  }

  const x_min = obj.x - obj.w / 2;
  const x_max = obj.x + obj.w / 2;
  const y_min = obj.y - obj.h / 2;
  const y_max = obj.y + obj.h / 2;

  let z_xmin = Infinity, z_xmax = -Infinity, z_ymin = Infinity, z_ymax = -Infinity;
  for (let i = 0; i < poly.length; i++) {
    z_xmin = Math.min(z_xmin, poly[i].x);
    z_xmax = Math.max(z_xmax, poly[i].x);
    z_ymin = Math.min(z_ymin, poly[i].y);
    z_ymax = Math.max(z_ymax, poly[i].y);
  }

  // Exact AABB Overlap Condition
  const overlapsAABB = (x_min < z_xmax && x_max > z_xmin && y_min < z_ymax && y_max > z_ymin);

  const dx = Math.max(0, z_xmin - x_max, x_min - z_xmax);
  const dy = Math.max(0, z_ymin - y_max, y_min - z_ymax);
  const distPx = Math.hypot(dx, dy);
  const distMeters = distPx === 0 ? 0.0 : Math.round((distPx / 45) * 10) / 10;

  let inZone = false;
  if (overlapsAABB) {
    inZone = true;
  }

  return {
    state: inZone ? 'BREACH' : 'SAFE',
    inZone: inZone,
    distancePx: distPx,
    distanceMeters: distMeters,
    bbox: { x_min, y_min, x_max, y_max },
    zoneBounds: { z_xmin, z_ymin, z_xmax, z_ymax }
  };
}

function updateObjectSafetyState(obj, rawEval) {
  if (!obj) return 'UNKNOWN';
  if (!rawEval || rawEval.state === 'UNKNOWN') {
    obj.safetyState = 'UNKNOWN';
    obj.breachFrames = 0;
    obj.clearFrames = 0;
    return 'UNKNOWN';
  }

  const CONFIRMATION_FRAMES = 2;
  if (rawEval.inZone) {
    obj.breachFrames = (obj.breachFrames || 0) + 1;
    obj.clearFrames = 0;
    if (obj.breachFrames >= CONFIRMATION_FRAMES || !obj.safetyState) {
      obj.safetyState = 'BREACH';
    }
  } else {
    obj.clearFrames = (obj.clearFrames || 0) + 1;
    obj.breachFrames = 0;
    if (obj.clearFrames >= CONFIRMATION_FRAMES || !obj.safetyState) {
      obj.safetyState = 'SAFE';
    }
  }
  return obj.safetyState || rawEval.state;
}

console.log("=== RUNNING SENTINEL-EDGE JAVASCRIPT GEOMETRY TEST SUITE ===");

// Case 1: Worker completely outside the zone -> SAFE, no zone-breach alert
{
  const workerA = { id: 101, name: "Worker-A", x: 220, y: 380, w: 56, h: 130 };
  const res = evaluateZoneSafety(workerA, hazardZone);
  assert.strictEqual(res.state, 'SAFE', "Case 1 Failed: Worker-A should be SAFE");
  assert.strictEqual(res.inZone, false, "Case 1 Failed: inZone should be false");
  assert.ok(res.distanceMeters > 0, "Case 1 Failed: distance should be > 0");
  console.log("  [PASS] Case 1: Worker completely outside zone -> SAFE");
}

// Case 2: Cargo completely outside the zone -> SAFE, no zone-breach alert
{
  const cargo = { id: 301, name: "Cargo Pallet", x: 340, y: 440, w: 65, h: 55 };
  const res = evaluateZoneSafety(cargo, hazardZone);
  assert.strictEqual(res.state, 'SAFE', "Case 2 Failed: Cargo should be SAFE");
  assert.strictEqual(res.inZone, false, "Case 2 Failed: inZone should be false");
  console.log("  [PASS] Case 2: Cargo completely outside zone -> SAFE");
}

// Case 3: Worker near the boundary but outside -> SAFE, no zone-breach alert
{
  // Zone starts at x=500. Worker right edge at x=495 (obj.x = 467, w = 56)
  const workerNear = { id: 104, name: "Worker-Near", x: 467, y: 300, w: 56, h: 130 };
  const res = evaluateZoneSafety(workerNear, hazardZone);
  assert.strictEqual(res.state, 'SAFE', "Case 3 Failed: Near boundary outside should be SAFE");
  assert.strictEqual(res.inZone, false);
  assert.ok(res.distancePx > 0, "Case 3 Failed: Distance must be positive");
  console.log("  [PASS] Case 3: Worker near boundary but outside -> SAFE");
}

// Case 4: Worker partially entering the zone -> BREACH
{
  // Worker crosses boundary: x_min = 480, x_max = 536 (obj.x = 508, w = 56)
  const workerOverlap = { id: 105, name: "Worker-Overlap", x: 508, y: 300, w: 56, h: 130 };
  const res = evaluateZoneSafety(workerOverlap, hazardZone);
  assert.strictEqual(res.state, 'BREACH', "Case 4 Failed: Partially entering should be BREACH");
  assert.strictEqual(res.inZone, true);
  assert.strictEqual(res.distanceMeters, 0.0);
  console.log("  [PASS] Case 4: Worker partially entering zone -> BREACH");
}

// Case 5: Worker fully inside the zone -> BREACH
{
  const workerInside = { id: 101, name: "Worker-A", x: 650, y: 350, w: 56, h: 130 };
  const res = evaluateZoneSafety(workerInside, hazardZone);
  assert.strictEqual(res.state, 'BREACH');
  assert.strictEqual(res.inZone, true);
  console.log("  [PASS] Case 5: Worker fully inside zone -> BREACH");
}

// Case 6: Cargo entering or overlapping the zone -> BREACH
{
  const cargoInside = { id: 301, name: "Cargo Pallet", x: 750, y: 480, w: 65, h: 55 };
  const res = evaluateZoneSafety(cargoInside, hazardZone);
  assert.strictEqual(res.state, 'BREACH');
  assert.strictEqual(res.inZone, true);
  console.log("  [PASS] Case 6: Cargo entering or overlapping zone -> BREACH");
}

// Case 7: Object leaving the zone -> Clear active breach and restore SAFE
{
  const obj = { id: 101, name: "Worker-A", breachFrames: 0, clearFrames: 0, safetyState: 'SAFE' };
  // 2 frames in breach
  updateObjectSafetyState(obj, { state: 'BREACH', inZone: true });
  updateObjectSafetyState(obj, { state: 'BREACH', inZone: true });
  assert.strictEqual(obj.safetyState, 'BREACH');

  // 2 frames cleared
  updateObjectSafetyState(obj, { state: 'SAFE', inZone: false });
  updateObjectSafetyState(obj, { state: 'SAFE', inZone: false });
  assert.strictEqual(obj.safetyState, 'SAFE');
  console.log("  [PASS] Case 7: Object leaving zone restores SAFE");
}

// Case 8: Object moving across boundary -> Updates in real time
{
  const workerMoving = { id: 101, name: "Worker-A", w: 56, h: 130 };
  // Step 1: Outside
  workerMoving.x = 450; workerMoving.y = 300;
  assert.strictEqual(evaluateZoneSafety(workerMoving, hazardZone).state, 'SAFE');

  // Step 2: Overlapping
  workerMoving.x = 520; workerMoving.y = 300;
  assert.strictEqual(evaluateZoneSafety(workerMoving, hazardZone).state, 'BREACH');

  // Step 3: Outside again
  workerMoving.x = 420; workerMoving.y = 300;
  assert.strictEqual(evaluateZoneSafety(workerMoving, hazardZone).state, 'SAFE');
  console.log("  [PASS] Case 8: Object moving across boundary updates in real time");
}

// Case 9: Object with invalid or missing coordinates -> UNKNOWN
{
  const res1 = evaluateZoneSafety(null, hazardZone);
  assert.strictEqual(res1.state, 'UNKNOWN');
  assert.strictEqual(res1.inZone, false);

  const res2 = evaluateZoneSafety({ x: NaN, y: 100, w: 50, h: 50 }, hazardZone);
  assert.strictEqual(res2.state, 'UNKNOWN');

  const res3 = evaluateZoneSafety({ x: 200, y: 100, w: -10, h: 50 }, hazardZone);
  assert.strictEqual(res3.state, 'UNKNOWN');
  console.log("  [PASS] Case 9: Invalid coordinates classify as UNKNOWN without false breach");
}

// Case 10: Multiple objects in different positions evaluated independently
{
  const scene = [
    { id: 101, name: "Worker-A", x: 220, y: 380, w: 56, h: 130 }, // SAFE
    { id: 102, name: "Worker-B", x: 340, y: 260, w: 56, h: 130 }, // SAFE
    { id: 103, name: "Worker-C", x: 700, y: 350, w: 56, h: 130 }, // BREACH
    { id: 201, name: "Forklift-01", x: 180, y: 560, w: 90, h: 75 }, // SAFE
    { id: 301, name: "Cargo Pallet", x: 750, y: 480, w: 65, h: 55 } // BREACH
  ];

  const results = scene.map(o => evaluateZoneSafety(o, hazardZone).state);
  assert.deepStrictEqual(results, ['SAFE', 'SAFE', 'BREACH', 'SAFE', 'BREACH']);
  console.log("  [PASS] Case 10: Multiple objects evaluated completely independently");
}

console.log("\n>>> ALL 10 JAVASCRIPT GEOMETRY TEST CASES PASSED SUCCESSFULLY! <<<\n");
