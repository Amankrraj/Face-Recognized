/**
 * FaceID · script.js
 * Real-time face detection & recognition using face-api.js
 * ─────────────────────────────────────────────────────────────
 * Flow:
 *  1. Load AI models (TinyFaceDetector + FaceLandmark68Net + FaceRecognitionNet)
 *  2. Start webcam stream
 *  3. Register: capture face descriptor → store {name, descriptor}
 *  4. Recognize: match live face against stored descriptors via Euclidean distance
 */

'use strict';

/* ── Constants ─────────────────────────────────────────────── */

// CDN base for face-api model weights
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

// Euclidean distance threshold: lower = stricter matching
const MATCH_THRESHOLD = 0.52;

// How often (ms) to run recognition while active
const RECOGNITION_INTERVAL = 900;

// How long (ms) to show the result badge
const BADGE_DURATION = 2500;

/* ── DOM references ────────────────────────────────────────── */
const video         = document.getElementById('video');
const overlay       = document.getElementById('overlay');
const btnRegister   = document.getElementById('btnRegister');
const btnRecognize  = document.getElementById('btnRecognize');
const btnStop       = document.getElementById('btnStop');
const personNameIn  = document.getElementById('personName');
const registeredList= document.getElementById('registeredList');
const resultBadge   = document.getElementById('resultBadge');
const resultName    = document.getElementById('resultName');
const resultIcon    = document.getElementById('resultIcon');
const faceCountEl   = document.getElementById('faceCount');
const logBox        = document.getElementById('logBox');
const statusPill    = document.getElementById('statusPill');
const statusText    = document.getElementById('statusText');
const loadingModal  = document.getElementById('loadingModal');
const loadingMsg    = document.getElementById('loadingMsg');

/* ── State ─────────────────────────────────────────────────── */

// Array of { name: string, descriptor: Float32Array }
let registeredFaces = [];

// Interval handle for recognition loop
let recognitionTimer = null;

// Canvas 2D context for drawing face boxes
let ctx = null;

/* ── Utility: set status pill ───────────────────────────────── */
function setStatus(text, type = '') {
  statusText.textContent = text;
  statusPill.className = 'status-pill ' + type;
}

/* ── Utility: add log entry ─────────────────────────────────── */
function addLog(name, isUnknown = false, isInfo = false) {
  // Remove placeholder if present
  const placeholder = logBox.querySelector('.log-placeholder');
  if (placeholder) placeholder.remove();

  const now = new Date();
  const time = now.toTimeString().slice(0, 8);

  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const cls = isInfo ? 'log-info' : (isUnknown ? 'log-unknown' : 'log-name');
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="${cls}">${name}</span>
  `;

  logBox.appendChild(entry);
  logBox.scrollTop = logBox.scrollHeight; // auto-scroll
}

/* ── Utility: show result badge ─────────────────────────────── */
let badgeTimeout = null;
function showBadge(name, unknown = false) {
  resultName.textContent = name;
  resultIcon.textContent = unknown ? '?' : '✓';
  resultBadge.classList.toggle('unknown', unknown);
  resultBadge.classList.add('visible');

  // Hide badge after BADGE_DURATION ms
  clearTimeout(badgeTimeout);
  badgeTimeout = setTimeout(() => {
    resultBadge.classList.remove('visible');
  }, BADGE_DURATION);
}

/* ── Step 1: Load AI models ─────────────────────────────────── */
async function loadModels() {
  loadingMsg.textContent = 'Loading AI models…';

  // face-api.js needs three model files:
  // • TinyFaceDetector  – fast, lightweight face detection
  // • FaceLandmark68Net – 68-point landmark mesh (required for alignment)
  // • FaceRecognitionNet– 128-D face embedding / descriptor
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  loadingMsg.textContent = 'Loading landmark model…';
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  loadingMsg.textContent = 'Loading recognition model…';
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
}

/* ── Step 2: Start webcam ───────────────────────────────────── */
async function startCamera() {
  loadingMsg.textContent = 'Requesting camera…';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
    audio: false
  });
  video.srcObject = stream;

  // Wait until video metadata is ready before sizing canvas
  return new Promise(resolve => {
    video.addEventListener('loadedmetadata', () => {
      overlay.width  = video.videoWidth;
      overlay.height = video.videoHeight;
      ctx = overlay.getContext('2d');
      resolve();
    });
  });
}

/* ── Step 3: Register a face ────────────────────────────────── */
btnRegister.addEventListener('click', async () => {
  const name = personNameIn.value.trim();
  if (!name) {
    personNameIn.focus();
    personNameIn.style.borderColor = 'var(--accent2)';
    setTimeout(() => (personNameIn.style.borderColor = ''), 1200);
    return;
  }

  setStatus('Capturing face…', 'active');
  btnRegister.disabled = true;

  try {
    // Detect a single face with full pipeline (detect → landmarks → descriptor)
    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      addLog('No face detected – try again', false, true);
      setStatus('No face found', 'error');
      setTimeout(() => setStatus('Ready', 'ready'), 2000);
      return;
    }

    // Store 128-D float32 descriptor alongside the name
    registeredFaces.push({ name, descriptor: detection.descriptor });

    // Draw confirmation box on overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const box = detection.detection.box;
    ctx.strokeStyle = '#00e5b0';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = '#00e5b0';
    ctx.font = '14px Share Tech Mono, monospace';
    ctx.fillText(name, box.x, box.y - 8);
    setTimeout(() => ctx.clearRect(0, 0, overlay.width, overlay.height), 1500);

    // Update UI chips & counter
    renderChips();
    faceCountEl.textContent = registeredFaces.length;
    personNameIn.value = '';

    addLog(`Registered: ${name}`, false, true);
    setStatus('Registered ✓', 'ready');
    setTimeout(() => setStatus('Ready', 'ready'), 2000);

  } catch (err) {
    console.error('Register error:', err);
    addLog('Error during capture', false, true);
    setStatus('Error', 'error');
  } finally {
    btnRegister.disabled = false;
  }
});

/* ── Render face name chips ─────────────────────────────────── */
function renderChips() {
  registeredList.innerHTML = '';
  registeredFaces.forEach((face, i) => {
    const chip = document.createElement('span');
    chip.className = 'face-chip';
    chip.innerHTML = `
      <span class="chip-dot"></span>
      ${face.name}
      <span class="chip-del" title="Remove" data-idx="${i}">✕</span>
    `;
    registeredList.appendChild(chip);
  });

  // Delegate removal clicks
  registeredList.querySelectorAll('.chip-del').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.idx);
      const removed = registeredFaces.splice(idx, 1)[0];
      faceCountEl.textContent = registeredFaces.length;
      addLog(`Removed: ${removed.name}`, false, true);
      renderChips();
    });
  });
}

/* ── Step 4: Recognition loop ───────────────────────────────── */
btnRecognize.addEventListener('click', startRecognition);
btnStop.addEventListener('click', stopRecognition);

function startRecognition() {
  if (registeredFaces.length === 0) {
    addLog('Register at least one face first!', false, true);
    return;
  }

  btnRecognize.classList.add('hidden');
  btnStop.classList.remove('hidden');
  setStatus('Recognizing…', 'active');
  addLog('Recognition started', false, true);

  // Build a FaceMatcher from stored descriptors for batch matching
  // LabeledFaceDescriptors groups descriptors by label (name)
  recognitionTimer = setInterval(runRecognition, RECOGNITION_INTERVAL);
}

function stopRecognition() {
  clearInterval(recognitionTimer);
  recognitionTimer = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  resultBadge.classList.remove('visible');
  btnStop.classList.add('hidden');
  btnRecognize.classList.remove('hidden');
  setStatus('Ready', 'ready');
  addLog('Recognition stopped', false, true);
}

async function runRecognition() {
  // Detect ALL faces in the current video frame
  const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();

  // Clear previous drawings
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!detections || detections.length === 0) return;

  // Build a LabeledFaceDescriptors per registered person
  // (supports multiple samples per person if same name registered >1 times)
  const labelMap = {};
  registeredFaces.forEach(({ name, descriptor }) => {
    if (!labelMap[name]) labelMap[name] = [];
    labelMap[name].push(descriptor);
  });

  const labeledDescriptors = Object.entries(labelMap).map(
    ([label, descs]) => new faceapi.LabeledFaceDescriptors(label, descs)
  );

  // FaceMatcher calculates Euclidean distance between descriptors
  const matcher = new faceapi.FaceMatcher(labeledDescriptors, MATCH_THRESHOLD);

  // Resize detections to match display size
  const dims = { width: overlay.width, height: overlay.height };
  const resized = faceapi.resizeResults(detections, dims);

  resized.forEach(det => {
    const bestMatch = matcher.findBestMatch(det.descriptor);
    const isUnknown = bestMatch.label === 'unknown';
    const label     = isUnknown ? 'Unknown Person' : bestMatch.label;

    // Draw bounding box
    const box = det.detection.box;
    ctx.strokeStyle = isUnknown ? '#ff4e6a' : '#00e5b0';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw label above box
    const textY = box.y > 20 ? box.y - 8 : box.y + box.height + 16;
    ctx.fillStyle = isUnknown ? '#ff4e6a' : '#00e5b0';
    ctx.font = '13px Share Tech Mono, monospace';
    ctx.fillText(
      isUnknown ? label : `${label} (${(1 - bestMatch.distance).toFixed(2)})`,
      box.x,
      textY
    );

    // Show badge & log (throttle: only log when a different name appears)
    showBadge(label, isUnknown);
    addLog(label, isUnknown);
  });
}

/* ── Bootstrap ─────────────────────────────────────────────── */
(async () => {
  try {
    await loadModels();
    await startCamera();

    // Hide loading modal
    loadingModal.classList.add('hidden');
    setStatus('Ready', 'ready');
    addLog('System ready – register a face to begin', false, true);

    // Enable buttons
    btnRegister.disabled  = false;
    btnRecognize.disabled = false;

  } catch (err) {
    console.error('Initialization failed:', err);
    loadingMsg.textContent = '⚠ ' + (err.message || 'Initialization failed');
    setStatus('Error', 'error');
  }
})();
