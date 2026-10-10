/**
 * ============================================================================
 * SIGN LANGUAGE DETECTION STUDIO - CORE APPLICATION LOGIC
 * ============================================================================
 */

(function () {
  'use strict';

  // --- HAND SKELETON CONNECTIONS (MediaPipe 21 landmarks) ---
  const HAND_CONNECTIONS = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle finger
    [9, 10], [10, 11], [11, 12],
    // Ring finger
    [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm base
    [5, 9], [9, 13], [13, 17]
  ];

  const FINGERTIPS = [4, 8, 12, 16, 20];

  // --- STATE ---
  const state = {
    cameraRunning: false,
    cameraStream: null,
    isProcessing: false,
    isMirrored: true,
    streamMode: 'browser', // 'browser' or 'server'
    ttsEnabled: true,
    lastSpokenGesture: null,
    stableGestureCount: 0,
    currentGesture: 'No hand detected',
    currentConfidence: 0,
    fps: 0,
    latency: 0,
    lastFrameTime: performance.now(),
    frameCount: 0,
    sentenceWords: [],
    availableClasses: ['GoodJob', 'Hi', 'Nice', 'peace'],
    metadata: {
      Hi: { icon: '👋', title: 'Hello / Wave', desc: 'Open hand with spread fingers' },
      GoodJob: { icon: '👍', title: 'Thumbs Up', desc: 'Fist with thumb pointing up' },
      Nice: { icon: '👌', title: 'OK / Pinch', desc: 'Thumb & index circle, 3 fingers up' },
      peace: { icon: '✌️', title: 'Peace / Victory', desc: 'Index & middle in V-shape' }
    }
  };

  // --- DOM ELEMENTS ---
  const el = {
    video: document.getElementById('webcamVideo'),
    canvas: document.getElementById('landmarkCanvas'),
    mjpeg: document.getElementById('serverMjpeg'),
    placeholder: document.getElementById('viewportPlaceholder'),
    btnStart: document.getElementById('btnStartCam'),
    btnStop: document.getElementById('btnStopCam'),
    btnMirror: document.getElementById('btnMirror'),
    selectCamera: document.getElementById('cameraSelect'),
    selectMode: document.getElementById('modeSelect'),

    // HUD & Telemetry
    hudStatus: document.getElementById('hudStatus'),
    hudFps: document.getElementById('hudFps'),
    hudResolution: document.getElementById('hudResolution'),
    metricFps: document.getElementById('metricFps'),
    metricLatency: document.getElementById('metricLatency'),
    metricHandState: document.getElementById('metricHandState'),
    metricTotalSamples: document.getElementById('metricTotalSamples'),

    // Hero Prediction
    heroGesture: document.getElementById('heroGesture'),
    heroIcon: document.getElementById('heroIcon'),
    heroMeaning: document.getElementById('heroMeaning'),
    gaugeFill: document.getElementById('gaugeFill'),
    gaugeValue: document.getElementById('gaugeValue'),

    // Probabilities list
    probContainer: document.getElementById('probabilitiesList'),

    // Sentence builder
    sentenceBox: document.getElementById('sentenceBox'),
    btnSpeakSentence: document.getElementById('btnSpeakSentence'),
    btnClearSentence: document.getElementById('btnClearSentence'),
    btnBackspace: document.getElementById('btnBackspace'),
    btnCopySentence: document.getElementById('btnCopySentence'),
    toggleTTS: document.getElementById('toggleTTS'),

    // Gesture cards
    gestureGrid: document.getElementById('gestureGrid'),

    // Retrain trigger
    btnRetrain: document.getElementById('btnRetrain'),
    trainingStatus: document.getElementById('trainingStatus')
  };

  const canvasCtx = el.canvas ? el.canvas.getContext('2d') : null;
  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d');

  // --- INITIALIZATION ---
  async function init() {
    setupEventListeners();
    await fetchSystemStatus();
    populateDeviceList();
    renderGestureCards();
    updateSentenceUI();

    // Check Speech Synthesis
    if (!('speechSynthesis' in window)) {
      if (el.toggleTTS) {
        el.toggleTTS.disabled = true;
        el.toggleTTS.parentElement.title = 'Text-to-speech not supported in this browser';
      }
    }
  }

  // --- SYSTEM STATUS & METADATA ---
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status endpoint failed');
      const data = await res.json();

      if (data.classes && data.classes.length > 0) {
        state.availableClasses = data.classes;
      }
      if (data.gesture_metadata) {
        state.metadata = Object.assign(state.metadata, data.gesture_metadata);
      }

      if (data.dataset_stats && data.dataset_stats.total_samples !== undefined) {
        if (el.metricTotalSamples) {
          el.metricTotalSamples.textContent = data.dataset_stats.total_samples;
        }
      }

      renderProbabilityBars({});
      renderGestureCards(data.dataset_stats ? data.dataset_stats.classes : {});
    } catch (err) {
      console.warn('Could not load status:', err);
    }
  }

  // --- CAMERA MANAGEMENT ---
  async function populateDeviceList() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      if (el.selectCamera) {
        el.selectCamera.innerHTML = '';
        if (videoDevices.length === 0) {
          const opt = document.createElement('option');
          opt.text = 'Default Webcam';
          el.selectCamera.appendChild(opt);
          return;
        }
        videoDevices.forEach((dev, index) => {
          const opt = document.createElement('option');
          opt.value = dev.deviceId;
          opt.text = dev.label || `Camera ${index + 1}`;
          el.selectCamera.appendChild(opt);
        });
      }
    } catch (e) {
      console.error('Error enumerating devices:', e);
    }
  }

  async function startBrowserCamera() {
    try {
      const deviceId = el.selectCamera ? el.selectCamera.value : undefined;
      const constraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.cameraStream = stream;
      el.video.srcObject = stream;
      await el.video.play();

      state.cameraRunning = true;
      updateCameraUI(true);

      // Match canvas resolution
      el.canvas.width = el.video.videoWidth || 640;
      el.canvas.height = el.video.videoHeight || 480;
      offscreenCanvas.width = 320; // Lower resolution for fast transmission & inference
      offscreenCanvas.height = 240;

      if (el.hudResolution) {
        el.hudResolution.textContent = `${el.canvas.width}x${el.canvas.height}`;
      }

      startInferenceLoop();
    } catch (err) {
      console.error('Failed to start camera:', err);
      alert('Unable to access webcam. Please check permissions in your browser.');
      updateCameraUI(false);
    }
  }

  function stopBrowserCamera() {
    state.cameraRunning = false;
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(t => t.stop());
      state.cameraStream = null;
    }
    if (el.video) {
      el.video.srcObject = null;
    }
    if (canvasCtx) {
      canvasCtx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    }
    updateCameraUI(false);
    resetPredictionHero();
  }

  function startServerStream() {
    stopBrowserCamera();
    state.streamMode = 'server';
    el.video.style.display = 'none';
    el.canvas.style.display = 'none';
    el.mjpeg.style.display = 'block';
    el.mjpeg.src = '/video_feed?' + new Date().getTime();
    el.placeholder.classList.add('hidden');
    state.cameraRunning = true;
    updateCameraUI(true);
  }

  function stopServerStream() {
    el.mjpeg.src = '';
    el.mjpeg.style.display = 'none';
    el.video.style.display = 'block';
    el.canvas.style.display = 'block';
    fetch('/api/stop_video_feed', { method: 'POST' }).catch(() => {});
    state.cameraRunning = false;
    updateCameraUI(false);
    resetPredictionHero();
  }

  function updateCameraUI(running) {
    if (running) {
      el.placeholder.classList.add('hidden');
      el.btnStart.style.display = 'none';
      el.btnStop.style.display = 'inline-flex';
      if (el.hudStatus) {
        el.hudStatus.innerHTML = '<span class="status-dot active"></span> LIVE FEED';
      }
    } else {
      el.placeholder.classList.remove('hidden');
      el.btnStart.style.display = 'inline-flex';
      el.btnStop.style.display = 'none';
      if (el.hudStatus) {
        el.hudStatus.innerHTML = '<span class="status-dot"></span> OFFLINE';
      }
      if (el.metricHandState) el.metricHandState.textContent = 'Camera Off';
    }
  }

  // --- REAL-TIME INFERENCE PIPELINE ---
  async function startInferenceLoop() {
    if (!state.cameraRunning || state.streamMode !== 'browser') return;

    if (!state.isProcessing && el.video.readyState >= 2) {
      state.isProcessing = true;
      const t0 = performance.now();

      // Capture frame onto small offscreen canvas
      offscreenCtx.drawImage(el.video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
      const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.65);

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        });

        if (response.ok) {
          const result = await response.json();
          handlePredictionResult(result, t0);
        }
      } catch (err) {
        console.warn('Inference error:', err);
      } finally {
        state.isProcessing = false;
      }
    }

    // Measure FPS
    state.frameCount++;
    const now = performance.now();
    if (now - state.lastFrameTime >= 1000) {
      state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
      state.frameCount = 0;
      state.lastFrameTime = now;
      if (el.hudFps) el.hudFps.textContent = `${state.fps} FPS`;
      if (el.metricFps) el.metricFps.textContent = `${state.fps}`;
    }

    // Schedule next frame with ~40ms delay (25 fps cap for smooth performance)
    if (state.cameraRunning) {
      setTimeout(startInferenceLoop, 40);
    }
  }

  // --- RESULT PROCESSING & SKELETON DRAWING ---
  function handlePredictionResult(data, sendTimestamp) {
    state.latency = Math.round(performance.now() - sendTimestamp);
    if (el.metricLatency) el.metricLatency.textContent = `${state.latency} ms`;

    // 1. Draw Skeleton
    if (canvasCtx) {
      canvasCtx.clearRect(0, 0, el.canvas.width, el.canvas.height);
      if (data.hand_detected && data.landmarks && data.landmarks.length === 21) {
        drawHandSkeleton(data.landmarks, el.canvas.width, el.canvas.height);
      }
    }

    // 2. Update Prediction & Confidence
    if (data.hand_detected) {
      state.currentGesture = data.prediction;
      state.currentConfidence = data.confidence || 0;
      updatePredictionHero(data.prediction, state.currentConfidence);
      if (el.metricHandState) el.metricHandState.textContent = 'Hand Detected';

      // Gesture stability & sentence builder trigger
      processGestureAccumulation(data.prediction, state.currentConfidence);
    } else {
      state.currentGesture = 'No hand detected';
      state.currentConfidence = 0;
      resetPredictionHero();
      if (el.metricHandState) el.metricHandState.textContent = 'Searching...';
      state.stableGestureCount = 0;
    }

    // 3. Update Class Probability Distribution
    renderProbabilityBars(data.probabilities || {}, data.prediction);
  }

  // --- DRAW HOLOGRAPHIC HAND SKELETON ---
  function drawHandSkeleton(landmarks, w, h) {
    if (!canvasCtx) return;

    // Line styles for futuristic cyber glow
    canvasCtx.save();
    canvasCtx.lineWidth = 3.5;
    canvasCtx.strokeStyle = '#00f2fe';
    canvasCtx.shadowColor = '#00f2fe';
    canvasCtx.shadowBlur = 10;
    canvasCtx.lineCap = 'round';
    canvasCtx.lineJoin = 'round';

    // Draw connection lines
    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];
      if (!p1 || !p2) return;

      canvasCtx.beginPath();
      canvasCtx.moveTo(p1.x * w, p1.y * h);
      canvasCtx.lineTo(p2.x * w, p2.y * h);
      canvasCtx.stroke();
    });

    // Draw landmark joint nodes
    landmarks.forEach((lm, idx) => {
      const x = lm.x * w;
      const y = lm.y * h;
      const isTip = FINGERTIPS.includes(idx);

      canvasCtx.beginPath();
      canvasCtx.arc(x, y, isTip ? 5.5 : 3.5, 0, Math.PI * 2);

      if (isTip) {
        canvasCtx.fillStyle = '#10b981';
        canvasCtx.shadowColor = '#10b981';
        canvasCtx.shadowBlur = 14;
      } else {
        canvasCtx.fillStyle = '#ffffff';
        canvasCtx.shadowColor = '#00f2fe';
        canvasCtx.shadowBlur = 8;
      }
      canvasCtx.fill();
    });

    canvasCtx.restore();
  }

  // --- HERO CARD & PROBABILITY UI ---
  function updatePredictionHero(label, confidence) {
    const meta = state.metadata[label] || { icon: '🤟', title: label, desc: 'Recognized gesture' };

    if (el.heroGesture) el.heroGesture.textContent = label;
    if (el.heroIcon) {
      el.heroIcon.textContent = meta.icon;
      el.heroIcon.classList.remove('icon-bounce');
      void el.heroIcon.offsetWidth; // Trigger reflow
      el.heroIcon.classList.add('icon-bounce');
    }
    if (el.heroMeaning) el.heroMeaning.textContent = meta.title;

    // Update Circular Gauge (circ = 251.2)
    const pct = Math.min(Math.max(confidence, 0), 1);
    const offset = 251.2 - (251.2 * pct);
    if (el.gaugeFill) el.gaugeFill.style.strokeDashoffset = offset;
    if (el.gaugeValue) el.gaugeValue.textContent = `${Math.round(pct * 100)}%`;

    highlightActiveGestureCard(label);
  }

  function resetPredictionHero() {
    if (el.heroGesture) el.heroGesture.textContent = 'No Hand Detected';
    if (el.heroIcon) el.heroIcon.textContent = '✋';
    if (el.heroMeaning) el.heroMeaning.textContent = 'Position hand in front of camera';
    if (el.gaugeFill) el.gaugeFill.style.strokeDashoffset = 251.2;
    if (el.gaugeValue) el.gaugeValue.textContent = '0%';
    highlightActiveGestureCard(null);
  }

  function renderProbabilityBars(probs, activeLabel) {
    if (!el.probContainer) return;
    el.probContainer.innerHTML = '';

    state.availableClasses.forEach(cls => {
      const p = probs[cls] !== undefined ? probs[cls] : 0;
      const pct = Math.round(p * 100);
      const isActive = cls === activeLabel && p > 0.45;
      const meta = state.metadata[cls] || { icon: '✨' };

      const row = document.createElement('div');
      row.className = `prob-row ${isActive ? 'active' : ''}`;
      row.innerHTML = `
        <div class="prob-header">
          <span class="prob-name-group">
            <span>${meta.icon || ''}</span>
            <span>${cls}</span>
          </span>
          <span class="prob-percent">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-bar" style="width: ${pct}%"></div>
        </div>
      `;
      el.probContainer.appendChild(row);
    });
  }

  // --- GESTURE ACCUMULATION & TEXT-TO-SPEECH ---
  function processGestureAccumulation(gesture, confidence) {
    // Only accumulate valid trained gesture with high confidence
    if (!gesture || gesture === 'No hand detected' || confidence < 0.75) {
      state.stableGestureCount = 0;
      return;
    }

    if (gesture === state.lastSpokenGesture) {
      // Cooldown to prevent repeating the exact same word endlessly
      return;
    }

    state.stableGestureCount++;

    // When held stably for 6 consecutive frames (~200ms)
    if (state.stableGestureCount >= 6) {
      state.lastSpokenGesture = gesture;
      state.stableGestureCount = 0;

      // Add to sentence builder
      state.sentenceWords.push(gesture);
      updateSentenceUI();

      // Speak if enabled
      if (state.ttsEnabled) {
        speakWord(gesture);
      }
    }
  }

  function speakWord(text) {
    if (!('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel(); // Stop current utterance
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error:', e);
    }
  }

  function updateSentenceUI() {
    if (!el.sentenceBox) return;
    if (state.sentenceWords.length === 0) {
      el.sentenceBox.innerHTML = '<span class="sentence-empty-hint">Detected gestures will formulate into translated text here...</span>';
      return;
    }

    el.sentenceBox.innerHTML = '';
    state.sentenceWords.forEach((w, idx) => {
      const chip = document.createElement('span');
      chip.className = 'word-chip';
      chip.innerHTML = `${w} <small style="cursor:pointer; opacity:0.6; margin-left:4px;" title="Remove">&times;</small>`;
      chip.querySelector('small').addEventListener('click', (e) => {
        e.stopPropagation();
        state.sentenceWords.splice(idx, 1);
        updateSentenceUI();
      });
      el.sentenceBox.appendChild(chip);
    });
  }

  // --- GESTURE GALLERY CARDS ---
  function renderGestureCards(classCounts = {}) {
    if (!el.gestureGrid) return;
    el.gestureGrid.innerHTML = '';

    state.availableClasses.forEach(cls => {
      const meta = state.metadata[cls] || { icon: '✋', title: cls, description: 'Trained gesture' };
      const descText = meta.description || meta.desc || 'Trained gesture';
      const count = classCounts[cls] || 200;

      const card = document.createElement('div');
      card.className = 'gesture-card';
      card.dataset.gesture = cls;
      card.innerHTML = `
        <span class="gesture-card-icon">${meta.icon}</span>
        <span class="gesture-card-name">${cls}</span>
        <span class="gesture-card-desc">${descText}</span>
        <span class="gesture-card-count">${count} samples</span>
      `;

      card.addEventListener('click', () => {
        speakWord(cls);
      });

      el.gestureGrid.appendChild(card);
    });
  }

  function highlightActiveGestureCard(gesture) {
    if (!el.gestureGrid) return;
    const cards = el.gestureGrid.querySelectorAll('.gesture-card');
    cards.forEach(card => {
      if (gesture && card.dataset.gesture === gesture) {
        card.classList.add('highlight');
      } else {
        card.classList.remove('highlight');
      }
    });
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Camera buttons
    if (el.btnStart) {
      el.btnStart.addEventListener('click', () => {
        if (state.streamMode === 'server') {
          startServerStream();
        } else {
          startBrowserCamera();
        }
      });
    }

    if (el.btnStop) {
      el.btnStop.addEventListener('click', () => {
        if (state.streamMode === 'server') {
          stopServerStream();
        } else {
          stopBrowserCamera();
        }
      });
    }

    // Mirror toggle
    if (el.btnMirror) {
      el.btnMirror.addEventListener('click', () => {
        state.isMirrored = !state.isMirrored;
        if (state.isMirrored) {
          el.video.classList.remove('no-mirror');
          el.canvas.classList.remove('no-mirror');
        } else {
          el.video.classList.add('no-mirror');
          el.canvas.classList.add('no-mirror');
        }
      });
    }

    // Camera device switch
    if (el.selectCamera) {
      el.selectCamera.addEventListener('change', () => {
        if (state.cameraRunning && state.streamMode === 'browser') {
          stopBrowserCamera();
          startBrowserCamera();
        }
      });
    }

    // Mode switch (Browser vs Server stream)
    if (el.selectMode) {
      el.selectMode.addEventListener('change', (e) => {
        const mode = e.target.value;
        const wasRunning = state.cameraRunning;

        if (mode === 'server') {
          if (wasRunning) startServerStream();
          else state.streamMode = 'server';
        } else {
          if (wasRunning) {
            stopServerStream();
            startBrowserCamera();
          }
          state.streamMode = 'browser';
        }
      });
    }

    // TTS Toggle
    if (el.toggleTTS) {
      el.toggleTTS.addEventListener('change', (e) => {
        state.ttsEnabled = e.target.checked;
      });
    }

    // Sentence builder actions
    if (el.btnSpeakSentence) {
      el.btnSpeakSentence.addEventListener('click', () => {
        const fullText = state.sentenceWords.join(' ');
        if (fullText) speakWord(fullText);
      });
    }

    if (el.btnClearSentence) {
      el.btnClearSentence.addEventListener('click', () => {
        state.sentenceWords = [];
        state.lastSpokenGesture = null;
        updateSentenceUI();
      });
    }

    if (el.btnBackspace) {
      el.btnBackspace.addEventListener('click', () => {
        state.sentenceWords.pop();
        updateSentenceUI();
      });
    }

    if (el.btnCopySentence) {
      el.btnCopySentence.addEventListener('click', () => {
        const text = state.sentenceWords.join(' ');
        if (text) {
          navigator.clipboard.writeText(text).then(() => {
            const orig = el.btnCopySentence.innerHTML;
            el.btnCopySentence.innerHTML = '✓ Copied!';
            setTimeout(() => { el.btnCopySentence.innerHTML = orig; }, 1500);
          });
        }
      });
    }

    // Model Retrain
    if (el.btnRetrain) {
      el.btnRetrain.addEventListener('click', async () => {
        if (!confirm('Retrain Random Forest classifier on dataset/gestures.csv?')) return;
        el.btnRetrain.disabled = true;
        el.btnRetrain.innerHTML = '<span class="spinner"></span> Retraining...';
        if (el.trainingStatus) el.trainingStatus.textContent = 'Training Random Forest model in background...';

        try {
          const res = await fetch('/api/train', { method: 'POST' });
          const data = await res.json();
          if (data.success) {
            alert(`Model retrained successfully! Accuracy: ${(data.accuracy * 100).toFixed(1)}%`);
            await fetchSystemStatus();
          } else {
            alert('Training error: ' + (data.error || 'Failed to retrain'));
          }
        } catch (err) {
          alert('Network or server error during training: ' + err.message);
        } finally {
          el.btnRetrain.disabled = false;
          el.btnRetrain.innerHTML = '⚡ Retrain Model';
          if (el.trainingStatus) el.trainingStatus.textContent = '';
        }
      });
    }
  }

  // Auto-init on DOM loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
