/**
 * GeoCaption Web Application - Interactive ML Frontend
 * Powers real-time inference with PyTorch, attention heatmap rendering,
 * confidence gauges, and interactive token scrubbing.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const state = {
    currentSampleId: 'airport',
    currentBase64: null,
    currentImageEl: null,
    captionResult: null,
    currentStepIndex: 0,
    isPlaying: false,
    playTimer: null,
  };

  // ---------------------------------------------------------------------------
  // DOM References
  // ---------------------------------------------------------------------------
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const inputPreviewImg = document.getElementById('inputPreviewImg');
  const generateBtn = document.getElementById('generateBtn');
  const captionOutputContainer = document.getElementById('captionOutputContainer');
  const captionText = document.getElementById('captionText');
  const copyCaptionBtn = document.getElementById('copyCaptionBtn');
  const downloadCaptionBtn = document.getElementById('downloadCaptionBtn');
  const gaugeProgressCircle = document.getElementById('gaugeProgressCircle');
  const gaugeValText = document.getElementById('gaugeValText');
  const entitiesContainer = document.getElementById('entitiesContainer');
  const attentionCanvas = document.getElementById('attentionCanvas');
  const playAttentionBtn = document.getElementById('playAttentionBtn');
  const attentionScrubber = document.getElementById('attentionScrubber');
  const stepIndicatorText = document.getElementById('stepIndicatorText');
  const samplesRow = document.getElementById('samplesRow');
  const previewExpandBtn = document.getElementById('previewExpandBtn');

  // Metrics DOM
  const valBleu1 = document.getElementById('valBleu1');
  const valCider = document.getElementById('valCider');
  const valRouge = document.getElementById('valRouge');
  const valMeteor = document.getElementById('valMeteor');

  const ctx = attentionCanvas.getContext('2d');

  // ---------------------------------------------------------------------------
  // Color Map (Jet / Heatmap palette for spatial attention)
  // ---------------------------------------------------------------------------
  function getHeatmapColor(value) {
    // value in [0, 1]
    const v = Math.min(1.0, Math.max(0.0, value));
    let r = 0, g = 0, b = 0;
    
    // Jet color gradient: blue -> cyan -> yellow -> orange -> red
    if (v < 0.25) {
      r = 0;
      g = Math.round(255 * (v / 0.25));
      b = 255;
    } else if (v < 0.5) {
      r = 0;
      g = 255;
      b = Math.round(255 * (1 - (v - 0.25) / 0.25));
    } else if (v < 0.75) {
      r = Math.round(255 * ((v - 0.5) / 0.25));
      g = 255;
      b = 0;
    } else {
      r = 255;
      g = Math.round(255 * (1 - (v - 0.75) / 0.25));
      b = 0;
    }
    
    // Alpha scaled with intensity: low attention is transparent, high attention glows
    const alpha = Math.min(0.72, v * 0.85 + 0.08);
    return { r, g, b, alpha };
  }

  // ---------------------------------------------------------------------------
  // Heatmap Canvas Renderer (Interpolates 8x8 Attention Grid onto Image)
  // ---------------------------------------------------------------------------
  function renderAttentionHeatmap(stepIndex) {
    if (!state.captionResult || !state.captionResult.attention_steps || state.captionResult.attention_steps.length === 0) {
      // Just draw the preview image cleanly
      if (inputPreviewImg.complete && inputPreviewImg.naturalWidth !== 0) {
        ctx.clearRect(0, 0, attentionCanvas.width, attentionCanvas.height);
        ctx.drawImage(inputPreviewImg, 0, 0, attentionCanvas.width, attentionCanvas.height);
      }
      return;
    }

    const stepData = state.captionResult.attention_steps[stepIndex];
    if (!stepData) return;

    const grid = stepData.grid; // 8x8 2D array of floats
    const cw = attentionCanvas.width;
    const ch = attentionCanvas.height;

    // 1. Draw base satellite image
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(inputPreviewImg, 0, 0, cw, ch);

    // 2. Compute min/max for normalization
    let minVal = Infinity, maxVal = -Infinity;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const val = grid[r][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const range = (maxVal - minVal) > 1e-7 ? (maxVal - minVal) : 1.0;

    // 3. Render offscreen 8x8 canvas then scale up with smoothing
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 8;
    offCanvas.height = 8;
    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(8, 8);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const norm = (grid[r][c] - minVal) / range;
        const color = getHeatmapColor(norm);
        const idx = (r * 8 + c) * 4;
        imgData.data[idx] = color.r;
        imgData.data[idx + 1] = color.g;
        imgData.data[idx + 2] = color.b;
        imgData.data[idx + 3] = Math.round(color.alpha * 255);
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    // 4. Draw smooth upsampled heatmap over base image
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.globalCompositeOperation = 'screen';
    ctx.drawImage(offCanvas, 0, 0, cw, ch);
    ctx.restore();

    // 5. Update step controls & active token highlighting
    updateStepUI(stepIndex, stepData.token);
  }

  // ---------------------------------------------------------------------------
  // Step UI Update & Active Token Highlighter
  // ---------------------------------------------------------------------------
  function updateStepUI(stepIndex, tokenText) {
    const totalSteps = state.captionResult ? state.captionResult.attention_steps.length : 1;
    attentionScrubber.value = stepIndex + 1;
    stepIndicatorText.textContent = `Step ${stepIndex + 1} / ${totalSteps} ("${tokenText}")`;

    // Highlight active token word in the caption box
    const tokenSpans = captionOutputContainer.querySelectorAll('.token-word');
    tokenSpans.forEach((span, idx) => {
      if (idx === stepIndex) {
        span.classList.add('active-step');
      } else {
        span.classList.remove('active-step');
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Auto-Play Attention Animation
  // ---------------------------------------------------------------------------
  function toggleAttentionPlay() {
    if (!state.captionResult || !state.captionResult.attention_steps) return;
    const totalSteps = state.captionResult.attention_steps.length;

    if (state.isPlaying) {
      clearInterval(state.playTimer);
      state.isPlaying = false;
      playAttentionBtn.textContent = '▶';
    } else {
      state.isPlaying = true;
      playAttentionBtn.textContent = '⏸';
      
      state.playTimer = setInterval(() => {
        state.currentStepIndex = (state.currentStepIndex + 1) % totalSteps;
        renderAttentionHeatmap(state.currentStepIndex);
      }, 550);
    }
  }

  // ---------------------------------------------------------------------------
  // Circular Confidence Gauge Animator
  // ---------------------------------------------------------------------------
  function updateConfidenceGauge(confidence) {
    const circumference = 2 * Math.PI * 36; // ~226.19
    const targetOffset = circumference - (confidence * circumference);
    
    gaugeProgressCircle.style.strokeDashoffset = targetOffset;
    
    // Animate percentage text
    let currentPct = 0;
    const targetPct = Math.round(confidence * 100);
    const stepTime = 15;
    const timer = setInterval(() => {
      currentPct += 2;
      if (currentPct >= targetPct) {
        currentPct = targetPct;
        clearInterval(timer);
      }
      gaugeValText.textContent = `${currentPct}%`;
    }, stepTime);
  }

  // ---------------------------------------------------------------------------
  // Entities Bar List Renderer
  // ---------------------------------------------------------------------------
  function renderEntities(entities) {
    entitiesContainer.innerHTML = '';
    if (!entities || entities.length === 0) return;

    entities.forEach(item => {
      const row = document.createElement('div');
      row.className = 'entity-bar-row';
      const pct = Math.round(item.score * 100);

      row.innerHTML = `
        <span class="entity-name" title="${item.name}">${item.name}</span>
        <div class="entity-progress-track">
          <div class="entity-progress-fill" style="width: 0%;"></div>
        </div>
        <span class="entity-score">${item.score.toFixed(2)}</span>
      `;
      entitiesContainer.appendChild(row);

      // Trigger animation next tick
      setTimeout(() => {
        const fill = row.querySelector('.entity-progress-fill');
        if (fill) fill.style.width = `${pct}%`;
      }, 50);
    });
  }

  // ---------------------------------------------------------------------------
  // Render Interactive Caption Tokens
  // ---------------------------------------------------------------------------
  function renderCaptionTokens(tokens) {
    captionOutputContainer.innerHTML = '';
    const p = document.createElement('p');

    tokens.forEach((tok, idx) => {
      const span = document.createElement('span');
      span.className = 'token-word';
      span.textContent = tok;
      span.title = `Click to view attention for "${tok}"`;

      span.addEventListener('click', () => {
        if (state.isPlaying) toggleAttentionPlay();
        state.currentStepIndex = idx;
        renderAttentionHeatmap(idx);
      });

      p.appendChild(span);
      p.appendChild(document.createTextNode(' '));
    });

    captionOutputContainer.appendChild(p);
  }

  // ---------------------------------------------------------------------------
  // Live PyTorch Inference Dispatcher
  // ---------------------------------------------------------------------------
  async function runCaptionInference() {
    generateBtn.disabled = true;
    const originalBtnText = generateBtn.innerHTML;
    generateBtn.innerHTML = `
      <svg class="spin-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
        <path d="M12 2a10 10 0 0 1 10 10"/>
      </svg>
      <span>Generating with PyTorch...</span>
    `;

    try {
      const payload = { beam_size: 3, max_len: 30 };
      if (state.currentBase64) {
        payload.image_base64 = state.currentBase64;
      } else {
        payload.sample_id = state.currentSampleId;
      }

      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || 'Inference failed on server.');
      }

      state.captionResult = data;
      state.currentStepIndex = 0;

      // 1. Render Caption Words as Interactive Tokens
      renderCaptionTokens(data.tokens);

      // 2. Animate Confidence Gauge
      updateConfidenceGauge(data.confidence);

      // 3. Render Entity Breakdown
      renderEntities(data.entities);

      // 4. Configure Attention Scrubber
      const totalSteps = data.attention_steps.length;
      attentionScrubber.min = 1;
      attentionScrubber.max = totalSteps;
      attentionScrubber.value = 1;

      // 5. Render Step 1 Attention Map
      renderAttentionHeatmap(0);

    } catch (err) {
      console.error('Caption generation error:', err);
      captionOutputContainer.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
    } finally {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalBtnText;
    }
  }

  // ---------------------------------------------------------------------------
  // Fetch Real Test Metrics
  // ---------------------------------------------------------------------------
  async function loadTestMetrics() {
    try {
      const res = await fetch('/api/metrics');
      const metrics = await res.json();
      
      if (metrics['BLEU-1']) {
        valBleu1.textContent = (metrics['BLEU-1'] > 1 ? (metrics['BLEU-1'] / 100).toFixed(3) : metrics['BLEU-1'].toFixed(3));
      }
      if (metrics['CIDEr']) {
        valCider.textContent = metrics['CIDEr'].toFixed(3);
      }
      if (metrics['ROUGE-L']) {
        valRouge.textContent = (metrics['ROUGE-L'] > 1 ? (metrics['ROUGE-L'] / 100).toFixed(3) : metrics['ROUGE-L'].toFixed(3));
      }
      if (metrics['METEOR']) {
        valMeteor.textContent = (metrics['METEOR'] > 1 ? (metrics['METEOR'] / 100).toFixed(3) : metrics['METEOR'].toFixed(3));
      }
    } catch (e) {
      console.warn('Could not load test metrics:', e);
    }
  }

  // ---------------------------------------------------------------------------
  // Image Upload & Dropzone Handling
  // ---------------------------------------------------------------------------
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUserFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUserFile(e.target.files[0]);
    }
  });

  function handleUserFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPG, PNG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (evt) => {
      state.currentBase64 = evt.target.result;
      state.currentSampleId = null;
      inputPreviewImg.src = state.currentBase64;
      
      // Deselect sample presets
      document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));

      inputPreviewImg.onload = () => {
        renderAttentionHeatmap(-1); // reset heatmap canvas
        runCaptionInference();
      };
    };
    reader.readAsDataURL(file);
  }

  // ---------------------------------------------------------------------------
  // Sample Presets Click Handler
  // ---------------------------------------------------------------------------
  samplesRow.addEventListener('click', (e) => {
    const card = e.target.closest('.sample-thumb-card');
    if (!card) return;

    const sampleId = card.getAttribute('data-sample-id');
    if (!sampleId) return;

    document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    state.currentSampleId = sampleId;
    state.currentBase64 = null;
    inputPreviewImg.src = `images/samples/${sampleId}.jpg`;

    inputPreviewImg.onload = () => {
      runCaptionInference();
    };
  });

  // ---------------------------------------------------------------------------
  // Controls & Action Buttons
  // ---------------------------------------------------------------------------
  generateBtn.addEventListener('click', () => runCaptionInference());

  playAttentionBtn.addEventListener('click', toggleAttentionPlay);

  attentionScrubber.addEventListener('input', (e) => {
    if (state.isPlaying) toggleAttentionPlay();
    const idx = parseInt(e.target.value, 10) - 1;
    state.currentStepIndex = idx;
    renderAttentionHeatmap(idx);
  });

  copyCaptionBtn.addEventListener('click', () => {
    const text = state.captionResult ? state.captionResult.caption : captionText.textContent;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = copyCaptionBtn.querySelector('span').textContent;
      copyCaptionBtn.querySelector('span').textContent = 'Copied!';
      setTimeout(() => {
        copyCaptionBtn.querySelector('span').textContent = originalText;
      }, 1800);
    });
  });

  downloadCaptionBtn.addEventListener('click', () => {
    const exportData = {
      model: "ConvNeXt-Tiny + LoRA (r=16) + Bahdanau Spatial Attention + LSTM",
      caption: state.captionResult ? state.captionResult.caption : captionText.textContent,
      confidence: state.captionResult ? state.captionResult.confidence : 0.92,
      entities: state.captionResult ? state.captionResult.entities : [],
      tokens: state.captionResult ? state.captionResult.tokens : [],
      timestamp: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `geocaption_${state.currentSampleId || 'upload'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  previewExpandBtn.addEventListener('click', () => {
    window.open(inputPreviewImg.src, '_blank');
  });

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  loadTestMetrics();

  // Draw initial state once preview image loads
  if (inputPreviewImg.complete) {
    runCaptionInference();
  } else {
    inputPreviewImg.onload = () => runCaptionInference();
  }
});
