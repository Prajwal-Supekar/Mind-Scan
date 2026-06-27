/* =====================================================
   MindScan — main.js
   Neural constellation bg · 3D card tilt
   Multi-step form · Browser emotion detection
   ===================================================== */

/* ── Neural Constellation Background ────────────────── */
class NeuralField {
  constructor(canvas) {
    this.c   = canvas;
    this.ctx = canvas.getContext('2d');
    this.nodes = [];
    this.resize();
    this.init();
    this.tick();
    window.addEventListener('resize', () => { this.resize(); this.init(); });
  }

  resize() {
    this.c.width  = window.innerWidth;
    this.c.height = window.innerHeight;
  }

  init() {
    const n = Math.floor((this.c.width * this.c.height) / 14000);
    this.nodes = Array.from({ length: n }, () => ({
      x:  Math.random() * this.c.width,
      y:  Math.random() * this.c.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r:  Math.random() * 1.8 + 0.6,
    }));
  }

  tick() {
    const { ctx, c, nodes } = this;
    ctx.clearRect(0, 0, c.width, c.height);

    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > c.width)  n.vx *= -1;
      if (n.y < 0 || n.y > c.height) n.vy *= -1;

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(168,85,247,0.45)';
      ctx.fill();
    }

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          const alpha = 0.18 * (1 - d / 130);
          ctx.strokeStyle = `rgba(168,85,247,${alpha})`;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(() => this.tick());
  }
}

/* ── 3D Card Tilt ─────────────────────────────────── */
function initTilt() {
  document.querySelectorAll('.card-tilt').forEach(card => {
    const inner = card.querySelector('.card-inner');
    if (!inner) return;
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width  - 0.5;
      const y = (e.clientY - r.top)  / r.height - 0.5;
      inner.style.transform = `rotateY(${x*13}deg) rotateX(${-y*9}deg) scale(1.03)`;
    });
    card.addEventListener('mouseleave', () => {
      inner.style.transform = '';
    });
  });
}

/* ── Multi-Step Depression Form ───────────────────── */
let currentStep = 1;
const TOTAL_STEPS = 4;

function goToStep(step) {
  if (step < 1 || step > TOTAL_STEPS) return;

  document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
  const target = document.getElementById(`step-${step}`);
  if (target) target.classList.add('active');

  // Update step dots
  document.querySelectorAll('.step-dot').forEach((dot, i) => {
    dot.classList.remove('active', 'done');
    if (i + 1 < step)      dot.classList.add('done');
    else if (i + 1 === step) dot.classList.add('active');
    if (dot.classList.contains('done')) dot.textContent = '✓';
    else dot.textContent = i + 1;
  });

  // Update step lines
  document.querySelectorAll('.step-line-fill').forEach((line, i) => {
    line.classList.toggle('filled', i + 1 < step);
  });

  currentStep = step;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function nextStep() {
  if (validateStep(currentStep)) goToStep(currentStep + 1);
}

function prevStep() {
  goToStep(currentStep - 1);
}

function validateStep(step) {
  const s = document.getElementById(`step-${step}`);
  if (!s) return true;
  let valid = true;
  s.querySelectorAll('[required]').forEach(el => {
    el.style.borderColor = '';
    if (!el.value.trim()) {
      el.style.borderColor = 'rgba(251,113,133,0.7)';
      el.style.boxShadow   = '0 0 0 3px rgba(251,113,133,0.1)';
      valid = false;
      el.addEventListener('input', () => {
        el.style.borderColor = ''; el.style.boxShadow = '';
      }, { once: true });
    }
  });
  return valid;
}

/* ── Sentiment Analysis ───────────────────────────── */
const sentimentHistory = [];

function analyzeText() {
  const text = (document.getElementById('textInput') || {}).value || '';
  if (!text.trim()) return;

  const resultEl  = document.getElementById('sentResult');
  const gaugePos  = document.getElementById('gaugeFillPos');
  const gaugeNeg  = document.getElementById('gaugeFillNeg');
  const gaugePct  = document.getElementById('gaugePct');
  const histList  = document.getElementById('historyList');

  if (resultEl) resultEl.innerHTML = '<div class="spinner" style="margin-top:12px;"></div>';

  fetch('/analyze_text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'text=' + encodeURIComponent(text),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { if (resultEl) resultEl.textContent = data.error; return; }
    const item  = Array.isArray(data) ? data[0] : data;
    const label = (item.label || 'NEUTRAL').toUpperCase();
    const score = Math.round((item.score || 0.5) * 100);
    const isPos = label === 'POSITIVE';
    const isNeg = label === 'NEGATIVE';

    // Gauge animation
    if (gaugePos)  gaugePos.style.width = isPos ? score + '%' : '0%';
    if (gaugeNeg)  gaugeNeg.style.width = isNeg ? score + '%' : '0%';
    if (gaugePct)  gaugePct.textContent = score + '%';

    const cls = isPos ? 'positive' : isNeg ? 'negative' : 'neutral';

    if (resultEl) {
      resultEl.innerHTML = `
        <span class="result-pill ${cls}">
          <span class="status-dot"></span>
          ${label} — ${score}% confidence
        </span>`;
    }

    // History
    sentimentHistory.unshift({ text: text.slice(0, 60), label, score });
    if (histList) {
      histList.innerHTML = sentimentHistory.slice(0, 5).map(h => `
        <li class="history-item">
          <span>${h.text}${h.text.length >= 60 ? '…' : ''}</span>
          <span class="result-pill ${h.label.toLowerCase()} btn-sm">${h.label}</span>
        </li>`).join('');
    }

    const inp = document.getElementById('textInput');
    if (inp) inp.value = '';
  })
  .catch(err => { if (resultEl) resultEl.textContent = 'Error: ' + err.message; });
}

function getNextQuestion() {
  fetch('/get_next_question')
  .then(r => r.json())
  .then(d => {
    const el = document.getElementById('questionText');
    if (el) {
      el.style.opacity = 0;
      setTimeout(() => { el.textContent = d.question; el.style.opacity = 1; }, 200);
      el.style.transition = 'opacity 0.3s';
    }
  });
}

/* ── Live Emotion Detection ───────────────────────── */
let emotionStream    = null;
let emotionInterval  = null;
let isDetecting      = false;

const EMOTION_COLORS = {
  Happy:     '#34d399', Neutral:   '#38bdf8', Sad:       '#818cf8',
  Angry:     '#fb7185', Fearful:   '#fbbf24', Disgusted: '#a78bfa',
  Surprised: '#f0abfc', default:   '#a855f7',
};

async function startEmotionDetection() {
  const video   = document.getElementById('videoEl');
  const canvas  = document.getElementById('captureCanvas');
  const overlay = document.getElementById('overlayCanvas');
  if (!video) return;

  try {
    emotionStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = emotionStream;
    await video.play();
    isDetecting = true;
    updateEmotionUI(true);

    emotionInterval = setInterval(async () => {
      if (!isDetecting) return;
      canvas.width  = video.videoWidth  || 320;
      canvas.height = video.videoHeight || 240;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const frame = canvas.toDataURL('image/jpeg', 0.7);

      try {
        const res  = await fetch('/detect_emotion_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame }),
        });
        const data = await res.json();
        renderEmotionOverlay(overlay, video, data);
        updateEmotionPanel(data);
      } catch (e) { /* network hiccup — skip frame */ }
    }, 800);

  } catch (err) {
    alert('Camera access denied or not available: ' + err.message);
  }
}

function stopEmotionDetection() {
  isDetecting = false;
  clearInterval(emotionInterval);
  if (emotionStream) {
    emotionStream.getTracks().forEach(t => t.stop());
    emotionStream = null;
  }
  const video = document.getElementById('videoEl');
  if (video) { video.srcObject = null; }
  updateEmotionUI(false);
  const ol = document.getElementById('overlayCanvas');
  if (ol) ol.getContext('2d').clearRect(0, 0, ol.width, ol.height);
}

function updateEmotionUI(running) {
  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const status   = document.getElementById('emotionStatus');
  if (startBtn) startBtn.style.display = running ? 'none' : '';
  if (stopBtn)  stopBtn.style.display  = running ? '' : 'none';
  if (status)   status.textContent     = running ? 'DETECTING…' : 'CAMERA OFF';
}

function renderEmotionOverlay(canvas, video, data) {
  if (!canvas || !data.faces) return;
  canvas.width  = video.clientWidth;
  canvas.height = video.clientHeight;
  const ctx  = canvas.getContext('2d');
  const scaleX = canvas.width;
  const scaleY = canvas.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const face of data.faces) {
    const color = EMOTION_COLORS[face.emotion] || EMOTION_COLORS.default;
    // Mirror X because video is CSS-flipped
    const rx = (1 - face.x - face.w) * scaleX;
    const ry = face.y * scaleY;
    const rw = face.w * scaleX;
    const rh = face.h * scaleY;

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.shadowBlur  = 0;

    // Label badge
    ctx.fillStyle   = color + 'cc';
    ctx.fillRect(rx, ry - 26, rw, 26);
    ctx.fillStyle   = '#fff';
    ctx.font        = 'bold 12px IBM Plex Mono, monospace';
    ctx.fillText(`${face.emotion}  ${face.confidence}%`, rx + 6, ry - 8);
  }
}

const emotionBuffer = [];

function updateEmotionPanel(data) {
  const emoVal  = document.getElementById('emoValue');
  const emoConf = document.getElementById('emoConf');
  const emoHist = document.getElementById('emoHistory');

  if (emoVal)  emoVal.textContent  = data.emotion || '—';
  if (emoConf) emoConf.textContent = data.confidence ? data.confidence + '% confidence' : '';

  if (data.emotion && data.emotion !== 'No Face Detected' && data.emotion !== 'Error') {
    emotionBuffer.unshift(data.emotion);
    if (emotionBuffer.length > 8) emotionBuffer.pop();
  }

  if (emoHist) {
    emoHist.innerHTML = [...new Set(emotionBuffer)].map(e =>
      `<span class="emo-pill">${e}</span>`
    ).join('');
  }
}

/* ── Boot ─────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Neural background
  const canvas = document.getElementById('neuralCanvas');
  if (canvas) new NeuralField(canvas);

  // 3D tilt
  initTilt();

  // Wizard init
  const step1 = document.getElementById('step-1');
  if (step1) goToStep(1);

  // Sentiment enter key
  const textArea = document.getElementById('textInput');
  if (textArea) {
    textArea.addEventListener('keydown', e => {
      if (e.ctrlKey && e.key === 'Enter') analyzeText();
    });
  }

  // Login Enter key
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const loginBtn = document.getElementById('loginBtn');
      if (loginBtn) loginBtn.click();
    }
  });
});
