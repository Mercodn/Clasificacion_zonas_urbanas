/* ══════════════════════════════════════
   LIGHT POLLUTION CLASSIFIER · main.js
   JoanMoreno · Cundinamarca VIIRS
   ══════════════════════════════════════ */

'use strict';

// ── DEMO COORDINATES (puntos representativos de Cundinamarca) ──
const DEMO_POINTS = [
  { lat: 4.7110, lon: -74.0721, avg_rad: 35.5,  label: 'Bogotá Centro' },
  { lat: 4.6097, lon: -74.0817, avg_rad: 12.3,  label: 'Bogotá Sur' },
  { lat: 4.8633, lon: -74.0436, avg_rad: 4.2,   label: 'Chía' },
  { lat: 5.0333, lon: -73.9833, avg_rad: 0.85,  label: 'Zipaquirá' },
  { lat: 4.3500, lon: -74.3667, avg_rad: 0.28,  label: 'La Mesa' },
  { lat: 4.9167, lon: -74.6333, avg_rad: 0.15,  label: 'Villeta (rural)' },
  { lat: 4.4986, lon: -73.9928, avg_rad: 0.55,  label: 'Fusagasugá' },
  { lat: 5.5333, lon: -73.3667, avg_rad: 0.19,  label: 'Medina (selva)' },
];

// ── ZONE COLORS ──
const ZONE_COLORS = {
  0: '#000080',
  1: '#0040FF',
  2: '#00CFFF',
  3: '#00FF80',
  4: '#FFB300',
  5: '#FF1100'
};

// ── LEAFLET MAP ──
let map = null;
let markersLayer = null;
let predictedMarker = null;

function initMap() {
  if (map) return;

  map = L.map('map', {
    center: [4.7110, -74.0721],
    zoom: 8,
    zoomControl: true,
    attributionControl: false
  });

  // Tile oscuro
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
    attribution: '©CartoDB'
  }).addTo(map);

  L.control.attribution({ position: 'bottomright', prefix: false })
    .addAttribution('<span style="color:#4a6080;font-size:9px">©CartoDB ©OSM</span>')
    .addTo(map);

  markersLayer = L.layerGroup().addTo(map);

  // Cargar puntos del dataset
  loadMapData();

  // Click en mapa para autocompletar coords
  map.on('click', function(e) {
    const lat = e.latlng.lat.toFixed(6);
    const lon = e.latlng.lng.toFixed(6);
    document.getElementById('inp-lat').value = lat;
    document.getElementById('inp-lon').value = lon;
    showToast(`Coordenadas capturadas: ${lat}, ${lon}`);
  });
}

function loadMapData() {
  fetch('/api/data')
    .then(r => r.json())
    .then(points => {
      markersLayer.clearLayers();
      points.forEach(p => {
        const r = p.avg_rad;
        const size = Math.max(4, Math.min(14, 3 + Math.log1p(r) * 2.5));

        const icon = L.divIcon({
          className: '',
          html: `<div style="
            width:${size}px;height:${size}px;
            border-radius:50%;
            background:${p.pred_color};
            opacity:0.82;
            box-shadow:0 0 ${Math.round(size*1.5)}px ${p.pred_color}88;
          "></div>`,
          iconSize: [size, size],
          iconAnchor: [size / 2, size / 2]
        });

        const marker = L.marker([p.lat, p.lon], { icon })
          .bindPopup(`
            <div style="min-width:160px">
              <div style="font-weight:700;color:${p.pred_color};margin-bottom:4px">${p.pred_name}</div>
              <div style="color:#8aa0c0;font-size:10px">avg_rad: <b style="color:#d0dff5">${p.avg_rad.toFixed(4)}</b> nW/cm²/sr</div>
              <div style="color:#8aa0c0;font-size:10px">lat: ${p.lat.toFixed(4)} · lon: ${p.lon.toFixed(4)}</div>
            </div>
          `);
        markersLayer.addLayer(marker);
      });
    })
    .catch(err => console.error('Error loading map data:', err));
}

// ── FORMULARIO DE PREDICCIÓN ──
document.getElementById('predict-form').addEventListener('submit', function(e) {
  e.preventDefault();
  runPrediction();
});

function runPrediction() {
  const lat    = parseFloat(document.getElementById('inp-lat').value);
  const lon    = parseFloat(document.getElementById('inp-lon').value);
  const avgRad = parseFloat(document.getElementById('inp-rad').value);

  // Validar
  hideError();
  if (isNaN(lat) || isNaN(lon) || isNaN(avgRad)) {
    showError('Por favor completa todos los campos con valores numéricos válidos.');
    return;
  }
  if (lat < 3.6 || lat > 5.9 || lon < -75.0 || lon > -72.9) {
    showError('Las coordenadas están fuera del rango de Cundinamarca (lat: 3.6–5.9, lon: -75.0–-72.9).');
    return;
  }
  if (avgRad < 0 || avgRad > 500) {
    showError('La radiancia debe estar entre 0 y 500 nW/cm²/sr.');
    return;
  }

  showSpinner(true);
  hideResult();

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat, lon, avg_rad: avgRad })
  })
  .then(r => r.json())
  .then(data => {
    showSpinner(false);
    if (data.error) { showError(data.error); return; }
    renderResult(data);
    updateMapWithPoint(data);
  })
  .catch(err => {
    showSpinner(false);
    showError('Error de conexión con el servidor.');
    console.error(err);
  });
}

function renderResult(data) {
  const panel = document.getElementById('result-panel');

  // Badge de zona
  document.getElementById('res-badge').innerHTML = `
    <span class="result-zone-badge" style="color:${data.zone_color};border-color:${data.zone_color}20;background:${data.zone_color}12">
      <span style="width:10px;height:10px;border-radius:50%;background:${data.zone_color};box-shadow:0 0 8px ${data.zone_color};display:inline-block;flex-shrink:0"></span>
      ${data.zone_name}
    </span>
  `;

  // Coordenadas y radiancia
  document.getElementById('res-coords').innerHTML = `
    <span class="text-accent">▸</span> lat <b>${data.lat}</b> &nbsp;·&nbsp;
    lon <b>${data.lon}</b> &nbsp;·&nbsp;
    avg_rad <b>${data.avg_rad}</b> nW/cm²/sr
  `;

  // Descripción
  document.getElementById('res-desc').textContent = data.zone_desc;

  // Confianza
  document.getElementById('res-conf-val').textContent = `${data.confidence}%`;
  document.getElementById('res-conf-fill').style.width = `${data.confidence}%`;

  // Probabilidades por zona
  const probContainer = document.getElementById('res-probs');
  probContainer.innerHTML = '';
  Object.entries(data.probabilities).forEach(([name, pct]) => {
    const zoneId = Object.values({
      0:'Oscuro Natural',1:'Rural Bajo',2:'Suburbano',
      3:'Urbano Moderado',4:'Urbano Alto',5:'Metropolitano'
    }).indexOf(name);
    const color = ZONE_COLORS[zoneId] || '#00d4ff';

    probContainer.innerHTML += `
      <div class="prob-row">
        <div class="prob-label">${name}</div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="prob-val">${pct}%</div>
      </div>
    `;
  });

  // Heatmap actualizado
  if (data.heatmap_b64) {
    const img = document.getElementById('heatmap-img');
    img.style.opacity = '0.4';
    img.src = 'data:image/png;base64,' + data.heatmap_b64;
    img.onload = () => { img.style.opacity = '1'; };
  }

  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateMapWithPoint(data) {
  if (!map) return;

  if (predictedMarker) map.removeLayer(predictedMarker);

  const icon = L.divIcon({
    className: '',
    html: `<div style="
      width:20px;height:20px;border-radius:50%;
      background:${data.zone_color};
      border:2px solid white;
      box-shadow:0 0 16px ${data.zone_color},0 0 6px white;
      animation:pulse-marker 1.5s ease-in-out infinite;
    "></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });

  predictedMarker = L.marker([data.lat, data.lon], { icon, zIndexOffset: 1000 })
    .bindPopup(`
      <div style="min-width:180px">
        <div style="font-weight:700;color:${data.zone_color};font-size:12px;margin-bottom:6px">★ ${data.zone_name}</div>
        <div style="color:#8aa0c0;font-size:10px">avg_rad: <b style="color:#d0dff5">${data.avg_rad}</b> nW/cm²/sr</div>
        <div style="color:#8aa0c0;font-size:10px">lat: ${data.lat} · lon: ${data.lon}</div>
        <div style="color:#8aa0c0;font-size:10px;margin-top:4px">Confianza: <b style="color:${data.zone_color}">${data.confidence}%</b></div>
      </div>
    `)
    .addTo(map)
    .openPopup();

  map.flyTo([data.lat, data.lon], 10, { duration: 1.2 });
}

// ── BOTÓN DEMO ──
let demoIndex = 0;
document.getElementById('btn-demo').addEventListener('click', function() {
  const pt = DEMO_POINTS[demoIndex % DEMO_POINTS.length];
  demoIndex++;
  document.getElementById('inp-lat').value = pt.lat;
  document.getElementById('inp-lon').value = pt.lon;
  document.getElementById('inp-rad').value = pt.avg_rad;
  showToast(`🛰️ Demo: ${pt.label}`);
});

// ── UI HELPERS ──
function showSpinner(visible) {
  document.getElementById('spinner-wrap').style.display = visible ? 'block' : 'none';
  document.getElementById('btn-submit').disabled = visible;
}

function hideResult() {
  document.getElementById('result-panel').style.display = 'none';
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('error-msg').style.display = 'none';
}

function showToast(msg) {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    toast.style.cssText = `
      position:fixed;bottom:24px;right:24px;z-index:9999;
      background:#0d1426;border:1px solid #1a2644;
      color:#d0dff5;font-family:'Space Mono',monospace;
      font-size:11px;padding:.6rem 1.1rem;border-radius:8px;
      box-shadow:0 8px 30px rgba(0,0,0,.5);
      opacity:0;transition:opacity .3s;pointer-events:none;
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = '1';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { toast.style.opacity = '0'; }, 2800);
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', function() {
  // Inicializar mapa con pequeño delay para layout
  setTimeout(initMap, 200);

  // Animación de entrada de stat cards
  const cards = document.querySelectorAll('.stat-card');
  cards.forEach((c, i) => {
    c.style.opacity = '0';
    c.style.transform = 'translateY(16px)';
    setTimeout(() => {
      c.style.transition = 'opacity .5s ease, transform .5s ease';
      c.style.opacity = '1';
      c.style.transform = 'translateY(0)';
    }, 100 + i * 80);
  });

  // Inyectar keyframe para marker animado
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse-marker {
      0%,100% { transform: scale(1); opacity: 1; }
      50%      { transform: scale(1.3); opacity: 0.7; }
    }
  `;
  document.head.appendChild(style);
});
