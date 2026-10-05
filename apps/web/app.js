const $ = (selector) => document.querySelector(selector);
const app = {tracks: [], sourceId: null, destinationId: null, mix: null, animationFrame: null};

function toast(title, message) {
  const node = $('#toast');
  node.querySelector('b').textContent = title;
  node.querySelector('span').textContent = message;
  node.classList.add('show');
  window.setTimeout(() => node.classList.remove('show'), 3800);
}

async function post(url, body) {
  const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Something went wrong');
  return result;
}

const byId = (id) => app.tracks.find((track) => track.id === id);

function deckMarkup(track, deck) {
  if (!track) return;
  $(`#title-${deck}`).textContent = track.title;
  $(`#artist-${deck}`).textContent = track.artist;
  $(`#bpm-${deck}`).textContent = track.bpm.toFixed(1);
  $(`#key-${deck}`).textContent = track.key;
  $(`#energy-${deck}`).textContent = `${track.energy.toFixed(1)}/10`;
  $(`#cover-${deck}`).className = `cover ${track.color}`;
}

function updateSelection() {
  const source = byId(app.sourceId);
  const destination = byId(app.destinationId);
  deckMarkup(source, 'a'); deckMarkup(destination, 'b');
  document.querySelectorAll('.track-card').forEach((card) => {
    card.classList.toggle('selected-a', card.dataset.id === app.sourceId);
    card.classList.toggle('selected-b', card.dataset.id === app.destinationId);
  });
  const valid = source && destination && source.id !== destination.id;
  $('#mix-button').disabled = !valid;
  $('#selection-help').textContent = valid ? `${source.title} → ${destination.title}` : 'Choose two different songs below.';
}

function selectTrack(deck, id) {
  if (deck === 'a') app.sourceId = id; else app.destinationId = id;
  updateSelection();
}

function renderLibrary() {
  const grid = $('#track-grid'); grid.replaceChildren();
  app.tracks.forEach((track) => {
    const card = document.createElement('article');
    card.className = 'track-card'; card.dataset.id = track.id;
    card.innerHTML = `<div class="track-cover ${track.color}"></div><h3></h3><p></p><div class="card-meta"><span>${track.bpm.toFixed(1)} BPM</span><span>${track.key}</span><span>${track.duration}</span></div><div class="card-actions"><button type="button">Send to A</button><button type="button">Send to B</button></div>`;
    card.querySelector('h3').textContent = track.title; card.querySelector('p').textContent = track.artist;
    const [toA, toB] = card.querySelectorAll('button');
    toA.addEventListener('click', () => selectTrack('a', track.id));
    toB.addEventListener('click', () => selectTrack('b', track.id));
    grid.appendChild(card);
  });
  updateSelection();
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '00:00';
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
}

function renderWaveform(peaks) {
  const wave = $('#mix-wave');
  wave.replaceChildren(...peaks.map((peak) => { const bar = document.createElement('i'); bar.style.height = `${Math.max(5, peak * 100)}%`; return bar; }));
  $('#empty-wave').hidden = true; $('#playhead').style.opacity = '1';
}

function updatePlaybackDisplay() {
  const audio = $('#mix-audio');
  const progress = audio.duration ? Math.min(audio.currentTime / audio.duration, 1) : 0;
  const gainA = Math.cos(progress * Math.PI / 2); const gainB = Math.sin(progress * Math.PI / 2);
  $('#mix-clock').textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
  $('#playhead').style.left = `calc(15px + (100% - 30px) * ${progress})`;
  $('#crossfade-fill').style.width = `${progress * 100}%`; $('#crossfade-knob').style.left = `${progress * 100}%`;
  $('#gain-a').textContent = `${Math.round(gainA * 100)}%`; $('#gain-b').textContent = `${Math.round(gainB * 100)}%`;
  $('#cover-a').style.opacity = String(.42 + gainA * .58); $('#cover-b').style.opacity = String(.42 + gainB * .58);
  const bars = [...document.querySelectorAll('#mix-wave i')];
  bars.forEach((bar, index) => bar.classList.toggle('passed', index / bars.length <= progress));
  if (!audio.paused && !audio.ended) app.animationFrame = requestAnimationFrame(updatePlaybackDisplay);
}

function setPlaying(playing) {
  $('#play-button').textContent = playing ? '❚❚' : '▶';
  $('#cover-a').classList.toggle('playing', playing); $('#cover-b').classList.toggle('playing', playing);
  $('#render-state').innerHTML = `<i></i>${playing ? 'MIXING LIVE' : 'MIX READY'}`;
}

async function generateMix() {
  const button = $('#mix-button'); button.disabled = true;
  button.querySelector('span').textContent = 'Building your transition…';
  button.querySelector('small').textContent = 'Aligning beats and rendering audio';
  $('#render-state').innerHTML = '<i></i>RENDERING';
  try {
    const result = await post('/api/mix', {source_id: app.sourceId, destination_id: app.destinationId, technique: $('#technique').value, bars: Number($('#bars').value)});
    app.mix = result; renderWaveform(result.waveform);
    $('#visual-title-a').textContent = result.source.title.toUpperCase(); $('#visual-title-b').textContent = result.destination.title.toUpperCase();
    $('#transport-title').textContent = `${result.source.title} → ${result.destination.title}`;
    $('#transport-detail').textContent = $('#technique').selectedOptions[0].textContent;
    $('#metric-bpm').textContent = result.plan.target_bpm.toFixed(1); $('#metric-length').textContent = `${result.metrics.duration_seconds.toFixed(1)}s`; $('#metric-peak').textContent = `${result.metrics.peak_dbfs.toFixed(1)} dB`;
    const audio = $('#mix-audio'); audio.src = `${result.url}?v=${Date.now()}`; audio.load();
    $('#play-button').disabled = false; $('#render-state').innerHTML = '<i></i>MIX READY';
    $('#performance').scrollIntoView({behavior: 'smooth', block: 'center'});
    toast('Your mix is ready', `${result.plan.transition_bars} bars at ${result.plan.target_bpm.toFixed(1)} BPM.`);
    audio.play().catch(() => {});
  } catch (error) {
    $('#render-state').innerHTML = '<i></i>COULD NOT RENDER'; toast('Mix failed', error.message);
  } finally {
    button.querySelector('span').textContent = 'Generate my mix'; button.querySelector('small').textContent = 'AI analyzes and renders the transition'; updateSelection();
  }
}

$('#swap-button').addEventListener('click', () => { [app.sourceId, app.destinationId] = [app.destinationId, app.sourceId]; updateSelection(); });
$('#mix-button').addEventListener('click', generateMix);
$('#play-button').addEventListener('click', () => { const audio = $('#mix-audio'); if (audio.paused) audio.play(); else audio.pause(); });
const audio = $('#mix-audio');
audio.addEventListener('play', () => { setPlaying(true); cancelAnimationFrame(app.animationFrame); updatePlaybackDisplay(); });
audio.addEventListener('pause', () => { setPlaying(false); cancelAnimationFrame(app.animationFrame); updatePlaybackDisplay(); });
audio.addEventListener('ended', () => { setPlaying(false); updatePlaybackDisplay(); });
audio.addEventListener('loadedmetadata', updatePlaybackDisplay);

fetch('/api/status').then((response) => response.json()).then(({tracks, engine}) => {
  app.tracks = tracks; app.sourceId = tracks[0]?.id || null; app.destinationId = tracks[1]?.id || null;
  $('#engine-label').textContent = `${engine.model} · online`; renderLibrary();
}).catch((error) => { $('#engine-label').textContent = 'Engine offline'; toast('Cannot reach the audio engine', error.message); });
