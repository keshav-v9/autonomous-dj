const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function makeWave(selector, seed) {
  const node = $(selector);
  for (let i = 0; i < 120; i += 1) {
    const bar = document.createElement('i');
    const height = 5 + Math.abs(Math.sin((i + seed) * .37) * Math.cos((i + seed) * .11)) * 29;
    bar.style.height = `${height}px`;
    node.appendChild(bar);
  }
}
makeWave('#wave-a', 4);
makeWave('#wave-b', 13);

let seconds = 24;
setInterval(() => {
  seconds = seconds > 0 ? seconds - 1 : 24;
  $('#countdown').textContent = `00:${String(seconds).padStart(2, '0')}`;
}, 1000);

function toast(title, message) {
  const node = $('#toast');
  node.querySelector('b').textContent = title;
  node.querySelector('span').textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 3600);
}

async function post(url, body = {}) {
  const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}

$$('.control-strip button[data-action]').forEach((button) => {
  button.addEventListener('click', async () => {
    $$('.control-strip button').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    try {
      const result = await post('/api/control', {action: button.dataset.action});
      $('#energy-value').textContent = result.state.energy_target.toFixed(1);
      $('#energy-dial').style.background = `conic-gradient(var(--lime) 0 ${result.state.energy_target * 36}deg,#242927 ${result.state.energy_target * 36}deg 360deg)`;
      if (button.dataset.action === 'skip') {
        $('#current-title').textContent = result.state.current_track;
        $('#next-title').textContent = result.state.next_track;
      }
      toast('Direction registered', button.querySelector('span').innerText.replace('\n', ' · '));
    } catch (error) { toast('Engine unavailable', error.message); }
  });
});

const techniques = ['BASS SWAP', 'EQ BLEND', 'QUICK CUT'];
$('#technique-button').addEventListener('click', () => {
  const button = $('#technique-button');
  const current = techniques.findIndex((name) => button.textContent.includes(name));
  button.innerHTML = `${techniques[(current + 1) % techniques.length]} <span>⌄</span>`;
});

$('#request-open').addEventListener('click', () => $('#request-dialog').showModal());
$('#request-form').addEventListener('submit', async (event) => {
  if (event.submitter?.value === 'cancel') return;
  event.preventDefault();
  const value = $('#request-input').value.trim();
  if (!value) return;
  try {
    await post('/api/request', {title: value});
    $('#request-dialog').close();
    $('#request-input').value = '';
    toast('Request received', `${value} is now in the planner.`);
  } catch (error) { toast('Could not add request', error.message); }
});

$('#render-button').addEventListener('click', async () => {
  const button = $('#render-button');
  const technique = $('#technique-button').textContent.trim().toLowerCase().replace(' ', '_');
  button.classList.add('loading');
  button.querySelector('span').textContent = 'RENDERING AUDIO…';
  try {
    const result = await post('/api/demo/render', {technique, bars: 16});
    const audio = new Audio(`${result.url}?t=${Date.now()}`);
    audio.play();
    button.querySelector('span').textContent = 'PLAYING TRANSITION';
    toast('Transition ready', `${result.metrics.duration_seconds}s · ${result.metrics.peak_dbfs} dBFS · zero clipped samples`);
    audio.addEventListener('ended', () => { button.querySelector('span').textContent = 'RENDER TRANSITION'; });
  } catch (error) {
    toast('Render failed', error.message);
    button.querySelector('span').textContent = 'RENDER TRANSITION';
  } finally { button.classList.remove('loading'); }
});

fetch('/api/status').then((response) => response.json()).then(({state}) => {
  $('#energy-value').textContent = state.energy_target.toFixed(1);
}).catch(() => toast('Offline preview', 'Start the Python server to enable the audio engine.'));

