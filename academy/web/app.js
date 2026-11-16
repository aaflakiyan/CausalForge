// Lab 01 front end. This displays state returned by the *actual MuJoCo server*.
const $ = (id) => document.getElementById(id);
const controls = ['x','y','gain','noise'];
let running = false;
let stepping = false;
let connected = false;
const mapX = x => 330 + x*355;
const mapY = y => 250 - y*355;
const loc = (p) => `${mapX(p[0])},${mapY(p[1])}`;

async function api(route, body) {
  const options = body === undefined ? {} : { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) };
  const response = await fetch(route, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
  return data;
}
function message(text) { $('message').textContent = text || ''; }
function draw(s) {
  $('arm1').setAttribute('d', `M${loc([0,0])}L${loc(s.elbow_xy_m)}`);
  $('arm2').setAttribute('d', `M${loc(s.elbow_xy_m)}L${loc(s.tip_xy_m)}`);
  $('elbow').setAttribute('cx', mapX(s.elbow_xy_m[0])); $('elbow').setAttribute('cy', mapY(s.elbow_xy_m[1]));
  $('tip').setAttribute('cx', mapX(s.tip_xy_m[0])); $('tip').setAttribute('cy', mapY(s.tip_xy_m[1]));
  $('target').setAttribute('cx', mapX(s.target_xy_m[0])); $('target').setAttribute('cy', mapY(s.target_xy_m[1]));
  $('cup-label').setAttribute('x', Math.min(510,mapX(s.target_xy_m[0])+25));
  $('cup-label').setAttribute('y', mapY(s.target_xy_m[1])-25);
  $('error-line').setAttribute('d', `M${loc(s.tip_xy_m)}L${loc(s.target_xy_m)}`);
  $('error').textContent = `${(s.error_m*100).toFixed(1)} cm`;
  $('time').textContent = `${s.time_s.toFixed(2)} s`;
  $('outcome').textContent = s.success ? '✓ Reached' : 'Approaching';
  let h=s.history||[], max=Math.max(0.07,...h.map(p=>p.error));
  const points=h.map((p,i)=>`${12+i*476/Math.max(1,h.length-1)},${106-94*p.error/max}`).join(' ');
  $('trace').setAttribute('d',points ? `M${points.replaceAll(' ',' L')}` : '');
}
function updateLabels() {
  $('x-val').value = `${Number($('x').value).toFixed(2)} m`;
  $('y-val').value = `${Number($('y').value).toFixed(2)} m`;
  $('gain-val').value = Number($('gain').value).toFixed(1);
  $('noise-val').value = `${Number($('noise').value).toFixed(3)} m`;
}
async function configure() {
  updateLabels();
  try {
    const s = await api('/api/config', {x:+$('x').value, y:+$('y').value, gain:+$('gain').value, noise:+$('noise').value});
    draw(s); message('');
  } catch(err) { message(err.message); }
}
async function reset() {
  running = false; $('play').textContent = '▶ Run physics';
  try { draw(await api('/api/reset',{})); message(''); } catch(err) { message(err.message); }
}
async function stepOnce() {
  if(stepping) return;
  stepping=true;
  try { draw(await api('/api/step',{steps:10})); message(''); }
  catch(err) { running=false; $('play').textContent='▶ Run physics'; message(err.message); }
  finally { stepping=false; }
}
function animate() {
  if(!running) return;
  stepOnce().finally(()=>{if(running) setTimeout(animate,50);});
}
$('play').addEventListener('click',()=>{
  if(!connected) return;
  running=!running; $('play').textContent = running ? 'Ⅱ Pause' : '▶ Run physics';
  if(running) animate();
});
$('step').addEventListener('click',stepOnce);
$('reset').addEventListener('click',reset);
for (const id of controls) $(id).addEventListener('input',configure);
updateLabels();
api('/api/state').then(s=>{connected=true;$('engine-status').textContent='● MuJoCo ready';draw(s);}).catch(e=>{
  $('engine-status').textContent='Engine unavailable'; message(`Start the local server and install the simulator. ${e.message}`);
});
