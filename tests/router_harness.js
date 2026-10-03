// Minimal DOM stub to execute nlp-integration.js boot path in Node.
const fs = require('fs');

function makeEl(extra) {
  return Object.assign({
    style: {}, dataset: {}, textContent: '', innerHTML: '',
    children: [], value: '', checked: false, files: [],
    setAttribute() {}, removeAttribute() {},
    appendChild(c) { this.children.push(c); return c; },
    insertBefore(c) { this.children.unshift(c); return c; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
    closest() { return null; },
    scrollIntoView() {},
    parentElement: null,
  }, extra || {});
}

function makeMain(kind) {
  const m = makeEl();
  const inner = {};
  if (kind === 'analyzer') inner['#marathi-input-area'] = makeEl();
  if (kind === 'results') inner['#tab-pane-original'] = makeEl();
  if (kind === 'dataset') inner['#dataset'] = makeEl();
  m.querySelector = (sel) => inner[sel] || null;
  return m;
}

const mains = [makeMain('home'), makeMain('analyzer'), makeMain('results'), makeMain('dataset')];
const asides = [makeEl(), makeEl(), makeEl(), makeEl()];
const headers = [makeEl(), makeEl(), makeEl(), makeEl()];
function navLink(href) {
  return { getAttribute: (k) => (k === 'href' ? href : null),
           setAttribute() {}, removeAttribute() {}, style: {}, textContent: '' };
}
const navLinks = ['#/home', '#/analyzer', '#/results', '#/dataset', '#pipeline', '#search'].map(navLink);

const listeners = {};
global.document = {
  querySelectorAll(sel) {
    if (sel === 'main') return mains;
    if (sel === 'aside') return asides;
    if (sel.indexOf('header') >= 0) return headers;
    if (sel === 'aside nav a') return navLinks;
    if (sel === '[data-backend-status]') return [];
    return [];
  },
  querySelector() { return null; },
  getElementById() { return null; },
  addEventListener(ev, fn) { listeners[ev] = fn; },
  createElement() { return makeEl(); },
  title: '',
};
global.window = {
  location: { port: '', origin: 'http://localhost:3000', hash: '' },
  localStorage: { getItem: () => null },
  addEventListener() {}, scrollTo() {},
};
global.history = { replaceState() {} };
global.fetch = () => Promise.reject(new Error('offline'));
global.navigator = {};

const target = process.argv[2] ||
  'D:\\HARSH\\projects\\Timepass\\NLP\\frontend\\nlp-integration.js';
const src = fs.readFileSync(target, 'utf8');
eval(src);

(async () => {
  listeners['DOMContentLoaded']();
  await new Promise((r) => setTimeout(r, 50));
  const vis = mains.map((m) => m.style.display !== 'none');
  console.log('mains visible (expect true,false,false,false):', JSON.stringify(vis));
  console.log('asides hidden (expect 3):', asides.slice(1).filter((a) => a.style.display === 'none').length);
  console.log('headers hidden (expect 3):', headers.slice(1).filter((h) => h.style.display === 'none').length);
  console.log('title:', document.title);
  if (JSON.stringify(vis) !== '[true,false,false,false]') throw new Error('router initial state wrong');
  console.log('ROUTER HARNESS OK');
})().catch((e) => { console.error('HARNESS FAIL:', e); process.exit(1); });
