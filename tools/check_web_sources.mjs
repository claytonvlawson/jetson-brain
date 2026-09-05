// tools/check_web_sources.mjs - headless check of brain_web's clickable sources.
//
// Drives Chromium at an iPhone 14 viewport: asks a question, asserts every .md
// source is an obsidian:// link with a "view" button, opens the first note
// sheet, checks title/tags/headings, closes it, and screenshots both states.
// This is the check that caught the hidden-sheet-swallows-taps bug (b732363).
//
// Playwright is not installed in this repo; borrow it from a tree that has it:
//   cp tools/check_web_sources.mjs ~/soulsilver-3d/.check_tmp.mjs
//   cd ~/soulsilver-3d && S=/tmp URL=http://127.0.0.1:8897/ node .check_tmp.mjs; rm .check_tmp.mjs
// with brain_web running from the Mac venv (see HANDOFF "Run brain_web on the Mac").
import { chromium, devices } from 'playwright';
const S = process.env.S, URL = process.env.URL || 'http://127.0.0.1:8897/';
const browser = await chromium.launch();
const ctx = await browser.newContext({ ...devices['iPhone 14'] });
const page = await ctx.newPage();
const errors = []; page.on('pageerror', e => errors.push(String(e)));
await page.goto(URL, { waitUntil: 'load' });
await page.fill('#q', 'What does Humphries say about teaching your child to sign if you do not sign?');
await page.click('#send');
await page.waitForSelector('.a .srcs', { timeout: 180000 });
await page.screenshot({ path: `${S}/sources_list.png`, fullPage: false });
const links = await page.$$eval('.a .srcs .srow a.src', as => as.map(a => ({ text: a.textContent, href: a.getAttribute('href') })));
const nView = await page.$$eval('.a .srcs .view', b => b.length);
await page.click('.a .srcs .view');
await page.waitForFunction(() => { const b = document.querySelector('#nbody'); return b && b.textContent && !/loading/.test(b.textContent); }, null, { timeout: 30000 });
await page.waitForTimeout(300);
await page.screenshot({ path: `${S}/note_sheet.png`, fullPage: false });
const sheet = await page.evaluate(() => ({
  hidden: document.getElementById('note').hidden,
  title: document.getElementById('ntitle').textContent,
  path: document.getElementById('npath').textContent,
  tags: [...document.querySelectorAll('#ntags .tag')].map(t => t.textContent),
  obs: document.querySelector('#nbody .obs') && document.querySelector('#nbody .obs').getAttribute('href'),
  headings: [...document.querySelectorAll('#nbody h1,#nbody h2,#nbody h3')].slice(0,5).map(h => h.textContent),
  bodyChars: document.getElementById('nbody').textContent.length,
  scriptTags: document.querySelectorAll('#nbody script').length,
}));
await page.click('#nclose');
const closed = await page.evaluate(() => document.getElementById('note').hidden && document.getElementById('shade').hidden);
const answerStill = await page.$$eval('.a', a => a.length);
console.log(JSON.stringify({ links, nView, sheet, closed, answerStill, errors }, null, 2));
await browser.close();
