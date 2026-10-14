import {bundle} from '@remotion/bundler';
import {openBrowser, selectComposition, renderMedia, makeCancelSignal} from '@remotion/renderer';
import {readFile, mkdir, realpath} from 'node:fs/promises';
import {dirname, resolve, relative, isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
process.chdir(root);
const event = (data) => process.stdout.write(JSON.stringify(data) + '\n');
const within = (parent, child) => {
  const rel = relative(parent, child);
  return rel !== '' && !rel.startsWith('..') && !isAbsolute(rel);
};
const compile = (outDir) => bundle({entryPoint: resolve(root, 'src/index.tsx'),
  outDir, publicDir: null, enableCaching: true,
  onProgress: (progress) => {if (progress % 10 === 0) event({event: 'bundle', progress});}});

let browser;
const {cancelSignal, cancel} = makeCancelSignal();
process.on('SIGINT', cancel);
process.on('SIGTERM', cancel);
try {
  if (process.argv[2] === '--bundle') {
    const output = await compile(resolve(root, '.cache/bundle'));
    event({event: 'compiled', output});
  } else {
    const request = JSON.parse(await readFile(process.argv[2], 'utf8'));
    const directory = await realpath(request.directory);
    const clips = resolve(directory, 'clips');
    await mkdir(clips, {recursive: true});
    if (!within(directory, await realpath(clips))) throw new Error('动画输出目录越界');
    const serveUrl = await compile(resolve(directory, '.bundle'));
    browser = await openBrowser('chrome', {browserExecutable: request.browser,
      chromeMode: 'headless-shell', chromiumOptions: {gl: 'swangle'}, logLevel: 'error'});
    const fontData = 'data:font/' + (request.font.toLowerCase().endsWith('.otf') ? 'otf' : 'ttf')
      + ';base64,' + (await readFile(request.font)).toString('base64');
    for (let index = 0; index < request.jobs.length; index++) {
      const job = request.jobs[index];
      const output = resolve(clips, job.key + '.tmp.mp4');
      if (!/^[a-f0-9]{64}$/.test(job.key) || !within(clips, output)) throw new Error('动画缓存编号无效');
      const inputProps = {...job.props, fontData};
      const composition = await selectComposition({serveUrl, id: 'Explanation', inputProps,
        puppeteerInstance: browser, browserExecutable: request.browser, logLevel: 'error'});
      let last = -1;
      await renderMedia({serveUrl, composition, inputProps, outputLocation: output,
        codec: 'h264', pixelFormat: 'yuv420p', crf: 18, x264Preset: 'fast',
        muted: true, concurrency: request.concurrency, puppeteerInstance: browser,
        browserExecutable: request.browser, logLevel: 'error', cancelSignal,
        onProgress: ({progress}) => {
          const percent = Math.floor(progress * 20) * 5;
          if (percent !== last) {last = percent; event({event: 'progress', id: job.id,
            index, count: request.jobs.length, progress});}
        }});
      event({event: 'complete', id: job.id, key: job.key});
    }
  }
} catch (error) {
  event({event: 'error', message: error instanceof Error ? error.message : String(error)});
  process.exitCode = 1;
} finally {
  if (browser) await browser.close({silent: true});
}
