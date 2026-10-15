import {ensureBrowser} from '@remotion/renderer';
import {createRequire} from 'node:module';
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname, resolve, relative, isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
process.chdir(root);
await ensureBrowser({chromeMode: 'headless-shell', logLevel: 'info',
  browserExecutable: process.env.NARRATO_BROWSER_BINARY || undefined});
// Pinned Remotion version; its installer exposes the downloaded path here.
const require = createRequire(import.meta.url);
const {getRevisionInfo} = require(resolve(root, 'node_modules/@remotion/renderer/dist/browser/BrowserFetcher.js'));
const browser = process.env.NARRATO_BROWSER_BINARY || getRevisionInfo('headless-shell').executablePath;
const relativeBrowser = relative(root, resolve(browser));
const savedBrowser = relativeBrowser.startsWith('..') || isAbsolute(relativeBrowser) ? resolve(browser) : relativeBrowser;
await mkdir(resolve(root, '.cache'), {recursive: true});
await writeFile(resolve(root, '.cache/runtime.json'), JSON.stringify({browser: savedBrowser, node: process.execPath}, null, 2));
console.log(JSON.stringify({browser, node: process.execPath}));
