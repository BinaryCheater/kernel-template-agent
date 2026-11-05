// DSH v4 stores concatenated Zstandard frames; decode every frame, not only header.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const root = process.argv[2];
for (const rel of fs.readdirSync(root, {recursive: true}).filter(x => x.endsWith('.zstd'))) {
  const file = path.join(root, rel), data = fs.readFileSync(file), parts = [];
  let offset = 0;
  while (offset < data.length) {
    const frame = zlib.zstdDecompressSync(data.subarray(offset), {info: true});
    if (!frame.engine.bytesWritten) throw new Error('Zstandard frame consumed no bytes');
    parts.push(frame.buffer);
    offset += frame.engine.bytesWritten;
  }
  const raw = Buffer.concat(parts);
  fs.writeFileSync(file + '.expanded.jsonl', raw, {mode: 0o600});
  const rows = raw.toString().split('\n').filter(Boolean).map(JSON.parse);
  console.log(JSON.stringify({file, session: rows[0].id,
    settings: rows.filter(r => r.type === 'request/header').map(r => r.data.header.config),
    usage: rows.filter(r => r.type === 'assistant/message').map(r => r.data.usage)}));
}
