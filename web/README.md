# KTA web

Read-only frontend (WP9a). Design: [DESIGN.md](DESIGN.md).

```bash
cd web
npm ci            # lockfile only (conventions §7); Node version in .nvmrc
npm run dev       # http://127.0.0.1:5173
npm run build     # type-check + production build into web/dist (not committed)
```

Phase A renders a hand-written sample dataset (`src/sample/`, origin = fixture, SAMPLE DATA
banner on every page, D95). Phase B switches `src/data/source.ts` to the KTA API and deletes the
sample.
