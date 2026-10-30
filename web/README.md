# KTA web

Read-only frontend (WP9a). Design: [DESIGN.md](DESIGN.md).

```bash
cd web
npm ci            # lockfile only (conventions §7); Node version in .nvmrc
npm run dev       # http://127.0.0.1:5173, proxies /api to the KTA API
npm run build     # type-check + production build into web/dist (not committed)
```

The API must be running (from the repository root):

```bash
KTA_ENV=dev KTA_WP=WP9a-fixture KTA_DATA=~/kta-data scripts/api
```

`scripts/seed-dev` fills `$KTA_DATA/dev/WP9a-fixture/` with an origin=fixture toy world (D168);
the UI shows a FIXTURE DATA banner whenever the store holds such records. `?static=1` in the URL
turns off live updates (printing, screenshots).
