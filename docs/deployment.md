# Deployment

ANATO3D is a **single-user research prototype**. The guidance below is about
running it reproducibly; it is *not* a hardened production deployment.

> There is **no authentication** anywhere in this project. Do not expose it to an
> untrusted network or the public internet. Run it on `localhost` or inside a
> trusted network only.

## 1. Requirements

- Python ≥ 3.10
- Node.js ≥ 18 (only to build the frontend)
- ~2 GB RAM for typical CT series (more for large volumes)
- No GPU required

## 2. Local development

Two processes:

```bash
# terminal 1 — backend
cd backend
python -m pip install -r requirements.txt
python run.py                       # http://127.0.0.1:8000

# terminal 2 — frontend (hot reload)
cd frontend
npm install
npm run dev                         # http://127.0.0.1:5173
```

The dev server proxies `/api` to `http://127.0.0.1:8000`; override with
`VITE_BACKEND_URL`.

## 3. Single-process (static build)

```bash
cd frontend
npm run build                       # emits frontend/dist
cd ../backend
python run.py
```

`app/main.py` auto-mounts `frontend/dist` at `/` when it exists, so one process
serves the SPA and the API on port 8000. Open <http://127.0.0.1:8000>.

## 4. Docker Compose

```bash
docker compose up --build
```

| Service | Port | Notes |
| --- | --- | --- |
| `backend` | 8000 | FastAPI; data in the `anato3d-work` volume |
| `frontend` | 5173 | nginx serving the built SPA; proxies `/api` to `backend` |

The backend image is CPU-only and installs `requirements.txt` (the optional
MONAI/torch path is not installed). To enable AI segmentation in Docker, extend
`backend/Dockerfile` with your model dependencies and mount a bundle, then set
`ANATO3D_AI_MODEL_PATH`.

### Volumes and persistence

`ANATO3D_DATA_DIR` (`_work` by default) holds **all** uploads and derived
artefacts. In compose it is a named volume (`anato3d-work`). Removing the volume
removes every dataset — which is the intended "reset".

## 5. Configuration in production

Set environment variables (or a `backend/.env`):

```ini
ANATO3D_DATA_DIR=/data
ANATO3D_DEBUG=false
ANATO3D_MAX_UPLOAD_BYTES=2147483648
ANATO3D_WORKSPACE_TTL_MINUTES=240
ANATO3D_CORS_ORIGINS=["https://your.host"]
```

See the README configuration table for the full list.

## 6. Reverse proxy

If you place ANATO3D behind nginx/Caddy:

- Proxy `/` and `/api` to the backend (port 8000).
- Allow **large request bodies** (`client_max_body_size`) — uploads can be
  hundreds of MB.
- Increase proxy read timeouts for large reconstructions (heavy operations are
  still fast, but large uploads are not).
- Terminate TLS at the proxy. ANATO3D itself speaks plain HTTP.
- Prefer `wait=false` + polling for long operations so no single request is held
  open for the whole job (the frontend already does this).

Example nginx location:

```nginx
location / {
    proxy_pass http://backend:8000;
    proxy_set_header Host $host;
    client_max_body_size 4g;
    proxy_read_timeout 600s;
}
```

## 7. Resource sizing

| Knob | Effect |
| --- | --- |
| `max_upload_bytes` / `max_file_bytes` | Cap upload size |
| `max_upload_files` | Cap file count per request |
| `max_voxels_soft` | Above this, downsample before segmentation |
| `max_voxels_hard` | Refuse above this |
| `workspace_ttl_minutes` | Auto-delete idle workspaces |
| `JobManager(max_workers=4)` | Thread-pool size for background jobs |

Disk usage is dominated by temporary workspaces; the reaper keeps it bounded. If
you need long-term storage of results, use the **export bundle** and archive it
yourself.

## 8. Health and readiness

- `GET /api/health` returns liveness, configured limits, storage stats, and AI
  availability. Use it for readiness probes.
- `GET /api/jobs` lists recent jobs for debugging.

## 9. Security checklist

- [ ] Keep ANATO3D on a trusted network / localhost; there is no auth.
- [ ] Terminate TLS upstream.
- [ ] Use de-identified DICOM only.
- [ ] Set a non-default `ANATO3D_DATA_DIR` on a volume you can wipe.
- [ ] Review `max_upload_*` limits for your environment.
- [ ] Do not mount the data dir into any other container or share.

## 10. Upgrading

The application is stateless apart from `ANATO3D_DATA_DIR`. To upgrade:

1. `git pull` / replace the source.
2. Rebuild: `npm run build` and/or `docker compose build`.
3. Restart. Old workspaces are disposable; there is no database migration
   because there is no database.

## 11. Uninstalling / wiping

```bash
# local
rm -rf backend/_work

# docker
docker compose down -v
```

## 12. Troubleshooting (deployment)

| Symptom | Fix |
| --- | --- |
| Mixed content / CORS errors | Add the frontend origin to `ANATO3D_CORS_ORIGINS`, or serve both from one origin (§3). |
| `413` through a proxy | Raise `client_max_body_size`. |
| Uploads stall | Raise proxy read timeout; prefer background jobs + polling. |
| Out of disk | Lower `workspace_ttl_minutes`; the reaper reclaims space. |
| AI engine unavailable | Expected unless MONAI/torch are installed and a bundle path is set. |
