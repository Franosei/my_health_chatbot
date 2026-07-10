# FlynnMed React Frontend

Mobile-first React client for the production branch.

```powershell
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so start the FastAPI backend from the repository root:

```powershell
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

For deployment, build the client:

```powershell
npm run build
```

FastAPI serves `frontend/dist` automatically when the build folder exists.
