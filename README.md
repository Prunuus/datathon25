# Datathon25
## Repository structure

Top-level layout:

- `frontend/` — Vite + React application. Use Node.js to run, build and lint.
- `backend/` — Backend Python code. `main.py` is present but currently empty.

Files you’ll see right away:

- `README.md` — This file.
- `frontend/package.json` — Frontend scripts and dependencies.

## Prerequisites

- Node.js (LTS) and npm or yarn. Tested with Node 18+.
- (Optional) Python 3.8+ if you add or run the backend.

## Frontend

The frontend lives in the `frontend/` folder and uses Vite.

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the dev server:

```bash
npm run dev
```

This runs Vite (see `frontend/package.json`) and will serve the app (by default on `http://localhost:5173`).


## Backend

1. Create a virtual environment and install dependencies and run the backend server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi
fastapi dev <filename>
```