# Installation Guide

This guide matches the current repository setup scripts and dependency files.

## Backend Setup

1. Create a Python virtual environment in the project root.
2. Install the dependencies from `requirements.txt`.
3. Run `install_models.py` if you need the downloadable analyzer models.
4. Start the FastAPI application from `main.py`.

### Windows Commands

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe install_models.py
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --env-file .env
```

Keep `.env` local. It contains machine-specific analyzer paths and model settings;
`.env.example` is the safe template committed to the repository.

## Frontend Setup

1. Change into the `frontend/` directory.
2. Install the Node dependencies.
3. Run the Vite development server.

### Windows Commands

```powershell
cd frontend
npm install
npm run dev
```

## Optional Integrations

The repository also includes `optional_requirements.txt` for optional research tools.

- `transformers`
- `torch`
- `ufal.udpipe`

## Notes

- Java-backed tools such as AlKhalil and MADAMIRA require local runtime/model files.
- The repository uses a local `.env` file for configuration when needed.
