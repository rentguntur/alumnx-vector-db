# NexVec

NexVec is a Phase 1 FastAPI service for ingesting PDF documents, chunking them, generating embeddings, storing them in JSONL files, and retrieving relevant chunks with KNN search.

## Requirements

- Python 3.12+
- A valid `GOOGLE_API_KEY`
- A local virtual environment at `env/` or another Python environment with the project dependencies installed

## Project Layout

```text
nexvec/
  app/
  tests/
  config.yaml
  .env
  requirements.txt
  main.py
```

## Setup

1. Open a terminal in the `nexvec/` directory.
2. Create or activate your virtual environment.

On Windows PowerShell:

```powershell
.\env\Scripts\python.exe --version
```

If you need to create the virtual environment:

```powershell
python -m venv env
```

3. Install dependencies:

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure `.env`

Create a file named `.env` in the `nexvec/` directory.

Example:

```env
GOOGLE_API_KEY=your_api_key_here
```

The app loads `.env` automatically on startup.

Important:
- Do not commit real secrets to git.
- The repository already ignores `.env`.

## Run the API

Start the server on port `8009`:

```powershell
.\env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8009
```

Then open:

- http://localhost:8009/docs
- http://localhost:8009/redoc

## Run Tests

Run the test suite from the `nexvec/` directory:

```powershell
.\env\Scripts\python.exe -m pytest -q
```

If `GOOGLE_API_KEY` is missing, the suite is configured to skip.

## Endpoints

- `GET /chunking-strategies`
- `GET /retrieval-strategies`
- `GET /knowledgebases`
- `POST /ingest`
- `POST /retrieve`

## Notes

- The vector store is created lazily when ingest or retrieve first touches storage.
- `ann` is not supported in Phase 1 and returns HTTP 400.
- `excludevectors=true` can be sent to `/retrieve` to omit `embedding_vector` from the response.

## Project Objective

This project aims to build a vector database system where data is converted into embeddings and stored efficiently for similarity search.

## What We Are Trying to Achieve

- Convert raw data into chunks
- Generate embeddings for each chunk
- Store embeddings as vectors
- Separate metadata and store it in MySQL
- Enable efficient search using vector similarity

## Features

- Chunking of data
- Embedding generation
- Vector storage
- Metadata storage in MySQL
- Fast retrieval system

## My Contribution

- Updated README documentation
- Understood project workflow
- Tested project setup
