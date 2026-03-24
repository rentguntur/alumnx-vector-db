# Alumnx Vector DB - Local Setup & Usage Guide

This guide provides instructions for setting up and running the Alumnx Vector DB locally for development and testing.

## Prerequisites
- **Docker** and **Docker Compose** installed.
- **Python 3.12+** (if running without Docker).
- **uv** (recommended for Python dependency management).

---

## 🚀 Running Locally with Docker (Recommended)

The easiest way to get started is using Docker Compose. This ensures all system dependencies (like NLTK and PDF libraries) are correctly configured.

### 1. Build and Start
Run the following command in the project root:
```bash
docker-compose up --build
```

### 2. Access the API
- **Swagger Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

### 3. Persistent Storage
The Vector Store is persisted in the `./vector_store` directory on your host machine.

---

## 🐍 Running Locally with Python (Native)

If you prefer to run the application directly on your machine:

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment and Sync Dependencies
```bash
uv venv
uv sync
```

### 3. Start the Application
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ingest/pdf` | Upload and process a PDF file into the vector store. |
| `GET` | `/chunking/strategies` | List available text chunking strategies. |
| `GET` | `/status` | Check the health status of the API. |

---

## 🛠 Deployment & CI/CD (Production)

The production environment is hosted on **Edge Production (EC2)** and managed via **PM2**.

- **Endpoint**: [http://13.126.130.56:8001/docs](http://13.126.130.56:8001/docs)
- **Automatic Deploys**: Any push to the `main` branch triggers a GitHub Action that deploys the latest code to the server.
- **Manual Restart**: 
  ```bash
  pm2 restart alumnx-vector-db
  ```
- **View Logs**:
  ```bash
  pm2 logs alumnx-vector-db
  ```

---

## 🐳 Docker Installation Guide

### For Mac

1. Go to: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Click **Download for Mac** (Apple Silicon / Intel)
3. Open the `.dmg` file
4. Drag Docker to the **Applications** folder
5. Launch Docker Desktop from Applications
6. Accept the terms & complete setup
7. Verify installation:
   ```bash
   docker --version
   ```

### For Windows

1. Go to: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Click **Download for Windows**
3. Run the installer (`.exe`)
4. Enable WSL 2 when prompted
5. Restart your PC
6. Launch Docker Desktop
7. Verify installation:
   ```bash
   docker --version
   ```

### References

- [Docker Official Website](https://www.docker.com)
- [Docker Desktop Windows Installation Guide](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker Desktop Mac Installation Guide](https://docs.docker.com/desktop/setup/install/mac-install/)

---

## 📞 Support

For any issues, contact the Alumnx engineering team.
