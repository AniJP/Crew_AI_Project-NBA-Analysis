---
title: NBA Analysis
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# 🏀 NBA Data Analysis with CrewAI

Multi-agent NBA Q&A and analysis system. Ask natural-language basketball questions; an analyst agent calls **structured JSON tools**, returns **citations + tool traces**, and is scored by an offline **eval suite**.

## ✨ Features

- **NBA Insights API** (`POST /ask`): answer + citations + tool trace + latency + groundedness
- **Structured tools**: search / summary / pandas analysis return JSON facts (not raw CSV dumps)
- **Reliability**: in-memory TTL cache, tool-call budget, kickoff retries, groundedness gate (refuse ungrounded numbers)
- **Evals**: golden set with faithfulness, tool-call success, expected-tool hit, latency
- **Multi-agent CrewAI**: Engineer / Analyst / Storyteller (Gradio + CLI still supported)
- **OpenAI** (default `gpt-4o`) + optional Docker

## 🏗️ Architecture

```text
POST /ask { question }
        │
        ├─ cache hit? → return cached answer
        ▼
  Analyst crew (tool budget ≤ N)
        │
   structured tools (pandas / search / summary)
        │
   answer + citations + tool_trace
        │
   groundedness check → refuse if score < threshold
```

### Tech Stack

- **CrewAI** + **FastAPI** + **Pandas**
- **ChromaDB** / sentence-transformers (optional semantic search; disabled by default if transformers env breaks)
- **OpenAI**, **Docker** (optional), **Gradio** (optional UI)

## 📋 Prerequisites

- Python 3.11 or 3.12
- pip or uv package manager
- OpenAI API key
- (Optional) Docker for containerized development

## 🚀 Quick start — API

```bash
export OPENAI_API_KEY=your-key
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://localhost:8000/health | python -m json.tool
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who scored the most points in a single game?"}' | python -m json.tool
```

Response fields: `answer`, `citations`, `tool_trace`, `latency_ms`, `cached`, `groundedness`, `refused`.

Cache helpers: `GET /cache/stats`, `POST /cache/clear`.

### Reliability env vars

| Variable | Default | Meaning |
|----------|---------|---------|
| `ASK_CACHE_TTL_S` | `3600` | Response cache TTL (seconds) |
| `ASK_MAX_RETRIES` | `2` | Extra crew kickoff attempts on failure |
| `ASK_MAX_TOOL_CALLS` | `8` | Max tool calls per question |
| `ASK_GROUNDEDNESS_MIN` | `0.5` | Refuse if numeric groundedness score is below this |

## 🧪 Evals

```bash
# No LLM — validates structured tools against gold facts
python evals/run_eval.py --tools-only

# Full agent eval (needs OPENAI_API_KEY)
python evals/run_eval.py --limit 3

# Against a running API
python evals/run_eval_api.py --base-url http://localhost:8000
```

Golden set: `evals/dataset.json` (10 factual questions). Latest sample API run (6 cases): **~83% pass**, mean faithfulness **1.0**, tool-used **100%**, mean tool-call success **~92%**, p50 latency **~2.7s**. Tools-only probe on the expanded 10-case set: **100% pass**.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd NBA_Analysis
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your NBA CSV file in the project directory, or upload it through the web interface.

## 🐳 Docker Setup (Optional)

For a consistent development environment, you can use Docker.

### Prerequisites
- Docker installed on your system
- OpenAI API key (set as environment variable)

### Build the Docker Image

```bash
./docker_build.sh
```

This creates a Docker image named `nba-analysis:latest` with all dependencies installed.

### Run Docker Container

**Option 1: Start Jupyter Notebook**
```bash
export OPENAI_API_KEY=your-api-key-here
./docker_jupyter.sh
```

Jupyter will be available at `http://localhost:8888`. Check the terminal output for the access token.

**Option 2: Get a Bash Shell**
```bash
export OPENAI_API_KEY=your-api-key-here
./docker_bash.sh
```

This gives you an interactive shell inside the container where you can run Python scripts or notebooks.

### Docker Scripts

- `docker_build.sh` - Build the Docker image
- `docker_bash.sh` - Start an interactive bash shell
- `docker_jupyter.sh` - Start Jupyter notebook server

**Note**: The container mounts your current directory, so changes to files are reflected immediately.

## ⚙️ Configuration

### OpenAI Configuration

The application uses **OpenAI** as the LLM provider.

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY=your-openai-api-key
   ```

**Default Model**: `gpt-4o` (can be changed via `OPENAI_MODEL` environment variable)

**Available Models:**
- `gpt-4o` (default, best quality)
- `gpt-4` (high quality)
- `gpt-3.5-turbo` (faster, lower cost)

## 🎮 Usage

### Web Interface (Recommended)

```bash
python app.py
```

Then open your browser to the URL shown (usually `http://localhost:7860`).

**Features:**
- Upload CSV file
- Enter analysis query (or leave blank for comprehensive analysis)
- Click "Analyze Dataset" for full analysis
- Click "Analyze with Question" for quick queries

### Command Line

```bash
python main.py
```

## 📖 Example Queries

- "Who are the top 5 three-point shooters?"
- "Show me the best scoring games this season"
- "Which players have the highest field goal percentage?"
- "Analyze team performance trends"
- "Find games with triple doubles"
- "What are the most efficient shooters?"

## 🛠️ Project Structure

```
NBA_Analysis/
├── api.py                 # FastAPI NBA Insights API (/ask, /health, cache)
├── crewai_utils.py        # Tools, agents, run_ask, cache, groundedness
├── evals/
│   ├── dataset.json       # Golden questions + must_include facts
│   ├── run_eval.py        # Offline / agent eval runner
│   └── run_eval_api.py    # Eval via HTTP against running API
├── crewai.API.md          # API documentation
├── crewai.API.ipynb       # API examples notebook
├── crewai.example.md      # Example documentation
├── crewai.example.ipynb   # Example notebook
├── nba24-25.csv           # Season dataset
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔧 Available Tools

The agents have access to 5 data tools:

1. **read_nba_data**: Read sample rows to understand structure
2. **search_nba_data**: Filter and search CSV data
3. **get_nba_data_summary**: Get comprehensive dataset overview
4. **semantic_search_nba_data**: Natural language semantic search
5. **analyze_nba_data**: Execute pandas operations for advanced analysis

## 🚀 Deployment

### Hugging Face Spaces (Free)

1. **Get API Key:**
   - OpenAI API key: https://platform.openai.com/api-keys

2. **Create Space:**
   - Go to https://huggingface.co/spaces
   - Create new Space with Gradio SDK
   - Push your code

3. **Set Secrets:**
   - Space Settings → Repository secrets
   - Add `OPENAI_API_KEY` = your OpenAI API key
   - (Optional) Add `OPENAI_MODEL` = your preferred model (default: `gpt-4o`)

4. **Deploy:**
   ```bash
   git remote add hf https://huggingface.co/spaces/yourusername/nba-analysis
   git push hf main
   ```

See `EXECUTION_FLOW.md` for detailed deployment instructions.

## 🧪 Local Testing

### Quick Test

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key-here

# Run the app
python app.py
```

## 📊 How It Works

1. **User Input**: Upload CSV + enter query
2. **Crew Creation**: Three agents are initialized with their roles
3. **Parallel Execution**: 
   - Engineer validates data
   - Analyst performs analysis (runs in parallel)
   - Storyteller creates narrative (waits for Analyst)
4. **Tool Execution**: Agents use tools to access and analyze data
5. **LLM Processing**: AI generates insights and responses
6. **Result Aggregation**: All outputs are combined and formatted
7. **Display**: Results shown to user

See `EXECUTION_FLOW.md` for detailed flow documentation.

## 🎯 Key Features Explained

### Semantic Search
Uses vector embeddings to find semantically similar records. First run indexes the CSV, subsequent runs use cached embeddings.

### Parallel Processing
Engineer and Analyst tasks run simultaneously for faster results. Storyteller waits for Analyst to complete.

### Multi-Agent Collaboration
Each agent has a specialized role:
- **Engineer**: Data quality and structure
- **Analyst**: Statistical analysis and insights
- **Storyteller**: Narrative and presentation

## 🔒 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | **Required** |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` |

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'crewai'"
- Install dependencies: `pip install -r requirements.txt` or `uv sync`
- Or use Docker: `./docker_build.sh` then `./docker_bash.sh`

### "OPENAI_API_KEY not set"
- Set your OpenAI API key: `export OPENAI_API_KEY=your-key`
- In Docker: Pass it via `-e OPENAI_API_KEY=your-key` or use `docker_jupyter.sh` which handles it

### "Docker build fails"
- Make sure Docker is installed and running
- Check internet connection (needs to download base image and packages)
- Try: `docker system prune` to free up space

### Slow responses
- OpenAI API calls depend on your internet connection
- Consider using `gpt-3.5-turbo` for faster (but lower quality) responses
- Check OpenAI API status: https://status.openai.com

## 📝 License

This project is open source. Check individual dependencies for their licenses.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

- **Execution Flow**: See `EXECUTION_FLOW.md` for detailed flow
- **CrewAI Docs**: https://docs.crewai.com
- **Gradio Docs**: https://gradio.app/docs

## 🎓 What Was Built

This project demonstrates:
- Multi-agent AI systems with CrewAI
- Parallel task execution
- Semantic search with vector databases
- Integration with OpenAI API
- Web interface with Gradio
- Docker containerization
- Free-tier deployment on Hugging Face Spaces

## 💡 Tips

- **First Run**: Vector DB indexing takes time on first use
- **Large Files**: Use semantic search for large datasets
- **Complex Queries**: Use "Analyze with Question" for specific queries
- **Model Selection**: `gpt-4o` = best quality, `gpt-3.5-turbo` = faster/cheaper
- **Docker**: Use Docker for consistent development environment

## 🔗 Links

- **OpenAI**: https://platform.openai.com
- **CrewAI**: https://docs.crewai.com
- **Gradio**: https://gradio.app

---

**Built with ❤️ using CrewAI and OpenAI**
