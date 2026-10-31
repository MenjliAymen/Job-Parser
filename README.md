# Job Parser

Analyze freelance job postings with an LLM: break a job into subtasks with hour estimates, read the client's intent, compare jobs, check a freelancer's skills against a job, and build a knowledge graph of recurring tasks. Comes with a web UI and a JSON API.

## Quick start

```bash
python -m venv venv && venv\Scripts\activate      # source venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
copy .env.example .env                            # cp on macOS/Linux, then pick an LLM option below
uvicorn Job_parser.api.server:app --reload
```

Open **http://127.0.0.1:8000/ui/**, pick a tool, click **Load example**, then **Analyze**. The API docs are at `/docs`.

### Choosing an LLM

| Option | Setup in `.env` | Notes |
|---|---|---|
| **Groq** (hosted, free tier) | `LLM_PROVIDER=openai`, `LLM_API_KEY=<key from console.groq.com>` | Default models `openai/gpt-oss-20b` (fast) and `openai/gpt-oss-120b` (reasoning); `GET /health` lists what your key can use. Any OpenAI-compatible API works by also setting `LLM_BASE_URL` and `LLM_MODEL`. |
| **Ollama** (local) | nothing (default) | `ollama pull mistral && ollama pull deepseek-r1` |
| **Fake server** (no model) | nothing (default) | Run `python -m devtools.fake_llm_server` in a second terminal. Answers come from keyword heuristics: good for trying the UI or developing, not for real analysis. |

The badge in the UI's top-right corner shows which provider and models are in use and whether they are reachable.

## Tools

| UI tool | Endpoint(s) | What it does |
|---|---|---|
| Analyze job | `POST /analyze_jobs`, then `/build_profiles`, `/graph/ingest_subtasks` | Summary, domain, hour budget from the Upwork duration, 4 subtasks with category, effort share, hours and priority, skill-profile tree, add to graph |
| Intent & categories | `POST /analyze_intent`, `POST /transform_job` | Client's goal and technical understanding; cleaned description and general categories |
| Responsibilities | `POST /parse` | Bullet responsibilities → action summaries, subtasks, matched skills |
| Compare jobs | `POST /compare_jobs` | Shared vs. job-specific skills, transferability score |
| My skills vs job | `POST /validate_job_skills` | Relevant, not-needed and missing skills, with adjustable strictness |
| Compare skill lists | `POST /compare_skills_lists` | Overlap and unique skills of two lists |
| Knowledge graph | `GET /graph?q=&limit_per_group=`, `GET /export_graph` | Tasks merged across jobs (one LLM call per new task, whatever the graph size), searchable, grouped by category |

If the LLM is down or returns something unusable, the API answers **502** with `detail` and the model's `raw_response`.

## Tests

```bash
pytest
```

No LLM needed: unit tests use a scripted fake, and `tests/test_e2e_fake_server.py` runs every endpoint over real HTTP against the fake server, once per provider.

## Docker

```bash
docker build -t job-parser .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data job-parser
```

With Ollama on the host, the container reaches it at `host.docker.internal:11434` by default.

## Category clustering (offline)

`Clustering/` groups the free-text categories from `/transform_job` into canonical ones (see the docstring in `generalized_category_pipeline.py`):

```bash
pip install -r requirements-clustering.txt
python Clustering/generalized_category_pipeline.py --jobs-csv Clustering/cleaned_job_db.csv --output-dir artifacts
```
