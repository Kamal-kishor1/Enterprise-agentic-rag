# 🧠 Enterprise Agentic RAG System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Render-Backend_API-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![Neon](https://img.shields.io/badge/Neon-Serverless_Postgres-00E599?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech/)
[![Upstash](https://img.shields.io/badge/Upstash-Serverless_Redis-00E9A3?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_RRF_Search-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Locust](https://img.shields.io/badge/Locust-Load_Tested-708090?style=for-the-badge&logo=locust&logoColor=white)](https://locust.io/)
[![GitHub Actions](https://img.shields.io/badge/CI-Automated_Pipeline-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)

> 🚀 **Live Production Application:** [https://agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app/](https://agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app/)
>
> **Enterprise Agentic RAG** is an asynchronous, hardened, multi-tier Retrieval-Augmented Generation platform engineered for high-concurrency enterprise deployments. It combines **LangGraph cyclic state machines**, **Qdrant dense + sparse hybrid search with Reciprocal Rank Fusion (RRF)**, **FlashRank cross-encoders**, a **resilient LiteLLM gateway**, **durable PostgreSQL session checkpointing**, **Redis response caching**, and **Llama-Prompt-Guard security rails** into an orchestrated microservice architecture.

---

## 🏗️ End-to-End System Architecture

The following diagram illustrates the complete query lifecycle from client ingress through security, caching, agentic reasoning, hybrid search, and streaming token response:

```mermaid
flowchart TD
    subgraph ClientLayer ["1. Client & Presentation Layer"]
        User["👤 Enterprise User / API Client"]
        StreamlitUI["🖥️ Streamlit Interactive UI\n(ui/app.py:8501)"]
        User -->|Query / Stream| StreamlitUI
    end

    subgraph IngressLayer ["2. API Gateway & Ingress (app/main.py:8003)"]
        StreamlitUI -->|POST /query/stream| FastAPI["FastAPI Asynchronous Gateway"]
        
        FastAPI --> PII["🛡️ In-Memory PII & Secret Redactor\n(app/services/guardrails.py)"]
        PII --> CacheCheck{"⚡ Two-Tier Cache Check\n(Redis / In-Memory)"}
    end

    subgraph CacheTier ["3. Response Caching (app/services/cache_service.py)"]
        CacheCheck -->|Cache Hit: < 3ms| InstantStream["⚡ Instant SSE Return\n(0 Tokens • 2.5ms)"]
        InstantStream -.-> StreamlitUI
    end

    subgraph SecurityTier ["4. Threat Defense & Guardrails"]
        CacheCheck -->|Cache Miss| PromptGuard["🚨 Non-Blocking Prompt Guard\n(meta-llama/llama-prompt-guard-2-86m)"]
        PromptGuard -->|Injection / Jailbreak Detected| BlockResponse["⛔ 403 Security Interception\n(X-Security: BLOCKED)"]
        BlockResponse -.-> StreamlitUI
    end

    subgraph AgenticCore ["5. Agentic Brain (LangGraph StateGraph)"]
        PromptGuard -->|Security Passed| PlannerNode["🧠 Intent Planner Node\n(app/agents/nodes/planner.py)"]
        
        PlannerNode -->|Conversational Intent| ResponderNode["💬 Responder Synthesis Node\n(app/agents/nodes/responder.py)"]
        PlannerNode -->|Technical Intent| RetrieverNode["🔍 Hybrid Retriever Node\n(app/agents/nodes/retriever.py)"]
    end

    subgraph MemoryTier ["6. Durable State & Checkpointing"]
        Checkpointer["🐘 PostgreSQL Checkpointer Pool\n(langgraph-checkpoint-postgres:5432)"]
        Checkpointer <-->|Thread Context & History| AgenticCore
    end

    subgraph RetrievalEngine ["7. Hybrid Search & Neural Reranking"]
        RetrieverNode --> Dense["Gemini 3072-d Dense Embedding\n(gemini-embedding-2-preview)"]
        RetrieverNode --> Sparse["FastEmbed BM25 Sparse Vector\n(Qdrant Sparse Encoding)"]
        
        Dense --> QdrantRRF[("🎯 Qdrant Cloud Hybrid Index\n(Reciprocal Rank Fusion - Top 15)")]
        Sparse --> QdrantRRF
        
        QdrantRRF -->|Top 15 Candidates| CrossEncoder["⚡ FlashRank Cross-Encoder\n(ms-marco-TinyBERT Local Reranker)"]
        CrossEncoder -->|Top 5 Context Chunks| ResponderNode
        
        CircuitBreaker["🔌 Circuit Breaker Fallback\n(data/fallback.json)"] -.->|On Qdrant Downtime| CrossEncoder
    end

    subgraph LLMGateway ["8. Resilient Multi-Model LLM Gateway"]
        ResponderNode --> GatewayRouter["🔀 LiteLLM Router Gateway\n(app/services/llm/gateway.py)"]
        GatewayRouter -->|Primary: 100% Traffic| PrimaryLLM["Groq openai/gpt-oss-120b\n(Streaming SSE Chunking)"]
        GatewayRouter -.->|Auto-Failover on 429/500| FallbackLLM["Groq openai/gpt-oss-20b\n(Secondary Model)"]
    end

    ResponderNode --> SSETokenStream["📡 SSE Real-Time Token Generator\n(astream_events v2)"]
    SSETokenStream --> StreamlitUI
    SSETokenStream -.->|Async Cache Write| CacheTier
```

---

## 🌐 Cloud Production Deployment Topology

The system is deployed across a decoupled, multi-cloud microservice architecture ensuring high availability, sub-5ms caching, and automated failover:

```mermaid
flowchart TD
    subgraph FrontendPlatform ["Frontend Tier — Streamlit Community Cloud"]
        User["👤 End User / Web Browser"]
        StreamlitCloud["🖥️ Streamlit Cloud Dashboard\n(agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app)"]
        User -->|Interactive Chat & SSE Streams| StreamlitCloud
    end

    subgraph BackendPlatform ["API & Agentic Tier — Render Web Service"]
        RenderService["⚡ FastAPI Backend Engine\n(Dockerfile.backend on Render)\nDynamic $PORT • Health Check /health"]
        StreamlitCloud -->|HTTPS / POST /query/stream| RenderService
    end

    subgraph DatabasePlatform ["Persistence Tier — Neon Serverless PostgreSQL"]
        NeonDB[("🐘 Neon PostgreSQL 16\n(ep-...-pooler.neon.tech:5432)\nAsyncCompatiblePostgresSaver • Auto-Migrating")]
        RenderService <-->|Connection Pooling / SSL| NeonDB
    end

    subgraph CachePlatform ["Caching Tier — Upstash Serverless Redis"]
        UpstashRedis[("⚡ Upstash Redis Cache\n(frank-...upstash.io:6379)\nTLS/SSL (rediss://) • Volatile-LRU")]
        RenderService <-->|Sub-5ms Latency / SHA-256 Keys| UpstashRedis
    end

    subgraph AIPlatforms ["AI & Vector Cloud Infrastructure"]
        QdrantCloud[("🎯 Qdrant Cloud Cluster\nHybrid Dense + BM25 Sparse Vector Store")]
        GroqCloud["⚡ Groq Cloud LPU\nPrimary: gpt-oss-120b | Fallback: gpt-oss-20b\nPrompt Guard: Llama-Prompt-Guard-2-86m"]
        GeminiCloud["🧠 Google Gemini API\ngemini-embedding-2-preview (3072-dim)"]
        LogfireCloud["📊 Pydantic Logfire\nDistributed Tracing & Real-time Spans"]
        
        RenderService <-->|RRF Hybrid Search| QdrantCloud
        RenderService <-->|LiteLLM Resilient Router| GroqCloud
        RenderService <-->|Dense Embeddings| GeminiCloud
        RenderService -.->|OTel Spans & Metrics| LogfireCloud
    end
```

### Production Microservices Matrix

| Platform | Role | Service / Technology | Key Configuration & SLA |
| :--- | :--- | :--- | :--- |
| **Streamlit Cloud** | Frontend UI | `ui/app.py` | Real-time SSE token streaming, zero-latency state handling |
| **Render** | API Gateway | FastAPI + Uvicorn (`Dockerfile.backend`) | Auto-deploy on `main`, dynamic `$PORT` binding, `/health` probe |
| **Neon** | State Checkpoint | PostgreSQL 16 Serverless | Connection pooling (`-pooler`), SSL enforced, automatic schema migrations |
| **Upstash** | Response Cache | Redis 7 Serverless (TLS) | Sub-5ms latency (co-located in AWS Ohio), SHA-256 query hashing, Volatile-LRU |
| **Qdrant Cloud** | Vector Engine | Hybrid Dense + Sparse Vector DB | Reciprocal Rank Fusion (RRF), cosine metric, cloud payload filtering |
| **Groq Cloud** | LLM Synthesis & Security | LPU Ultra-Fast Inference | LiteLLM routing (120B -> 20B failover), Prompt Guard 86M (<120ms) |
| **Google Gemini** | Embeddings | `gemini-embedding-2-preview` | 3072-dimensional high-precision dense semantic vectors |
| **Pydantic Logfire** | Observability | OpenTelemetry APM | Distributed tracing, token counters, latency percentiles |

---

## 🏆 Key Production Engineering Milestones

| # | Engineering Milestone | Architecture / Module | Production Impact |
| :-: | :--- | :--- | :--- |
| **1** | **Strict Grounding & Exact Citations** | `responder.py` | Eliminates hallucinations; enforces bracketed citations (`[doc.ext]`) and honest abstention. |
| **2** | **Circuit Breaker & Local Fallback** | `qdrant_service.py` | Fail-fast local JSON fallback trips in <5ms during vector DB cloud outages. |
| **3** | **Token Budgeting & Sliding Memory** | `responder.py` | Enforces 4,000-char context window budgets preventing context-window overflow. |
| **4** | **Native Asynchronous Architecture** | `graph.py`, `app/main.py` | Non-blocking `ainvoke` and `astream_events` throughout the pipeline. |
| **5** | **Server-Sent Events (SSE) Streaming** | `main.py`, `ui/app.py` | Line-aware token streaming delivering immediate Time-To-First-Token (TTFT). |
| **6** | **Resilient Multi-Model Gateway** | `gateway.py` | LiteLLM routing with automatic failover between primary (`120b`) and fallback (`20b`) models. |
| **7** | **Durable PostgreSQL Checkpointing** | `graph.py` | Connection-pooled `PostgresSaver` state persistence with in-memory fallback. |
| **8** | **Qdrant Hybrid Search (Dense + Sparse)** | `qdrant_service.py` | 3072-d Gemini dense vectors + BM25 sparse vectors merged via Reciprocal Rank Fusion (RRF). |
| **9** | **Two-Tier Response Cache** | `cache_service.py` | Redis + local TTL dictionary delivering **2,858x faster responses (2.53ms)** with 0 tokens. |
| **10** | **Quantitative RAGAS Benchmark Suite** | `test_ragas_evals.py` | Verified **0.95 Faithfulness**, **0.95 Context Recall**, and **0.95 Relevance**. |
| **11** | **Llama-Prompt-Guard & PII Defense** | `guardrails.py` | **100% defense rate** against adversarial jailbreaks; auto-redacts API keys and emails. |
| **12** | **Enterprise Multi-Stage Docker** | `Dockerfile`, `docker-compose.yml` | Multi-target build caching, authenticated PostgreSQL (`pg_isready`), and Redis (`requirepass`). |
| **13** | **Locust Load Testing & Concurrency** | `tests/locustfile.py` | Diagnosed event loop starvation; refactored to Cache-First ingress achieving **0 failures under 50 users**. |
| **14** | **GitHub Actions Automated CI** | `.github/workflows/ci.yml` | Automated code quality, byte compilation, Postgres service container, and Docker build checks. |

---

## 📊 Empirical Benchmarks

### 1. Quantitative RAGAS Evaluation Suite
Evaluated against enterprise technical documentation test datasets using `tests/test_ragas_evals.py`:

| Evaluation Metric | Baseline RAG | Production Hardened RAG | Target SLA | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness (Factual Consistency)** | 0.62 | **0.95** | > 0.85 | ✅ Passed |
| **Context Recall (Retrieval Coverage)** | 0.58 | **0.95** | > 0.85 | ✅ Passed |
| **Answer Relevance (Query Alignment)** | 0.71 | **0.95** | > 0.85 | ✅ Passed |

---

### 2. Locust High-Concurrency Load Test (50 Concurrent Users)
Simulating 162 enterprise user sessions across cache hits, cold hybrid RAG, multi-turn memory, and adversarial attacks:

| Metric | Before Concurrency Optimization | After Cache-First + Async Guardrails | Magnitude of Improvement |
| :--- | :---: | :---: | :---: |
| **Median System Latency (p50)** | 11,000 ms (11.0s) | **560 ms** | **19.6x Faster** |
| **Cache Hit Median Latency** | 5,000 ms (5.0s) | **240 ms** | **20.8x Faster** |
| **Min Cache Response Time** | 158 ms | **2.64 ms** | **Sub-3ms Verified** |
| **Stream Failures (`ChunkedEncodingError`)** | 43 broken sockets | **0 FAILURES (100% Reliability)** | **100% Elimination** |
| **Aggregated Average Latency** | 21,012 ms (21.0s) | **3,679 ms (3.6s)** | **5.7x Latency Reduction** |

---

## 🐳 Microservices Topology (Docker Compose)

The multi-container production stack is orchestrated inside an isolated private bridge network (`rag-network`):

```mermaid
flowchart LR
    subgraph Host ["Host Machine"]
        ClientBrowser["🌐 Web Browser / API Client"]
    end

    subgraph DockerNetwork ["Docker Bridge Network (rag-network)"]
        direction TB
        
        subgraph FrontendContainer ["frontend (Streamlit UI)"]
            UIApp["ui/app.py\n(Port: 8501)"]
        end
        
        subgraph BackendContainer ["backend (FastAPI + LangGraph)"]
            APIApp["app/main.py\n(Port: 8003)"]
            HealthCheck["Health Probe\nGET /health"]
        end
        
        subgraph CacheContainer ["redis (Redis 7.0 Alpine)"]
            RedisServer["Authenticated Cache\n(--requirepass)\nPort: 6379"]
        end
        
        subgraph DBContainer ["postgres (PostgreSQL 16 Alpine)"]
            PGServer["Durable Checkpointer\n(rag_memory)\nPort: 5432"]
        end
    end

    ClientBrowser -->|Port 8501| UIApp
    ClientBrowser -->|Port 8003| APIApp
    UIApp -->|http://backend:8003| APIApp
    APIApp -->|redis://:password@redis:6379/0| RedisServer
    APIApp -->|postgresql://postgres:password@postgres:5432/rag_memory| PGServer
```

---

## 🚀 Quickstart & Deployment

### 🌐 Live Cloud Demo
Launch the live, production-grade application immediately without any local setup:
- **Streamlit Interactive UI:** [https://agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app/](https://agentic-rag-duevwyskduagcdkqhi6gqj.streamlit.app/)

---

### Method 1: Local Multi-Container Launch with Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kamal-kishor1/Enterprise-agentic-rag.git
   cd Enterprise-agentic-rag
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to include your `GROQ_API_KEY`, `GEMINI_API_KEY`, `POSTGRES_PASSWORD`, and `REDIS_PASSWORD`.*

3. **Build and launch the entire stack:**
   ```bash
   docker-compose up -d --build
   ```

4. **Access your services:**
   - **Streamlit Chat Workspace:** `http://localhost:8501`
   - **FastAPI Documentation:** `http://localhost:8003/docs`
   - **Liveness Health Probe:** `http://localhost:8003/health`

---

### Method 2: Local Development Setup

1. **Create and activate a Python 3.11 virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Backend:**
   ```powershell
   uvicorn app.main:app --port 8000 --reload
   ```

4. **Run the Streamlit Frontend (in a separate terminal):**
   ```powershell
   streamlit run ui/app.py
   ```

---

## 🧪 Testing & Quality Assurance

### Run Unit & Guardrail Red-Teaming Tests:
```powershell
pytest -v
```

### Run Full RAGAS Evaluation Benchmark:
```powershell
python -m tests.test_ragas_evals
```

### Run High-Concurrency Locust Load Test:
```powershell
locust -f tests/locustfile.py --host http://localhost:8000
```
*Open `http://localhost:8089` to launch interactive virtual user swarms.*

---

## 🛡️ Security & Guardrails Profile

- **Adversarial Red-Teaming:** Intercepts prompt injections, jailbreaks, and privilege-escalation commands using Groq's `meta-llama/llama-prompt-guard-2-86m`.
- **Zero-Secret Logging:** Output secret sanitization scrubs API tokens (`gsk_`, `AIzaSy`, `Bearer`) and emails before emitting final LLM tokens.
- **12-Factor App Compliance:** Zero passwords or keys committed to Git. All secrets enforced with fail-fast validation (`${VAR:?error}`).