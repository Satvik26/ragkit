# ragkit

A local document question-answering system that cites the chunk it used, or says it could not find the answer.

<p align="left">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white"/>
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white"/>
  <img alt="Redis" src="https://img.shields.io/badge/Redis-FF4438?style=for-the-badge&logo=redis&logoColor=white"/>
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img alt="Langfuse" src="https://img.shields.io/badge/Langfuse-000000?style=for-the-badge&logo=langfuse&logoColor=white"/>
  <img alt="pytest" src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white"/>
</p>

## Overview

Most RAG demos answer even when the documents do not contain the fact. ragkit is the opposite shape. A question is embedded, searched with access control inside Qdrant and Postgres, reranked, and then answered only from the chunks that clear a score gate. If the evidence is missing, or a cited number is not in the chunk, the system abstains.

It is built as one process with ports and adapters. Ingest, retrieval, and the answer path depend on interfaces. `config/container.py` is the only place that picks Qdrant, Postgres, Redis, Ollama, or OpenAI. The same services are meant to move to AWS later by swapping those adapters, not by rewriting the query path.

Every retrieval change is scored on a golden question set (Recall@8 and MRR) before it stays. The chat at `GET /` shows the scores, the gate, the citations, and the LangGraph steps, so a wrong answer can be traced to the chunk that caused it.

## Why

- Cited answers, or an explicit abstention, instead of a fluent guess.
- Tenant tags enforced inside vector search and BM25, not as a filter after the fact.
- A local stack that is already the production shape: queue, cache, signed tokens, traces, and a switchable model.
- A written path to ECS, S3, SQS, and RDS once the local adapters are the ones worth deploying.

## Tech stack

| | |
| --- | --- |
| Language | Python 3.12 |
| API | FastAPI, Pydantic, Uvicorn |
| Orchestration | LangGraph corrective RAG |
| Vectors | Qdrant, cosine HNSW, 1024-d BGE-M3 |
| Keywords and rows | PostgreSQL 16, Okapi BM25, document catalog |
| Cache and sessions | Redis 7 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Models | Ollama `gemma4:e2b` or OpenAI `gpt-4.1`, temperature 0 |
| Traces | Langfuse over OpenTelemetry |
| Local run | Docker Compose |
| Tests | pytest, mypy strict, ruff |

AWS (ECS, S3, SQS, RDS, Terraform) is the next phase. It is not in this stack yet.

The first diagram is the system that runs locally. Blue is the API, green is ingest, teal is storage, sky is the query path, gold is the models, and purple is trust and traces. The second diagram pairs each running adapter with its AWS replacement. Amber dashed nodes are Phase 5 and are not built. The cloud profile still raises `NotImplementedError`. MinIO starts with Compose, and the app does not call it.

```mermaid
flowchart TB
  classDef api fill:#1e3a8a,stroke:#93c5fd,color:#eff6ff,stroke-width:1px
  classDef ingest fill:#14532d,stroke:#86efac,color:#f0fdf4,stroke-width:1px
  classDef store fill:#134e4a,stroke:#5eead4,color:#f0fdfa,stroke-width:1px
  classDef query fill:#0c4a6e,stroke:#7dd3fc,color:#f0f9ff,stroke-width:1px
  classDef model fill:#713f12,stroke:#fde047,color:#fefce8,stroke-width:1px
  classDef trust fill:#4a044e,stroke:#f0abfc,color:#fdf4ff,stroke-width:1px

  ui["Chat and inspector<br/>GET /"]:::api

  subgraph http ["FastAPI · container.py"]
    direction TB
    auth["HMAC Bearer<br/>POST /v1/auth/token<br/>x-user-tags only if no secret"]:::api
    probes["/health liveness<br/>/ready checks Qdrant, Postgres,<br/>Redis, embedder, reranker, LLM"]:::api
    qapi["POST /v1/query · LangGraph<br/>POST /v1/query/stream · tokens<br/>POST /v1/retrieve"]:::api
    iapi["POST /v1/documents<br/>POST /v1/documents/folder<br/>POST /v1/jobs/documents<br/>GET /v1/jobs/dead-letter"]:::api
    tapi["GET /v1/traces<br/>POST /v1/feedback"]:::api
  end

  ui --> auth
  auth --> qapi
  auth --> iapi

  subgraph ingestg ["Ingest"]
    direction TB
    corpus["Corpus folder<br/>.txt .md .markdown<br/>PDF and path escape rejected"]:::ingest
    queue["InMemoryIngestionQueue<br/>content-hash job id<br/>3 attempts, then DLQ"]:::ingest
    chunk["FixedSizeChunker<br/>500 words, 100 overlap<br/>id = SHA-256 of doc id + text<br/>acl_tags, start, end, content_hash"]:::ingest
  end

  iapi --> corpus --> queue --> chunk

  subgraph models ["Models"]
    direction TB
    bge["HttpEmbedder<br/>BAAI/bge-m3 · 1024-d<br/>POST /v1/embeddings"]:::model
    rerank["HttpCrossEncoderReranker<br/>ms-marco-MiniLM-L-6-v2<br/>re-score top 16<br/>blend alpha 0.65, keep 8"]:::model
    llm["build_llm · temperature 0<br/>Ollama gemma4:e2b<br/>or OpenAI gpt-4.1<br/>empty key refuses startup<br/>no URL falls back to ExtractiveLLM"]:::model
  end

  chunk --> bge

  subgraph stores ["Stores"]
    direction TB
    catalog["Postgres catalog<br/>documents and chunk rows<br/>parent lookup"]:::store
    bm25["Postgres BM25<br/>Okapi k1=1.5 b=0.75<br/>ACL filter inside the SQL"]:::store
    qdrant["Qdrant collection chunks<br/>cosine HNSW<br/>ACL MatchAny inside search"]:::store
    rcache["Redis query cache<br/>SHA-256 of epoch, ACL tags, model, query<br/>ingest bumps the epoch"]:::store
    rconv["Redis conversations<br/>ACL-scoped session turns<br/>in-memory when Redis URL is empty"]:::store
  end

  bge -->|upsert vectors| qdrant
  chunk -->|save rows| catalog
  chunk -->|upsert terms| bm25
  chunk -.->|bump epoch| rcache

  subgraph queryg ["Query"]
    direction TB
    guard["DefaultGuardrail<br/>block prompt injection<br/>redact email, card, IBAN"]:::query
    ctx["contextualize_follow_up<br/>pronoun, what about, or no topic words<br/>tell me in more detail keeps the previous question"]:::query
    cache["Cache lookup<br/>identical question and model hits<br/>a rewrite or a model switch misses"]:::query
    route["RuleBasedRetrievalRouter<br/>single, or multi_hop with interleaved sub-queries"]:::query
    lex["TechnicalQueryRewriter<br/>rewrites the BM25 string only"]:::query
    hybrid["Hybrid fetch<br/>embed the query with BGE-M3<br/>dense top 100 + BM25 top 100<br/>dense + 0.10 normalized BM25 + RRF"]:::query
    parents["expand_parent_chunks<br/>siblings from the catalog, still ACL-checked"]:::query
    gate["Evidence gate<br/>keep score >= 0.35<br/>otherwise abstain no_evidence"]:::trust
    prompt["Cite-only prompt<br/>one sentence per subject<br/>yes or no when two figures might match<br/>format_model_answer strips LaTeX"]:::trust
    verify["GroundingClaimVerifier<br/>tokens, numbers, and ids such as B12<br/>NOT_FOUND becomes llm_abstained<br/>a bad number becomes unsupported_answer"]:::trust
    trace["Langfuse OTLP<br/>root output is the answer<br/>generate span is the completion, clipped at 8,000 chars<br/>plus in-memory tracer and golden-set Recall@8 / MRR"]:::trust
  end

  qapi --> guard --> ctx
  ctx <--> rconv
  ctx --> cache
  cache <--> rcache
  cache -->|hit| ui
  cache -->|miss| route --> lex --> hybrid
  hybrid --> bge
  hybrid --> qdrant
  hybrid --> bm25
  hybrid --> rerank --> parents
  parents --> catalog
  parents --> gate
  gate -->|no evidence| ui
  gate -->|pass| prompt --> llm --> verify
  verify -->|store answer| rcache
  verify --> trace
  verify --> ui
  tapi --> trace
```

Each row is one port. The left adapter runs today. The right adapter is the AWS replacement from the build plan.

```mermaid
flowchart LR
  classDef local fill:#14532d,stroke:#4ade80,color:#f0fdf4,stroke-width:1px
  classDef planned fill:#7c2d12,stroke:#fdba74,color:#fff7ed,stroke-width:1px,stroke-dasharray: 5 4

  subgraph now ["Local · running"]
    direction TB
    api["API · FastAPI in Docker"]:::local
    queue["Queue · in-memory, DLQ"]:::local
    files["Files · corpus folder"]:::local
    pg["Catalog and BM25 · Postgres"]:::local
    qd["Vectors · Qdrant"]:::local
    redis["Cache and sessions · Redis"]:::local
    models["Embed and rerank · BGE-M3, ms-marco"]:::local
    llm2["LLM · Ollama or OpenAI"]:::local
    obs["Traces · Langfuse"]:::local
    compose["Infra · Docker Compose"]:::local
  end

  subgraph later ["AWS · planned"]
    direction TB
    ecs["API · ECS Fargate, ALB"]:::planned
    sqs["Queue · SQS, ECS workers"]:::planned
    s3["Files · S3"]:::planned
    rds["Catalog · RDS Postgres"]:::planned
    qcloud["Vectors · Qdrant Cloud or OpenSearch"]:::planned
    elastic["Cache · ElastiCache"]:::planned
    gpu["Embed and rerank · GPU or hosted"]:::planned
    bedrock["LLM · Bedrock"]:::planned
    cw["Traces · CloudWatch and Langfuse"]:::planned
    tf["Infra · Terraform"]:::planned
  end

  api -.-> ecs
  queue -.-> sqs
  files -.-> s3
  pg -.-> rds
  qd -.-> qcloud
  redis -.-> elastic
  models -.-> gpu
  llm2 -.-> bedrock
  obs -.-> cw
  compose -.-> tf
```

## Features

### Done

| Area | What runs |
| --- | --- |
| Architecture | Ports and adapters. Adapters are constructed only in `config/container.py`. |
| Ingest | 500-word chunks, 100-word overlap. Folder ingest of `.txt` and `.md`. Postgres catalog. Qdrant vectors. |
| Retrieval | Dense top 100 plus Postgres BM25 top 100, score-calibrated fusion, `ms-marco` rerank of the top 16, keep 8, parent-chunk expansion. ACL is inside both searches. |
| Trust | Evidence gate at 0.35. Cite-only prompt. `GroundingClaimVerifier` rejects a number or id that is not in the cited chunk. LangGraph runs route, retrieve, rewrite, generate, verify. |
| Access | HMAC Bearer tokens. `x-user-tags` is only the fallback when `RAG_AUTH_SECRET` is empty. |
| Speed and jobs | Redis answer cache keyed by ingest epoch, ACL tags, model name, and query. In-memory ingest queue with idempotent job ids, 3 attempts, and a dead-letter queue. |
| Chat | `GET /` inspector. `POST /v1/query/stream`. Guardrail blocks prompt injection and redacts email, card, and IBAN. Session memory. A turn with no topic words continues the previous question. Markdown bullets and bold. LaTeX is stored as plain text before verification. |
| Models | `build_llm` selects Ollama `gemma4:e2b` or OpenAI `gpt-4.1` at temperature 0. Startup fails when the OpenAI key is missing. |
| Observability | Langfuse OTLP. The trace carries the answer. The generate span carries the completion, clipped at 8,000 characters. |
| Quality bar | 129 unit tests, 93% coverage, `mypy --strict` and `ruff` clean. Golden-set Recall@8 and MRR are in the table below. |

The local index currently holds 202 Qdrant points: 48 handbook files and 154 SciFact abstracts. `corpus/scifact/` has more files than that index. A question whose only source is a missing abstract, including `3215494.md`, cannot be answered.

### Plan

| Next | Adds |
| --- | --- |
| Phase 5 · AWS | Adapters only, selected by config: Terraform, ECS Fargate and an ALB, S3, SQS workers, RDS Postgres, Qdrant Cloud or OpenSearch, ElastiCache, Bedrock, CloudWatch. The cloud profile still raises `NotImplementedError`. MinIO is the unused local stand-in for S3. |
| Phase 6 · 20k test | Ingest about 20k documents in the cloud and record throughput, p95 latency, cost, and failure drills. |

These limits stay:

- One uncited number discards the whole answer. The chat replaces the streamed draft with the abstention.
- The verifier checks that cited numbers exist. It does not check that every part of the question was answered.

---

## Live Retrieval & Grounding Inspector (`GET /`)

`ragkit` serves a chat and a retrieval inspector at `http://127.0.0.1:8000/`. Answers stream from `POST /v1/query/stream`. The inspector shows hybrid scores, the evidence gate (`>= 0.35`), multi-hop routing, signed Bearer sessions (`eng`, `finance`, a forged `401`), Redis cache hits, guardrail blocks, and the LangGraph node waterfall. Markdown bullets and bold numbers render in the chat. LaTeX from the model (`$0.8 \text{ mg/d}$`) is stored as plain text (`0.8 mg/d`) before verification.

![Retrieval & Grounding Inspector UI](images/ui.png)
![Retrieval & Grounding Inspector UI](images/cachehit.png)
![Retrieval & Grounding Inspector UI](images/Promptinjection.png)

---

## Evaluation & Ablation Log

Every retrieval, ranking, and fusion change is measured against two golden sets using `uv run python -m ragkit.evals.run`:
1. **Handbook Smoke Set (`evals/golden_set.jsonl`):** 48 company policy documents (`corpus/handbook`), 116 questions (96 answerable + 20 unanswerable out-of-scope questions).
2. **BEIR SciFact Benchmark (`evals/scifact.jsonl`):** Scientific claims and PubMed biomedical abstracts from the official BEIR SciFact test split (evaluated at both **150 abstracts / 100 claims** on CPU `BAAI/bge-m3` and **1,000 abstracts / 300 claims** on Apple Metal GPU `nomic-embed-text`).

| Stage | Dataset | Corpus | Questions | Retrieval Pipeline | Recall@8 | MRR | Step Delta | Cumulative Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Module 1 Baseline** | BEIR SciFact (`scifact.jsonl`) | 150 abstracts | 100 claims | Dense only (`BAAI/bge-m3` 1024-d + Qdrant Cosine) | **0.958** | **0.890** | — | Baseline |
| **Phase 2 · Row 1** | BEIR SciFact (`scifact.jsonl`) | 150 abstracts | 100 claims | **+ Hybrid BM25:** Dense + Okapi BM25 + RRF (`k=60`) | **0.967** | **0.909** | `+0.9%` Recall, `+1.9%` MRR | `+0.9%` Recall, `+1.9%` MRR |
| **Phase 2 · Row 2** | BEIR SciFact (`scifact.jsonl`) | 150 abstracts | 100 claims | **+ Reranker:** `ms-marco-MiniLM-L-6-v2` Cross-Encoder (top 16) | **0.970** | **0.911** | `+0.3%` Recall, `+0.2%` MRR | **`+1.2%` Recall, `+2.1%` MRR** (cuts 29% of misses) |
| **Phase 2 · Row 3** | BEIR SciFact (`scifact.jsonl`) | 150 abstracts | 100 claims | **+ Query Rewrite & Parent-Child:** `TechnicalQueryRewriter` + `expand_parent_chunks` | **0.970** | **0.911** | `+0.0%` on 200w abstracts (targets multi-chunk docs) | **`+1.2%` Recall, `+2.1%` MRR** |
| **1k-Doc Scale Baseline** | BEIR SciFact (`scifact.jsonl`) | **1,000 abstracts** | **300 claims** | Dense only (`nomic-embed-text` 768-d on Apple GPU + Qdrant) | **0.923** | **0.812** | — | 1k-Doc Baseline |
| **1k-Doc Scale Hybrid** | BEIR SciFact (`scifact.jsonl`) | **1,000 abstracts** | **300 claims** | **+ Score-Calibrated Hybrid BM25** (`dense + 0.10 * norm_bm25 + RRF`) | **0.942** | **0.838** | **`+1.9%` Recall, `+2.6%` MRR** | **`+1.9%` Recall, `+2.6%` MRR** (cuts 25% of misses) |

### Key Empirical Findings
- **Score-Calibrated Fusion vs. Naive RRF at Scale:** Scaling from 150 to 1,000 biomedical abstracts showed that unweighted rank-only RRF (`1/(60+rank)`) slightly degraded dense retrieval (`0.923 -> 0.913`) because it discarded the cosine similarity confidence margin between Rank 1 and Rank 2. Preserving the dense cosine score and adding a calibrated BM25 bonus (`dense_score + 0.10 * (bm25 / max_bm25) + 1e-3 * rrf`) lifted **Recall@8 from `0.923` to `0.942` (+1.9%)** and **MRR from `0.812` to `0.838` (+2.6%)** across all 300 claims.
- **Hardware Placement Behind Ports:** Moving embedding inference from the CPU Docker sidecar (`0.78 docs/s`) to native Apple Metal GPU via Ollama's OpenAI-compatible `/v1/embeddings` endpoint (`23.4 docs/s`) increased ingestion throughput **30x** (1,000 documents ingested in **55.4s**) with **zero changes** to `IngestionService` or `RetrievalService`.

---

## Core Production Capabilities Proven

| Capability | Implementation | Verified Behavior |
| --- | --- | --- |
| **1. Hexagonal Architecture (Ports & Adapters)** | Pure `domain/`, narrow `ports/`, vendor `adapters/`, and a single composition root (`config/container.py`). `build_llm` selects the `LLMClient`. | **129 unit tests**, **93% coverage**, `mypy --strict` and `ruff` clean, with no external services. |
| **2. LangGraph Corrective-RAG (`StateGraph`)** | `LangGraphCRAGRunner` (`adapters/agent/langgraph_crag.py`) orchestrates `route -> retrieve -> [rewrite_query] -> generate -> verify`. | Direct lookups take 1 hop; compound questions decompose into `multi_hop` sub-queries and interleave top evidence across documents. |
| **3. Three-Layer Hallucination & Abstention Guards** | **Guard 1:** Evidence gate (`score >= 0.35`, `no_evidence`). **Guard 2:** Cite-only prompt (`llm_abstained`). **Guard 3:** `GroundingClaimVerifier` (`unsupported_answer`). | Deterministic verifier checks content tokens, numeric equivalence (`14 <-> fourteen`), and alphanumeric IDs (`B12`, `PGE2`, `V600E`), blocking entity substitution and ungrounded numbers. |
| **4. Multi-Tenant ACL Security Across Every Layer** | `HmacTokenVerifier` (`Bearer` token) + Qdrant `MatchAny` filter + Postgres SQL `acl_tags && %s` + ACL-scoped Redis cache keys. | Forged tokens fail at the HTTP boundary with `401 Unauthorized`; `finance` tokens cannot retrieve or read cached `eng` answers. |
| **5. ACL-Scoped Redis Query Cache** | `RedisQueryCache` keys entries by `SHA-256(epoch \| acl_tags \| model \| query)` and bumps the epoch on ingest. The stream path reads the cache after follow-up rewrite. | An identical question with the same model returns `cache_hit`. A different ACL tag or a different model cannot read that entry. |
| **6. Durable Ingestion Queue & Dead-Letter Queue** | `InMemoryIngestionQueue` (`adapters/queue/redis_queue.py`) with content-hash idempotency, retry budget (`max_attempts=3`), and DLQ. | Poison documents retry up to 3 times and move to `GET /v1/jobs/dead-letter` instead of blocking ingestion. |
| **7. Langfuse OpenTelemetry (`OTLP/HTTP`) Tracing** | `LangfuseHttpTracer` ships OTLP JSON spans (`agent`, `retriever`, `generation`, `evaluator`) to `/api/public/otel/v1/traces`. | The trace output includes the answer. The generate span includes the model completion (clipped at 8,000 characters), plus the active model name. |
| **8. Switchable LLM** | `LLMClient` is the strategy. `OllamaClient` and `OpenAIChatClient` are the adapters. `build_llm` (`adapters/llm/factory.py`) is the only provider switch. | `RAG_LLM_PROVIDER=ollama` keeps the local model. `RAG_LLM_PROVIDER=openai` calls Chat Completions at temperature 0 and fails startup when `RAG_OPENAI_API_KEY` is empty. |
| **9. Chat, guardrails, and session memory** | `POST /v1/query/stream`, `DefaultGuardrail`, and an ACL-scoped `ConversationStore`. | Prompt injection stops at `guard_input`. Email, card, and IBAN are redacted before retrieval. A follow-up is rewritten for a pronoun, for "what about", or when no topic words remain. A named topic stays a new search. |

---

## Repository Layout

```
src/ragkit/
  domain/     Pure entities: Chunk, ScoredChunk, Citation, Answer, StoredDocument, StoredChunk
  ports/      Interfaces: Chunker, Embedder, VectorWriter/Searcher, LexicalWriter/Searcher,
              QueryRewriter, Reranker, RetrievalRouter, LLMClient, ClaimVerifier,
              QueryCache, IngestionQueue, TokenVerifier, Tracer, Guardrail, ConversationStore
  adapters/
    agent/    LangGraphCRAGRunner (compiled StateGraph: route -> retrieve -> rewrite -> generate -> verify)
    auth/     HmacTokenVerifier & sign_token (HMAC-SHA256 Bearer tokens)
    cache/    InMemoryQueryCache & RedisQueryCache (RESP2 socket client, epoch invalidation)
    catalog/  InMemoryDocumentCatalog & PostgresDocumentCatalog
    chunk/    FixedSizeChunker (500w window, 100w overlap, deterministic SHA-256 chunk IDs)
    embed/    HttpEmbedder (OpenAI-compatible /v1/embeddings client)
    lexical/  InMemoryBM25Store & PostgresBM25Store (Okapi BM25 with SQL ACL filter)
    llm/      OllamaClient, OpenAIChatClient, ExtractiveLLM, and build_llm
    conversation/ ACL-scoped session turns
    guardrail/ DefaultGuardrail (injection block, PII redaction)
    queue/    InMemoryIngestionQueue (idempotent job IDs, retry budget, dead-letter queue)
    rerank/   PassthroughReranker & HttpCrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
    rewrite/  PassthroughQueryRewriter, TechnicalQueryRewriter, LLMQueryRewriter
    route/    PassthroughRouter & RuleBasedRetrievalRouter (single vs multi_hop decomposition)
    trace/    InMemoryTracer & LangfuseHttpTracer (OpenTelemetry OTLP/HTTP exporter)
    vector/   InMemoryVectorStore & QdrantVectorStore (Cosine HNSW with ACL MatchAny)
    verify/   PassthroughVerifier, GroundingClaimVerifier, LLMClaimVerifier
  services/   IngestionService, RetrievalService, AnswerService, folder_ingest (depend on ports only)
  config/     Settings & the single composition root (container.py)
  api/        FastAPI app factory + built-in Inspector UI (ui.py)
  evals/      Recall@k, MRR, and golden-set CLI runner
deploy/
  embedder/   Isolated FastAPI sidecar serving BAAI/bge-m3 (/v1/embeddings) & Cross-Encoder (/v1/rerank)
tests/
  unit/       129 unit tests across services, adapters, LangGraph, auth, queue, cache, and API
  contract/   Shared Liskov contract suites every VectorStore and LexicalStore adapter must pass
evals/        Handbook golden set, BEIR SciFact builder, and benchmark JSONL files
docs/         Local notes only. Gitignored, so this folder is not published
```

---

## Quick Start

### 1. Run Unit & Contract Tests (Zero External Services Needed)

```bash
uv sync
uv run ruff check . && uv run mypy && uv run pytest
```

### 2. Start the Full Local Stack (Qdrant, Postgres 16, Redis 7, BGE-M3 + Reranker Sidecar, API)

```bash
docker compose up -d qdrant postgres redis embedder api
curl -s http://127.0.0.1:8000/ready
```

Open the chat and inspector at **`http://127.0.0.1:8000/`**. Answers stream from `POST /v1/query/stream`.

The default LLM is Ollama (`RAG_LLM_PROVIDER=ollama`, model `gemma4:e2b`). To use OpenAI, set these in `.env` and recreate the API container. `docker compose restart` does not reload `.env`.

```bash
# .env
# RAG_LLM_PROVIDER=openai
# RAG_OPENAI_API_KEY=...
# RAG_OPENAI_MODEL=gpt-4.1
docker compose up -d --no-deps api
```

After a provider change, the same question runs again. The Redis key includes the model name, so it does not return the previous model's answer.

### 3. Ingest Corpus & Query via Signed Bearer Token

```bash
# Ingest the 48-file company handbook
curl -s http://127.0.0.1:8000/v1/documents/folder \
  -H 'content-type: application/json' \
  -H 'x-user-tags: eng' \
  -d '{"path":"handbook"}'

# Mint a signed HMAC-SHA256 Bearer token for user "satvik" with ACL tag "eng"
TOKEN=$(curl -s http://127.0.0.1:8000/v1/auth/token \
  -H 'content-type: application/json' \
  -d '{"sub":"satvik","tags":["eng"]}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

# Run a multi-hop query authenticated via Bearer token
curl -s http://127.0.0.1:8000/v1/query \
  -H 'content-type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query":"How long do refunds take and what is the deadline for an incident postmortem?"}'
```

### 4. Run the Golden-Set Evaluation CLI

```bash
RAG_QDRANT_URL=http://127.0.0.1:6333 \
RAG_DATABASE_URL=postgresql://rag:rag@127.0.0.1:5433/rag \
RAG_EMBEDDING_URL=http://127.0.0.1:8081 \
RAG_RERANKER_URL=http://127.0.0.1:8081 \
RAG_EMBEDDING_DIM=1024 \
uv run python -m ragkit.evals.run --path evals/golden_set.jsonl
```

