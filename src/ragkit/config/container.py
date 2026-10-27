"""The only place that knows about concrete adapters (composition root)."""

from __future__ import annotations

from collections.abc import Callable

from qdrant_client import QdrantClient

from ragkit.adapters.cache.redis_cache import InMemoryQueryCache, RedisQueryCache
from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.catalog.postgres import PostgresDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.conversation.store import (
    InMemoryConversationStore,
    RedisConversationStore,
)
from ragkit.adapters.dev.stubs import HashingEmbedder
from ragkit.adapters.embed.http import HttpEmbedder
from ragkit.adapters.guardrail.default_guardrail import DefaultGuardrail
from ragkit.adapters.lexical.bm25 import InMemoryBM25Store, PostgresBM25Store
from ragkit.adapters.llm.factory import build_llm
from ragkit.adapters.rerank.http import HttpCrossEncoderReranker
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.rewrite.rewriter import TechnicalQueryRewriter
from ragkit.adapters.route.router import RuleBasedRetrievalRouter
from ragkit.adapters.trace.langfuse import InMemoryTracer, LangfuseHttpTracer
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.adapters.vector.qdrant_store import QdrantVectorStore, qdrant_is_ready
from ragkit.adapters.verify.verifier import GroundingClaimVerifier
from ragkit.config.settings import Settings
from ragkit.ports.cache import QueryCache
from ragkit.ports.catalog import DocumentCatalog
from ragkit.ports.conversation import ConversationStore
from ragkit.ports.embedder import Embedder
from ragkit.ports.lexical_store import LexicalSearcher, LexicalWriter
from ragkit.ports.llm import LLMClient
from ragkit.ports.reranker import Reranker
from ragkit.ports.tracer import Tracer
from ragkit.services.answering import AnswerService
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService


def build_services(
    settings: Settings,
) -> tuple[AnswerService, IngestionService, Callable[[], bool] | None]:
    """One store and one embedder, shared by ingest and query.

    The third value checks every configured dependency. In-memory mode has nothing to check.
    """
    if settings.profile == "cloud":
        raise NotImplementedError(
            "Cloud profile: add real Embedder and LLMClient adapters here (see ports/)."
        )
    store: InMemoryVectorStore | QdrantVectorStore
    checks: list[Callable[[], bool]] = []
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, timeout=2)
        store = QdrantVectorStore(client, settings.collection, settings.embedding_dim)

        def qdrant_check() -> bool:
            return qdrant_is_ready(client)

        checks.append(qdrant_check)
    else:
        store = InMemoryVectorStore()
    catalog: DocumentCatalog
    lexical_writer: LexicalWriter
    lexical_searcher: LexicalSearcher
    if settings.database_url:
        postgres = PostgresDocumentCatalog(settings.database_url)
        catalog = postgres
        pg_bm25 = PostgresBM25Store(settings.database_url)
        lexical_writer = pg_bm25
        lexical_searcher = pg_bm25
        checks.append(postgres.ready)
    else:
        catalog = InMemoryDocumentCatalog()
        mem_bm25 = InMemoryBM25Store()
        lexical_writer = mem_bm25
        lexical_searcher = mem_bm25
    embedder: Embedder
    if settings.embedding_url:
        http_embedder = HttpEmbedder(
            settings.embedding_url, settings.embedding_dim, settings.embedding_model
        )
        embedder = http_embedder
        checks.append(http_embedder.ready)
    else:
        embedder = HashingEmbedder(settings.embedding_dim)
    reranker: Reranker
    if settings.reranker_url:
        http_reranker = HttpCrossEncoderReranker(settings.reranker_url, settings.reranker_model)
        reranker = http_reranker
        checks.append(http_reranker.ready)
    else:
        reranker = PassthroughReranker()
    binding = build_llm(settings)
    llm: LLMClient = binding.client
    if binding.ready is not None:
        checks.append(binding.ready)
    cache: QueryCache
    conversations: ConversationStore
    if settings.redis_url:
        redis_cache = RedisQueryCache(settings.redis_url, model_name=binding.model_name)
        cache = redis_cache
        conversations = RedisConversationStore(settings.redis_url)
        checks.append(redis_cache.ready)
    else:
        cache = InMemoryQueryCache(model_name=binding.model_name)
        conversations = InMemoryConversationStore()
    tracer: Tracer
    if settings.langfuse_url and settings.langfuse_public_key and settings.langfuse_secret_key:
        tracer = LangfuseHttpTracer(
            settings.langfuse_url,
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            model_name=binding.model_name,
        )
    else:
        tracer = InMemoryTracer()
    retrieval = RetrievalService(
        embedder,
        store,
        reranker,
        lexical_searcher=lexical_searcher,
        query_rewriter=TechnicalQueryRewriter(),
        catalog=catalog,
        router=RuleBasedRetrievalRouter(),
    )
    answer = AnswerService(
        retrieval,
        llm,
        settings.min_evidence_score,
        verifier=GroundingClaimVerifier(),
        tracer=tracer,
        cache=cache,
        guardrail=DefaultGuardrail(),
        conversations=conversations,
    )
    ingestion = IngestionService(
        FixedSizeChunker(),
        embedder,
        store,
        catalog,
        lexical_writer=lexical_writer,
        cache=cache,
    )
    return answer, ingestion, _all_ready(checks)


def build_answer_service(settings: Settings) -> AnswerService:
    return build_services(settings)[0]


def _all_ready(checks: list[Callable[[], bool]]) -> Callable[[], bool] | None:
    if not checks:
        return None

    def ready() -> bool:
        return all(check() for check in checks)

    return ready
