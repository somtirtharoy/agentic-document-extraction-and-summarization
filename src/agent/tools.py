import json
from collections import Counter, defaultdict
from typing import Any

from config.settings import get_settings
from src.agent.memory import AgentMemory
from src.agent.schemas import (
    AggregateEntitiesInput,
    CompareSummariesInput,
    ExtractEntitiesInput,
    SearchDocumentsInput,
    SummarizeDocumentInput,
)
from src.extraction.gemini_extractor import GeminiExtractor
from src.extraction.gcp_nl_extractor import GCPNLExtractor
from src.gcp.bq_client import BQClient
from src.gcp.vertex_client import get_model
from src.summarization.gemini_summarizer import GeminiSummarizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _err(message: str, hint: str = "") -> dict:
    return {"error": message, "hint": hint}


class AgentTools:
    def __init__(self, memory: AgentMemory) -> None:
        self._memory = memory
        self._bq = BQClient()
        self._nl_extractor = GCPNLExtractor()
        self._gemini_extractor = GeminiExtractor()
        self._summarizer = GeminiSummarizer()
        settings = get_settings()
        self._project = settings.gcp_project_id

    # ── Tool 1: search_documents ──────────────────────────────────────────────

    def search_documents(self, query: str, top_k: int = 5) -> dict:
        try:
            args = SearchDocumentsInput(query=query, top_k=top_k)
        except Exception as e:
            return _err(str(e), "Check query is a non-empty string and top_k is 1–10")

        try:
            # BQ keyword search — production upgrade path is VECTOR_SEARCH
            # with text-embedding-004 for semantic similarity
            sql = f"""
                SELECT
                    doc_id,
                    SUBSTR(text, 1, 300) AS snippet,
                    (
                        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(LOWER(text), LOWER(@query)))
                    ) AS score
                FROM `{self._bq.table_ref('documents_clean')}`
                WHERE REGEXP_CONTAINS(LOWER(text), LOWER(@query))
                ORDER BY score DESC
                LIMIT @top_k
            """
            from google.cloud.bigquery import ScalarQueryParameter, QueryParameterValue
            params = [
                ScalarQueryParameter("query", "STRING", args.query),
                ScalarQueryParameter("top_k", "INT64", args.top_k),
            ]
            rows = self._bq.query(sql, params)

            if not rows:
                return {"results": [], "message": f"No documents found matching '{args.query}'"}

            return {
                "results": [
                    {"doc_id": r["doc_id"], "snippet": r["snippet"], "score": r["score"]}
                    for r in rows
                ]
            }
        except Exception as e:
            logger.error("search_documents failed", extra={"error": str(e)})
            return _err(str(e), "Try a simpler query with fewer special characters")

    # ── Tool 2: extract_entities ──────────────────────────────────────────────

    def extract_entities(self, doc_id: str) -> dict:
        try:
            ExtractEntitiesInput(doc_id=doc_id)
        except Exception as e:
            return _err(str(e), "Provide a valid doc_id string")

        # Check cache first
        cached = self._memory.cache_get(doc_id, "entities")
        if cached:
            return {"doc_id": doc_id, "entities": cached, "source": "cache"}

        # Fetch text from BQ
        try:
            rows = self._bq.query(
                f"SELECT text FROM `{self._bq.table_ref('documents_clean')}` WHERE doc_id = @doc_id LIMIT 1",
                [__import__("google.cloud.bigquery", fromlist=["ScalarQueryParameter"])
                 .ScalarQueryParameter("doc_id", "STRING", doc_id)],
            )
            if not rows:
                return _err(f"doc_id '{doc_id}' not found", "Use search_documents to find valid doc_ids")
            text = rows[0]["text"]
        except Exception as e:
            return _err(str(e), "BQ fetch failed")

        # Run Cloud NL extraction
        try:
            entities, categories = self._nl_extractor.extract(doc_id, text)
            entity_list = [
                {
                    "name": e.name,
                    "type": e.type,
                    "salience": e.salience,
                    "sentiment_score": e.sentiment_score,
                }
                for e in entities[:20]  # cap to top 20 by salience
            ]
            result = {"entities": entity_list, "categories": categories}
            self._memory.cache_set(doc_id, "entities", result)
            return {"doc_id": doc_id, **result, "source": "cloud_nl"}
        except Exception as e:
            logger.error("extract_entities failed", extra={"doc_id": doc_id, "error": str(e)})
            return _err(str(e), "Cloud NL API call failed — try again")

    # ── Tool 3: summarize_document ────────────────────────────────────────────

    def summarize_document(self, doc_id: str, style: str = "tldr") -> dict:
        try:
            args = SummarizeDocumentInput(doc_id=doc_id, style=style)  # type: ignore[arg-type]
        except Exception as e:
            return _err(str(e), "style must be one of: tldr, bullets, abstract")

        # Check BQ summaries table first
        try:
            rows = self._bq.query(
                f"SELECT tldr, bullets, abstract FROM `{self._bq.table_ref('summaries')}` WHERE doc_id = @doc_id LIMIT 1",
                [__import__("google.cloud.bigquery", fromlist=["ScalarQueryParameter"])
                 .ScalarQueryParameter("doc_id", doc_id, "STRING")],
            )
            if rows:
                row = rows[0]
                summary = {
                    "tldr": row["tldr"],
                    "bullets": json.loads(row["bullets"] or "[]"),
                    "abstract": row["abstract"],
                }
                return {"doc_id": doc_id, "summary": summary[args.style], "style": args.style, "source": "cache"}
        except Exception:
            pass  # fall through to live summarization

        # Fetch text and summarize live
        try:
            rows = self._bq.query(
                f"SELECT text FROM `{self._bq.table_ref('documents_clean')}` WHERE doc_id = @doc_id LIMIT 1",
                [__import__("google.cloud.bigquery", fromlist=["ScalarQueryParameter"])
                 .ScalarQueryParameter("doc_id", "STRING", doc_id)],
            )
            if not rows:
                return _err(f"doc_id '{doc_id}' not found", "Use search_documents to find valid doc_ids")

            result = self._summarizer.summarize(doc_id, rows[0]["text"])
            summary_map = {"tldr": result.tldr, "bullets": result.bullets, "abstract": result.abstract}
            return {"doc_id": doc_id, "summary": summary_map[args.style], "style": args.style, "source": "live"}
        except Exception as e:
            logger.error("summarize_document failed", extra={"doc_id": doc_id, "error": str(e)})
            return _err(str(e), "Summarization failed — try again")

    # ── Tool 4: aggregate_entities ────────────────────────────────────────────

    def aggregate_entities(self, doc_ids: list[str], top_n: int = 10) -> dict:
        try:
            args = AggregateEntitiesInput(doc_ids=doc_ids, top_n=top_n)
        except Exception as e:
            return _err(str(e), "Provide a list of doc_id strings")

        if not args.doc_ids:
            return _err("doc_ids list is empty", "Provide at least one doc_id")

        try:
            id_list = ", ".join(f"'{d}'" for d in args.doc_ids)
            sql = f"""
                SELECT entity_name, entity_type, sentiment_score
                FROM `{self._bq.table_ref('entities')}`
                WHERE doc_id IN ({id_list})
                AND extractor = 'cloud_nl'
            """
            rows = self._bq.query(sql)

            if not rows:
                return {"message": "No entity data found. Run extraction first.", "top_entities": []}

            freq: Counter = Counter()
            sentiment_sum: dict[str, float] = defaultdict(float)
            entity_type_map: dict[str, str] = {}

            for r in rows:
                name = r["entity_name"]
                freq[name] += 1
                sentiment_sum[name] += r.get("sentiment_score") or 0.0
                entity_type_map[name] = r.get("entity_type", "OTHER")

            top_entities = [
                {
                    "name": name,
                    "type": entity_type_map[name],
                    "frequency": count,
                    "avg_sentiment": round(sentiment_sum[name] / count, 3),
                }
                for name, count in freq.most_common(args.top_n)
            ]

            sentiment_dist = {
                "positive": sum(1 for s in sentiment_sum.values() if s > 0.1),
                "negative": sum(1 for s in sentiment_sum.values() if s < -0.1),
                "neutral": sum(1 for s in sentiment_sum.values() if -0.1 <= s <= 0.1),
            }

            return {
                "doc_count": len(args.doc_ids),
                "top_entities": top_entities,
                "sentiment_distribution": sentiment_dist,
            }
        except Exception as e:
            logger.error("aggregate_entities failed", extra={"error": str(e)})
            return _err(str(e), "BQ query failed")

    # ── Tool 5: compare_summaries ─────────────────────────────────────────────

    def compare_summaries(self, doc_ids: list[str], focus: str) -> dict:
        try:
            args = CompareSummariesInput(doc_ids=doc_ids, focus=focus)
        except Exception as e:
            return _err(str(e), "Provide doc_ids list and a focus string")

        summaries: list[str] = []
        for doc_id in args.doc_ids[:8]:  # cap at 8 to manage context length
            result = self.summarize_document(doc_id, style="abstract")
            if "error" not in result:
                summaries.append(f"[{doc_id}]: {result['summary']}")

        if not summaries:
            return _err("Could not retrieve summaries for any provided doc_ids")

        combined = "\n\n".join(summaries)
        prompt = (
            f"Based on these article summaries, answer the following question:\n\n"
            f"Question: {args.focus}\n\n"
            f"Summaries:\n{combined}\n\n"
            f"Provide a concise, evidence-based answer citing the doc_ids."
        )

        try:
            settings = get_settings()
            model = get_model(settings.gemini_model)
            response = model.generate_content(prompt)
            return {
                "synthesis": response.text,
                "doc_ids_used": [d for d in args.doc_ids[:8]],
            }
        except Exception as e:
            logger.error("compare_summaries failed", extra={"error": str(e)})
            return _err(str(e), "Gemini synthesis failed — try with fewer doc_ids")

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def dispatch(self, tool_name: str, args: dict) -> dict:
        dispatch_map = {
            "search_documents": lambda a: self.search_documents(**a),
            "extract_entities": lambda a: self.extract_entities(**a),
            "summarize_document": lambda a: self.summarize_document(**a),
            "aggregate_entities": lambda a: self.aggregate_entities(**a),
            "compare_summaries": lambda a: self.compare_summaries(**a),
        }
        fn = dispatch_map.get(tool_name)
        if fn is None:
            return _err(f"Unknown tool: {tool_name}", f"Available: {list(dispatch_map)}")
        return fn(args)
