import csv
import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

try:
    from Cheer.config.load_key import load_key
except ModuleNotFoundError:
    from config.load_key import load_key


BASE_DIR = Path(__file__).resolve().parent
COUPLET_CSV_PATH = BASE_DIR / "resource" / "coupletData_top10000.csv"
EMBEDDING_MODEL = os.environ.get(
    "COUPLET_EMBEDDING_MODEL",
    "tongyi-embedding-vision-plus-2026-03-06",
)
_MODEL_TAG = EMBEDDING_MODEL.replace("/", "_").replace(":", "_")
CHROMA_DIR = BASE_DIR / f"chroma_couplet_db_{_MODEL_TAG}"
COLLECTION_NAME = f"couplet_pairs_{_MODEL_TAG}"
# 首次建库最多写入多少条，避免初次运行太慢；需要全量可手动调大到 10000
BOOTSTRAP_LIMIT = int(os.environ.get("COUPLET_BOOTSTRAP_LIMIT", "100"))

_vector_store: Chroma | None = None


class DashScopeVisionTextEmbeddings(Embeddings):
    """使用 DashScope 多模态嵌入模型对纯文本做向量化。"""

    def __init__(self, model: str):
        self.model = model

    @staticmethod
    def _parse_output_embeddings(resp: Any) -> list[list[float]]:
        output = getattr(resp, "output", None)
        if output is None and isinstance(resp, dict):
            output = resp.get("output")
        if isinstance(output, dict):
            items = output.get("embeddings", [])
        else:
            items = getattr(output, "embeddings", [])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import dashscope

        embeddings: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = [{"text": text} for text in batch]
            resp = dashscope.MultiModalEmbedding.call(model=self.model, input=payload)
            if resp.status_code != 200:
                raise ValueError(
                    f"status_code: {resp.status_code}\ncode: {resp.code}\nmessage: {resp.message}"
                )
            embeddings.extend(self._parse_output_embeddings(resp))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _ensure_dashscope_key() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")


def _read_couplet_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with COUPLET_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            upper = (row.get("text1") or "").strip()
            lower = (row.get("text2") or "").strip()
            if upper and lower:
                rows.append({"upper": upper, "lower": lower})
    return rows


def _build_vector_store() -> Chroma:
    _ensure_dashscope_key()
    embedding_model = DashScopeVisionTextEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def _bootstrap_if_empty(store: Chroma) -> None:
    if store._collection.count() > 0:
        return

    rows = _read_couplet_rows()
    if not rows:
        raise ValueError(f"对联数据为空: {COUPLET_CSV_PATH}")

    if BOOTSTRAP_LIMIT > 0:
        rows = rows[:BOOTSTRAP_LIMIT]

    texts = [item["upper"] for item in rows]
    metadatas = [{"lower": item["lower"]} for item in rows]
    ids = [f"couplet-{i}" for i in range(len(rows))]
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
        _bootstrap_if_empty(_vector_store)
    return _vector_store


def _simple_local_retrieve(query: str, k: int) -> list[dict[str, Any]]:
    rows = _read_couplet_rows()
    q = query.strip()
    if not q:
        return []

    scored: list[tuple[int, dict[str, str]]] = []
    q_set = set(q)
    for item in rows:
        upper = item["upper"]
        overlap = len(q_set & set(upper))
        if overlap > 0:
            scored.append((overlap, item))
    scored.sort(key=lambda x: x[0], reverse=True)

    samples: list[dict[str, Any]] = []
    for score, item in scored[:k]:
        samples.append(
            {
                "upper": item["upper"],
                "lower": item["lower"],
                "score": score,
                "source": "local_fallback",
            }
        )
    return samples


def retrieve_couplet_samples(query: str, k: int = 10) -> list[dict[str, Any]]:
    store = get_vector_store()
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception as e:
        fallback = _simple_local_retrieve(query, k)
        if fallback:
            return fallback
        raise RuntimeError(
            "对联检索失败，请检查嵌入模型可用性或余额。"
            f" 当前模型: {EMBEDDING_MODEL}；原始错误: {e}"
        ) from e
    samples: list[dict[str, Any]] = []
    for doc, score in results:
        upper = (doc.page_content or "").strip()
        lower = str(doc.metadata.get("lower", "")).strip()
        if upper and lower:
            samples.append(
                {"upper": upper, "lower": lower, "score": score, "source": "vector"}
            )
    return samples
