import os
from src.parsing import Parser
from src.text_splitter import split_page
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from api import get_embeddings


sved_base_url = "https://www.hse.ru/sveden/"
sved_names = ["vacant", "common", "managers", "struct", "employees", "budget", "catering"]
urls = ["https://ba.hse.ru/result2025", "https://ba.hse.ru/price#pagetop", "https://ba.hse.ru/discount", "https://ba.hse.ru/dost",
        "https://ba.hse.ru/information", "https://ba.hse.ru/intexam"]
for sved_name in sved_names:
    urls.append(sved_base_url + sved_name)

# Парсинг
parser = Parser(crawl_delay=1.0, include_tables=True, output_format="markdown")
parsed = parser.parse_urls(urls)

# Чанкинг
all_chunks = []
for page in parsed:
    chunks = split_page(page)
    all_chunks.extend(chunks)


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

EMB_DIM = 3072


COLLECTION = "abitu_bot_db2"
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
)

BATCH_EMB_SIZE = 16
BATCH_QDRANT_SIZE = 16

points = []

for start in range(0, len(all_chunks), BATCH_EMB_SIZE):
    batch_chunks = all_chunks[start:start + BATCH_EMB_SIZE]
    texts = [ch["text"] for ch in batch_chunks]
    embeddings = get_embeddings(texts)

    for i, vec in enumerate(embeddings):
        global_idx = start + i
        chunk = batch_chunks[i]
        points.append(
            PointStruct(
                id=global_idx,
                vector=vec,
                payload={
                    "text": chunk["text"],
                    "url": chunk.get("url")
                },
            )
        )
        if len(points) >= BATCH_QDRANT_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

if points:
    client.upsert(collection_name=COLLECTION, points=points)
