from api import get_embeddings
from qdrant_client import QdrantClient

COLLECTION = "abitu_bot_db2"

client = QdrantClient(
    url="https://6a1eb5a2-ba09-4f61-ace2-342159de637e.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0Hht7lNOiPM_5B2Nf2oREgYC808e3LjrVzh5PQb5sn4",
)

query = "Какой проходной бал был на Прикладную математику и информатику в 2025"
q_vec = get_embeddings([query])[0]
hits = client.query_points(collection_name=COLLECTION, query=q_vec, limit=3, with_payload=True)

for hit in hits:
    print(hit)