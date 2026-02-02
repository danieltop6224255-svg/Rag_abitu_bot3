import yaml
from pathlib import Path
from api import get_model_answer, get_embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


qas_path = "..\data\qa.yaml"
COLLECTION = "abitu_bot_db2"

client = QdrantClient(
    url="https://6a1eb5a2-ba09-4f61-ace2-342159de637e.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0Hht7lNOiPM_5B2Nf2oREgYC808e3LjrVzh5PQb5sn4",
)

qas_text = Path(qas_path).read_text(encoding="utf-8")
qas = yaml.safe_load(qas_text)["easy_qa"]


for qa in qas:
    question = qa["q"]
    true_answer = qa["a"]

    q_vec = get_embeddings([question])[0]
    points = client.query_points(collection_name=COLLECTION, query=q_vec, limit=3, with_payload=True).points
    relevant_chunks = [point.payload["text"] for point in points]

    # print(question)
    # print("-" * 50)
    # print(('\n' + "*"*50 + '\n').join(relevant_chunks))
    # print()
    # print()

    model_ans = get_model_answer(question, relevant_chunks)
    print(question)
    print("-" * 50)
    print(model_ans)
    print("-" * 50)
    print(true_answer)
    print()
    print()
    print()