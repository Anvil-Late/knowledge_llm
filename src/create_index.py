import qdrant_client as qc
import qdrant_client.http.models as qmodels
import uuid
import json
import argparse
from tqdm import tqdm

client = qc.QdrantClient(url="localhost")
METRIC = qmodels.Distance.DOT
COLLECTION_NAME = "pep_docs"

def create_index():
    client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config = qmodels.VectorParams(
            size=DIMENSION,
            distance=METRIC,
        )
    )


def create_subsection_vector(
    subsection_content,
    section_anchor,
    title
    ):

    id = str(uuid.uuid1().int)[:32]
    payload = {
        "text": subsection_content,
        "title": title,
        "section_anchor": section_anchor,
        "block_type": 'text'
    }
    return id, payload


def add_doc_to_index(embeddings):
    ids = []
    vectors = []
    payloads = []
    
    for section_anchor_pepno, content in tqdm(embeddings.items()):
        title = content['title']
        section_vector = content['embedding']
        section_content = content['content']
        section_content = content['title'] + '\n' + content['content']
        id, payload = create_subsection_vector(
            section_content,
            section_anchor_pepno,
            title
    )
        ids.append(id)
        vectors.append(section_vector)
        payloads.append(payload)

        # Add vectors to collection
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=qmodels.Batch(
                ids = [id],
                vectors=[section_vector],
                payloads=[payload]
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', type=str, default='instructor')
    parser.add_argument('--embeddings_path', type=str, default='./embeddings/instructor_embeddings.json')
    args = parser.parse_args()

    if args.embedder == 'openai':
        DIMENSION = 1536
    else:
        DIMENSION = 768
    
    with open(args.embeddings_path, 'r') as f:
        embeddings = json.load(f)
    
    create_index()
    add_doc_to_index(embeddings)