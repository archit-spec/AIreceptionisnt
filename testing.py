from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

qdrant_client = QdrantClient(
    url="bc80cbd1-33ac-4712-b969-f2145ba40aef.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="DeOUSJcHud18mAZzgfVy2f0LsIXkO_Q5bOEH49ABQtlktxpRL5V_8g",
)

def encode_single(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)


def search(query, limit=1):
    query_vector = encode_single(query)
    search_result = qdrant_client.query_points(
    
            collection_name="emergency_instructions",
            query=query_vector.tolist(),
            limit=4
    )
    return search_result

#print all the keys in collections

# Get the collection info to see how many vectors are in the collection
collection_info = qdrant_client.get_collection("intents")
print(f"Number of vectors in the collection: {collection_info.vectors_count}")

print(qdrant_client.get_collections())
print(search("help im having a heart attack"))