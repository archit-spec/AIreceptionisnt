from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="bc80cbd1-33ac-4712-b969-f2145ba40aef.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="DeOUSJcHud18mAZzgfVy2f0LsIXkO_Q5bOEH49ABQtlktxpRL5V_8g",
)

print(qdrant_client.get_collections())