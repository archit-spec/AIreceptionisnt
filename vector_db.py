from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json
import torch

class VectorDB:
    def __init__(self):
        self.client = QdrantClient(
            url="bc80cbd1-33ac-4712-b969-f2145ba40aef.europe-west3-0.gcp.cloud.qdrant.io", 
            port=6333,
            api_key="DeOUSJcHud18mAZzgfVy2f0LsIXkO_Q5bOEH49ABQtlktxpRL5V_8g"
        )
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model and move it to the appropriate device
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = self.model.to(self.device)
        
        self.collection_name = "intents"

    def initialize_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)
        )

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        points = []
        for i, intent in enumerate(data['root']['intents']):
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']

            # Batch encode patterns for efficiency
            embeddings = self.model.encode(patterns, device=self.device, show_progress_bar=True)

            for pattern, embedding in zip(patterns, embeddings):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=embedding.tolist(),
                        payload={
                            "tag": tag,
                            "pattern": pattern,
                            "response": responses[0]
                        }
                    )
                )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query, limit=1):
        query_vector = self.model.encode([query], device=self.device)[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        return search_result

    def get_collections(self):
        return self.client.get_collections()
