import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class SemanticCodeSearch:
    def __init__(self, persistent_directory = "./data/chroma_db", embedding_model = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client(Settings(
            persistent_directory=persistent_directory,
            anonymized_telemetry=False
        ))

        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Model loaded")

        self.collections = {}

    def _get_collection(self, repo_id):
        if repo_id not in self.collections:
            try:
                self.collections[repo_id] = self.client.get_collection(name=repo_id)
                print(f"Collection loaded for {repo_id}")
            except Exception as e:
                raise ValueError(f"Repository {repo_id} not indexed!")
            
        return self.collections[repo_id]
    
    def search(self, repo_id, query, top_k = 5):
        collection = self._get_collection(repo_id)

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        filtered_functions = []

        for i in range(len(results['ids'][0])):
            filtered_functions.append({
                "name": results["metadatas"][0][i]['name'],
                "signature": results['metadatas'][0][i]['signature'],
                "file_path": results['metadatas'][0][i]['file_path'],
                "start_line": results['metadatas'][0][i]['start_line'],
                "end_line": results['metadatas'][0][i]['end_line'],
                "docstring": results['metadatas'][0][i].get('docstring', ''),
                "relevance_score": 1 - results['distances'][0][i],
                "matched_text": results['documents'][0][i]
            })

        return filtered_functions
    
    def list_indexed_repos(self):
        collections = self.client.list_collections()
        return [col.name for col in collections]