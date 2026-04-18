from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointsSelector,
    FilterSelector,
    PayloadSchemaType,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from tools.embeddings import embeddings
from typing import List
import os
import dotenv
dotenv.load_dotenv()

# Collection name constant
COLLECTION_NAME = "test"
USER_ID_PAYLOAD_FIELD = "metadata.user_id"


def _get_env(*keys: str) -> str | None:
    """Return the first non-empty environment value from keys."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None

# 1. Initialize ONLY the Cloud Client
qdrant_url = _get_env("QDRANT_CLIENT_URL", "QdrantClient_url")
qdrant_api_key = _get_env("QDRANT_CLIENT_API_KEY", "QdrantClient_api_key")

if not qdrant_url:
    raise RuntimeError(
        "Missing Qdrant URL. Set QDRANT_CLIENT_URL in backend/.env (or env vars in Docker)."
    )

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# 2. Embedding vector size (override via env if needed)
# gemini-embedding-001 defaults to 3072 dimensions.
vector_size = int(os.getenv("EMBEDDING_VECTOR_SIZE", "3072"))

# 3. Create or recreate the collection ON THE CLOUD
# Check if existing collection has wrong dimensions and recreate if needed
needs_recreate = False
if client.collection_exists(COLLECTION_NAME):
    collection_info = client.get_collection(COLLECTION_NAME)
    existing_size = collection_info.config.params.vectors.size
    if existing_size != vector_size:
        print(f"⚠️ Collection has {existing_size} dimensions but embeddings are {vector_size}. Recreating...")
        needs_recreate = True
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists on Cloud.")
else:
    needs_recreate = True
    print(f"Creating collection '{COLLECTION_NAME}' on Cloud...")

if needs_recreate:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created with {vector_size} dimensions.")

# 4. Initialize Vector Store using the SAME client
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

def _payload_schema_type_value(field_schema) -> str:
    """Extract payload field type from Qdrant schema objects/dicts across client versions."""
    if field_schema is None:
        return ""
    if isinstance(field_schema, dict):
        data_type = field_schema.get("data_type") or field_schema.get("type")
    else:
        data_type = getattr(field_schema, "data_type", None) or getattr(field_schema, "type", None)
    return str(data_type).lower() if data_type is not None else ""


def has_user_id_payload_index() -> bool:
    """Return True when metadata.user_id is indexed as a keyword payload field."""
    collection_info = client.get_collection(COLLECTION_NAME)
    payload_schema = getattr(collection_info, "payload_schema", None) or {}
    field_schema = payload_schema.get(USER_ID_PAYLOAD_FIELD)
    field_type = _payload_schema_type_value(field_schema)
    return field_type.endswith("keyword")


def ensure_user_id_payload_index() -> None:
    """Create and verify payload index used by user-scoped filtering."""
    if has_user_id_payload_index():
        print(f"Payload index already exists for {USER_ID_PAYLOAD_FIELD}")
        return

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=USER_ID_PAYLOAD_FIELD,
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    if not has_user_id_payload_index():
        raise RuntimeError(
            f"Failed to verify payload index for {USER_ID_PAYLOAD_FIELD}. "
            "Filtered searches will fail until the index exists."
        )

    print(f"Created payload index for {USER_ID_PAYLOAD_FIELD}")


# 5. Ensure payload index for user_id filtering (required by Qdrant for filtered searches)
ensure_user_id_payload_index()

print("Vector Store successfully connected to Cloud!")


# ============================================
# USER-SCOPED VECTOR STORE FUNCTIONS
# ============================================

def add_documents_for_user(documents: List[Document], user_id: str) -> List[str]:
    """
    Add documents to vector store with user_id in metadata for isolation.
    Each document's metadata is updated to include the user_id.
    """
    for doc in documents:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["user_id"] = user_id
    
    document_ids = vector_store.add_documents(documents=documents)
    print(f"✅ Added {len(documents)} documents for user: {user_id}")
    return document_ids


def search_for_user(query: str, user_id: str, k: int = 4) -> List[Document]:
    """
    Search vector store with user_id filter for isolation.
    Only returns documents that belong to the specified user.
    """
    user_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )
    
    try:
        results = vector_store.similarity_search(
            query=query,
            k=k,
            filter=user_filter
        )
    except Exception as e:
        # Self-heal once if deployment/environment race caused missing payload index.
        if "index required but not found" in str(e).lower() and USER_ID_PAYLOAD_FIELD in str(e):
            ensure_user_id_payload_index()
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=user_filter
            )
        else:
            raise
    print(f"🔍 Found {len(results)} documents for user: {user_id}")
    return results


def delete_user_documents(user_id: str) -> bool:
    """
    Delete all documents belonging to a specific user.
    Useful for account cleanup or data reset.
    """
    user_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )
    
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=user_filter)
        )
        print(f"🗑️ Deleted all documents for user: {user_id}")
        return True
    except Exception as e:
        print(f"❌ Error deleting documents for user {user_id}: {e}")
        return False


def clear_collection() -> bool:
    """
    Delete ALL documents from the collection.
    Use this once to clear legacy data without user_id.
    WARNING: This will delete everything!
    """
    try:
        # Delete and recreate collection to clear all data
        client.delete_collection(collection_name=COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        ensure_user_id_payload_index()
        print(f"🗑️ Cleared entire collection: {COLLECTION_NAME}")
        return True
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        return False