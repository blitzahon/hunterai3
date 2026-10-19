import os
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.getenv("EMBEDDING_LOCATION", "us-central1"),
)

for name in os.getenv("TRY_MODELS", "text-embedding-005").split(","):
    name = name.strip()
    try:
        r = client.models.embed_content(
            model=name,
            contents=["hello world"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        print("OK  ", name, "-> dimensions:", len(r.embeddings[0].values))
    except Exception as e:
        print("FAIL", name, "->", str(e)[:150])