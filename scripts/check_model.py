import os
from google import genai

# Use 'global' as a safe default for Gemini 3.8 models on Vertex AI
location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
project = os.environ.get("GOOGLE_CLOUD_PROJECT", "hunter-ai-512006")

client = genai.Client(
    vertexai=True,
    project=project,
    location=location,
)

# Read model list from environment or default to gemini-3.8-flash
try_models_str = os.getenv("TRY_MODELS", "gemini-3.8-flash")

for name in try_models_str.split(","):
    name = name.strip()
    if not name:
        continue

    try:
        # Standard generate_content call
        r = client.models.generate_content(
            model=name,
            contents="Reply with the single word: ok"
        )
        print("OK  ", name, "->", (r.text or "").strip())
    except Exception as e:
        print("FAIL", name, "->", str(e).strip())
