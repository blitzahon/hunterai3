"""Tool schemas exposed to Claude in the agent loop."""

RETRIEVE_TOOL = {
    "name": "retrieve",
    "description": (
        "Search the knowledge base for relevant passages. Use this whenever "
        "you need specific facts from the documents rather than general "
        "knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused search query"},
            "top_k": {
                "type": "integer",
                "description": "Number of passages to retrieve",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
