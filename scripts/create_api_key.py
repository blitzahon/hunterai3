"""Utility to create an API key for a workspace.

Usage: RAG_DB_URL must be set in environment. Run:

    python scripts/create_api_key.py --workspace default --name "My Key"

It will print the plaintext token to store safely.
"""
import argparse
import os

from services.auth import init_db, create_api_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--name", default="default")
    args = parser.parse_args()

    if not os.getenv("RAG_DB_URL"):
        print("RAG_DB_URL not configured. Set it in the environment to use Postgres.")
        return

    init_db()
    token = create_api_key(args.workspace, args.name)
    print("Created API key for workspace=", args.workspace)
    print("Token (store this securely):")
    print(token)


if __name__ == "__main__":
    main()
