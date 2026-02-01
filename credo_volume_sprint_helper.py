# credo_volume_sprint_helper.py - Super efficient helper with Grokipedia + Grok API expansion
# Run examples: python credo_volume_sprint_helper.py --size 5000 --expand-sources --grok-key YOUR_KEY

import json
import random
import argparse
import requests

try:
    from grokipedia_api import GrokipediaClient  # pip install grokipedia-api
except ImportError:
    print("grokipedia-api not installed — falling back to Grok API only")
    GrokipediaClient = None

from beliefs.credo_node_adder import add_nodes_batch

# Defaults...
DEFAULT_SIZE = 10000
DEFAULT_RES_MIN = 0.86
DEFAULT_RES_MAX = 0.96
DEFAULT_OPEN_MIN = 5
DEFAULT_OPEN_MAX = 14

# Initial curated lists (fallback)
INITIAL_TRADITIONS = [...]  # copy from previous
INITIAL_THEMES = [...]  # copy from previous

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-beta"

def fetch_grok_expansion(query, api_key, max_items=100):
    """Grok API for list generation"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "Generate concise lists of mythologies/wisdom traditions or mythological themes. Format as numbered list."},
            {"role": "user", "content": f"{query} Return up to {max_items} items as a simple numbered list."}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }
    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        items = [line.strip()[3:].strip() for line in content.split('\n') if line.strip().startswith(tuple(str(i) for i in range(1, 101)))]
        return items
    except Exception as e:
        print(f"Grok API failed: {e}")
        return []

def fetch_grokipedia_expansion(query, max_items=100):
    """Unofficial Grokipedia API for list expansion"""
    if GrokipediaClient is None:
        print("GrokipediaClient not available — skip")
        return []
    client = GrokipediaClient()
    try:
        results = client.search(query, limit=max_items)
        items = [result["title"] for result in results if "myth" in result["title"].lower() or "theme" in result["title"].lower()]
        return list(set(items))  # dedupe
    except Exception as e:
        print(f"Grokipedia API failed: {e}")
        return []

def expand_lists(use_grok=False, grok_key=None):
    traditions = list(INITIAL_TRADITIONS)
    themes = list(INITIAL_THEMES)

    if use_grok and grok_key:
        print("Expanding via Grok API + Grokipedia...")
        trad_query = "List 100 diverse global mythologies and wisdom traditions."
        theme_query = "List 100 common mythological and philosophical themes across cultures."

        grok_trad = fetch_grok_expansion(trad_query, grok_key)
        grok_theme = fetch_grok_expansion(theme_query, grok_key)

        grokipedia_trad = fetch_grokipedia_expansion("mythologies", max_items=100)
        grokipedia_theme = fetch_grokipedia_expansion("mythological themes", max_items=100)

        traditions.extend(grok_trad + grokipedia_trad)
        themes.extend(grok_theme + grokipedia_theme)

    traditions = sorted(list(set(traditions)))
    themes = sorted(list(set(themes)))

    return traditions, themes

# ... rest of the script (generate_batch, main()) same as previous

if __name__ == "__main__":
    # ... parser same
    args = parser.parse_args()
    trads, thms = expand_lists(use_grok=args.expand_sources, grok_key=args.grok_key)
    # ... generate and add batch