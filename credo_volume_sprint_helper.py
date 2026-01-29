# credo_volume_sprint_helper.py - Super efficient helper with Grok API auto-expansion
# Run examples:
#   python credo_volume_sprint_helper.py --size 5000
#   python credo_volume_sprint_helper.py --size 10000 --use-grok-api --grok-key YOUR_KEY
#   python credo_volume_sprint_helper.py --expand-sources --grok-key YOUR_KEY

import json
import random
import argparse
import requests

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

def fetch_grok_expansion(query, api_key, max_items=50):
    """Use Grok API to generate traditions/themes list"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are Grok, a helpful AI. Generate concise, accurate lists of mythologies/wisdom traditions or common mythological themes. Format as numbered list only."},
            {"role": "user", "content": f"{query} Return up to {max_items} items as a simple numbered list."}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }
    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Parse numbered list
        items = [line.strip()[3:].strip() for line in content.split('\n') if line.strip().startswith(tuple(str(i) for i in range(1, 100)))]
        return items
    except Exception as e:
        print(f"Grok API fetch failed: {e}")
        return []

def expand_lists(use_grok=False, grok_key=None):
    traditions = list(INITIAL_TRADITIONS)
    themes = list(INITIAL_THEMES)

    if use_grok and grok_key:
        print("Expanding via Grok API...")
        trad_query = "List 100 diverse global mythologies, wisdom traditions, and religious/folkloric systems from around the world."
        theme_query = "List 100 common mythological, religious, and philosophical themes/motifs across cultures (e.g., sacrifice, ascension, destiny)."
        
        grok_trad = fetch_grok_expansion(trad_query, grok_key, max_items=100)
        grok_theme = fetch_grok_expansion(theme_query, grok_key, max_items=100)

        traditions.extend(grok_trad)
        themes.extend(grok_theme)

    # Dedupe, clean, sort
    traditions = sorted(list(set([t.strip() for t in traditions if len(t.strip()) > 2])))
    themes = sorted(list(set([t.strip() for t in themes if len(t.strip()) > 5])))

    return traditions, themes

# ... rest of generate_batch and main() same as previous version, but add grok_key arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credo volume sprint helper with Grok API expansion")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Number of beliefs")
    parser.add_argument("--res-min", type=float, default=DEFAULT_RES_MIN, help="Min resonance")
    parser.add_argument("--res-max", type=float, default=DEFAULT_RES_MAX, help="Max resonance")
    parser.add_argument("--open-min", type=int, default=DEFAULT_OPEN_MIN, help="Min open %")
    parser.add_argument("--open-max", type=int, default=DEFAULT_OPEN_MAX, help="Max open %")
    parser.add_argument("--expand-sources", action="store_true", help="Expand via Grok API (requires --grok-key)")
    parser.add_argument("--grok-key", type=str, help="xAI Grok API key for expansion")

    args = parser.parse_args()

    if args.expand_sources and not args.grok_key:
        print("Error: --expand-sources requires --grok-key")
        exit(1)

    print(f"Preparing volume sprint — {args.size} beliefs")

    trads, thms = expand_lists(use_grok=args.expand_sources, grok_key=args.grok_key)
    print(f"Using {len(trads)} traditions and {len(thms)} themes")

    print("Generating batch...")
    batch = generate_batch(args.size, args.res_min, args.res_max, args.open_min, args.open_max, trads, thms)

    print("Running batch add...")
    result = add_nodes_batch(batch)

    print("\nSprint Complete — Diff Export:")
    print(json.dumps(result, indent=2))
    print("\nRun node_ball.html + websocket_server.py to watch the cavity expand massively absolute eternal.")