
def fuzzy_merge_nodes(nodes, global_threshold=85):
    groups = defaultdict(list)
    for node in nodes:
        tags = node["properties"].get("tags", [])
        trad_tag = next((t for t in tags if "tradition" in t.lower() or t in INITIAL_TRADITIONS), None)
        trad = trad_tag or node.get("name", "").split()[0]
        groups[trad].append(node)

    merged = []
    archive = []

    # Exclusion list — protected rare traditions
    RARE_EXCLUDE = [
        "Dogon", "Hopi", "Inuit", "Basque", "Sedna", "Nommo", "Kachina", "Mari", "Vahagn",
        "Louhi", "Boldogasszony", "Tengri", "Baiame", "Anansi", "Pele"
    ]

    for trad, group_nodes in groups.items():
        group_size = len(group_nodes)
        threshold = global_threshold

        # Apply exclusion list first (overrides other thresholds)
        if any(ex.lower() in trad.lower() for ex in RARE_EXCLUDE):
            threshold = 99
            logger.info(f"Excluded rare tradition {trad} — merge disabled (threshold 99)")
        # Then apply size-based protection
        elif group_size < 30:
            threshold = 97
            logger.debug(f"Rare/small {trad} ({group_size}) → threshold {threshold}")
        elif group_size < 100:
            threshold = min(94, global_threshold + 5)
            logger.debug(f"Small {trad} ({group_size}) → threshold {threshold}")

        # ... rest of the merge loop remains the same
        # (loop over group_nodes, compare to merged, merge if score >= threshold, etc.)