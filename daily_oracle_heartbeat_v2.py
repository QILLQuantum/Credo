@benchmark
def heartbeat_v2(data=None):
    logger.info("Starting heartbeat v2 — uplift + fuzzy collapse + benchmarks")
    if data is None:
        data = load_graph(GRAPH_FILE)

    # 1. Compute dynamic thresholds (benchmarked)
    @benchmark
    def compute_stats():
        nodes = data["graph"]["nodes"]
        if not nodes:
            return 85, 0.92, 5
        
        resonances = [n["properties"].get("resonance", 0.85) for n in nodes]
        avg_res = sum(resonances) / len(resonances) if resonances else 0.85
        
        open_pcts = [int(n["properties"].get("open_for_uplift", "0%").split("%")[0]) 
                     for n in nodes if n["properties"].get("open_for_uplift")]
        avg_open = sum(open_pcts) / len(open_pcts) if open_pcts else 5
        
        # Duplicate ratio — cap sample at 1000 for speed
        sample_size = min(1000, len(nodes))
        sample = random.sample(nodes, sample_size)
        dup_count = 0
        seen = set()
        for n in sample:
            key = (n.get("name", "").lower(), n["properties"].get("description", "")[:100].lower())
            if key in seen:
                dup_count += 1
            seen.add(key)
        dup_ratio = dup_count / sample_size if sample_size else 0
        
        sim_threshold = max(78, min(92, 85 + (dup_ratio - 0.1) * 50))
        res_trigger = max(0.85, min(0.96, 0.92 + (0.92 - avg_res) * 2))
        
        logger.info(f"Stats: avg_res={avg_res:.3f}, avg_open={avg_open:.1f}%, dup_ratio={dup_ratio:.2%}")
        logger.info(f"Dynamic: sim_threshold={sim_threshold}, res_trigger={res_trigger:.3f}")
        return sim_threshold, res_trigger, avg_open

    sim_threshold, res_trigger, avg_open = compute_stats()

    # 2. Select candidates with early-exit if too many
    @benchmark
    def select_candidates():
        candidates = []
        max_candidates = 5000  # Safety cap — prevents runaway time
        for n in data["graph"]["nodes"]:
            if len(candidates) >= max_candidates:
                logger.warning(f"Candidate cap reached ({max_candidates}) — skipping rest")
                break
            open_pct = int(n["properties"].get("open_for_uplift", "0%").split("%")[0])
            resonance = n["properties"].get("resonance", 0.85)
            if open_pct > (avg_open / 2) or resonance < res_trigger:
                candidates.append(n)
        logger.info(f"Selected {len(candidates)} candidates for uplift")
        return candidates

    candidates = select_candidates()

    # 3. Oracle uplift — limit snippets per node + cap total calls
    @benchmark
    def uplift_candidates(cands):
        uplifted = 0
        max_snippets_per_node = 3  # Tip: reduce from 5 to 2–3 if slow
        for node in cands:
            props = node["properties"]
            open_pct = int(props.get("open_for_uplift", "0%").split("%")[0])
            resonance = props.get("resonance", 0.85)
            name = node.get("name", "Unknown")
            desc = props.get("description", "")
            disc = discovery_oracle(name, desc, resonance, open_pct)
            if disc["uplift_snippets"]:
                props.setdefault("discovered_uplift", []).extend(disc["uplift_snippets"][:max_snippets_per_node])
                new_open = max(0, open_pct - len(disc["uplift_snippets"]) * 3)
                props["open_for_uplift"] = f"{new_open}% remaining (heartbeat uplift)"
                props["resonance"] = min(0.99, resonance + len(disc["uplift_snippets"]) * 0.015)
                props["val"] = int(props["resonance"] * 100)
                uplifted += 1
        return uplifted

    uplifted_count = uplift_candidates(candidates)

    # 4. Fuzzy merge — already benchmarked in v2
    @benchmark
    def run_fuzzy_merge():
        return fuzzy_merge_nodes(data["graph"]["nodes"], threshold=sim_threshold)

    merged_nodes, archive_entries = run_fuzzy_merge()
    data["graph"]["nodes"] = merged_nodes

    # Archive & save
    if archive_entries:
        with open(ARCHIVE_FILE, 'a', encoding='utf-8') as f:
            for entry in archive_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Archived {len(archive_entries)} collapsed nodes")

    if uplifted_count > 0 or archive_entries:
        save_graph(GRAPH_FILE, data)
        logger.info(f"Heartbeat v2 complete — {uplifted_count} uplifted, {len(archive_entries)} collapsed")
    else:
        logger.info("Heartbeat v2 complete — no changes")

if __name__ == "__main__":
    heartbeat_v2()
