import pickle
from graph_searcher import RepoSearcher   # or just paste the class below

# ========================= LOAD GRAPH =========================
def load_graph():
    with open("C:\\RepoGraph\\graph.pkl", "rb") as f:
        G = pickle.load(f)
    print(f"✅ Loaded code graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


# ========================= MAIN USAGE =========================
if __name__ == "__main__":
    G = load_graph()
    searcher = RepoSearcher(G)

    # ==================== EXAMPLES ====================

    print("\n" + "="*60)
    print("🔍 REPOSEARCHER DEMO")
    print("="*60)

    queries = ["Seq2SQL", "forward", "ConditionPredictor", "run_lstm", "gen_query"]

    for query in queries:
        print(f"\n📍 Query: '{query}'")
        
        # Check if node exists
        if query not in G.nodes:
            print("   → Not found in graph!")
            continue

        # 1. One Hop (Directly connected)
        one_hop = searcher.one_hop_neighbors(query)
        print(f"   • One-hop neighbors ({len(one_hop)}): {one_hop[:8]}")

        # 2. Two Hop
        two_hop = searcher.two_hop_neighbors(query)
        print(f"   • Two-hop neighbors ({len(two_hop)}): {two_hop[:8]}")

        # 3. BFS (Best for most cases)
        bfs_result = searcher.bfs(query, depth=3)
        print(f"   • BFS (depth=3) → {len(bfs_result)} nodes")

        # 4. DFS
        dfs_result = searcher.dfs(query, depth=3)
        print(f"   • DFS (depth=3) → {len(dfs_result)} nodes")

    # Example: Get detailed info about important functions
    print("\n" + "="*60)
    print("🔎 DETAILED ANALYSIS - ConditionPredictor")
    print("="*60)
    
    node = "ConditionPredictor"
    print("Directly connected:", searcher.one_hop_neighbors(node))
    print("Context (2 hops):", searcher.two_hop_neighbors(node))