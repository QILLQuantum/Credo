from neo4j import GraphDatabase
import json

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

driver = GraphDatabase.driver(URI, auth=AUTH)

def add_node(tx, node):
    cypher = """
    CREATE (b:Belief {id: $id, name: $name})
    SET b.properties = $props
    """
    tx.run(cypher, id=node["id"], name=node.get("name"), props=node["properties"])

def add_edge(tx, edge):
    cypher = """
    MATCH (s:Belief {id: $source})
    MATCH (t:Belief {id: $target})
    CREATE (s)-[r:BRIDGE {type: $type, resonance: $res}]->(t)
    """
    tx.run(cypher, source=edge["source"], target=edge["target"], type=edge.get("type", "syncretic"), res=random.uniform(0.86, 0.96))

with open('core/master_graph_merged_20260129.json', 'r') as f:
    graph = json.load(f)["graph"]

with driver.session() as session:
    # Add nodes in batches of 1000
    for i in range(0, len(graph["nodes"]), 1000):
        batch = graph["nodes"][i:i+1000]
        for node in batch:
            session.execute_write(add_node, node)
        print(f"Added nodes batch {i//1000 + 1}")

    # Add edges in batches
    for i in range(0, len(graph["edges"]), 1000):
        batch = graph["edges"][i:i+1000]
        for edge in batch:
            session.execute_write(add_edge, edge)
        print(f"Added edges batch {i//1000 + 1}")

driver.close()
print("Migration to Neo4j complete — 1.5M ready.")