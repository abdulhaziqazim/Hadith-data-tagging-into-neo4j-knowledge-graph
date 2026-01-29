import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx

# Neo4j Connection Details
NEO4J_URI = "bolt://localhost:7687"  # Update with your Neo4j URI
NEO4J_USER = "neo4j"  # Update with your username
NEO4J_PASSWORD = "password@dul"  # Update with your password

# Function to execute query
def execute_query(query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        result = session.run(query)
        data = [record.data() for record in result]
    
    driver.close()
    return pd.DataFrame(data)

# Function to upload CSV and import into Neo4j
def upload_csv_to_neo4j(file):
    df = pd.read_csv(file)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Clear previous data
        session.run("MATCH (n) DETACH DELETE n")

        for _, row in df.iterrows():
            query = """
            MERGE (h:Hadith {text: $hadith})
            MERGE (s:Source {name: $source})
            MERGE (b:Book {title: $book_title})
            MERGE (n:Narrator {name: $narrator})
            MERGE (t:Theme {name: $theme})
            MERGE (h)-[:NARRATED_BY]->(n)
            MERGE (h)-[:HAS_THEME]->(t)
            MERGE (s)-[:HAS_BOOK]->(b)
            MERGE (b)-[:CONTAINS_HADITH]->(h)
            """
            session.run(query, hadith=row["Context"], source=row["Source"], book_title=row["Book Title"], narrator=row["Narrator"], theme=row["Theme"])
    
    driver.close()
    return df

# Function to visualize knowledge graph and extract relationships
def visualize_knowledge_graph():
    net = Network(height="500px", width="100%", directed=True)
    G = nx.DiGraph()
    relationships = []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Source)-[:HAS_BOOK]->(b:Book)-[:CONTAINS_HADITH]->(h:Hadith)
            OPTIONAL MATCH (h)-[:NARRATED_BY]->(n:Narrator)
            OPTIONAL MATCH (h)-[:HAS_THEME]->(t:Theme)
            RETURN s.name AS source, b.title AS book, h.text AS hadith, n.name AS narrator, t.name AS theme
        """)
        
        for record in result:
            source = record["source"]
            book = record["book"]
            hadith = record["hadith"]
            narrator = record.get("narrator", "Unknown")  # Safely get 'narrator', default to "Unknown"
            theme = record.get("theme", "Unknown")  # Safely get 'theme', default to "Unknown"

            G.add_node(source, label=source, color="orange")
            G.add_node(book, label=book, color="beige")
            G.add_node(hadith, label=hadith, color="lightblue")
            G.add_node(narrator, label=narrator, color="lightgreen") 
            G.add_node(theme, label=theme, color="lightpink")

            G.add_edge(source, book)
            G.add_edge(book, hadith)
            G.add_edge(hadith, narrator)
            G.add_edge(hadith, theme)

            # Store relationships in a list
            relationships.append((source, "HAS_BOOK", book))
            relationships.append((book, "CONTAINS_HADITH", hadith))
            relationships.append((hadith, "NARRATED_BY", narrator))
            relationships.append((hadith, "HAS_THEME", theme))

    for node in G.nodes:
        net.add_node(node, label=node)
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    net.save_graph("graph.html")

    # Convert relationships list to DataFrame
    df_relationships = pd.DataFrame(relationships, columns=["Node 1", "Relationship", "Node 2"])

    return "graph.html", df_relationships

# Streamlit UI
st.title("Streamlit Neo4j Knowledge Graph and Data Viewer")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df_uploaded = upload_csv_to_neo4j(uploaded_file)
    st.write("### Uploaded Data to Neo4j")
    st.dataframe(df_uploaded)

query = st.text_area("Enter your Cypher Query:", "MATCH (n) RETURN n LIMIT 10")

if st.button("Run Query"):
    df = execute_query(query)
    st.write("### Query Results")
    st.dataframe(df)

# Display Graph
st.subheader("Knowledge Graph Visualization")
graph_html, df_graph_table = visualize_knowledge_graph()
st.components.v1.html(open(graph_html, "r").read(), height=550)

# Display Relationships as Table
st.subheader("Knowledge Graph Relationships Table")
st.dataframe(df_graph_table)
