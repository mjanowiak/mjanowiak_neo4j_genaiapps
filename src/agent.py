from typing import Any
import json
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
# Langchain version of the query
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(
    uri=NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=DATABASE
)

# Get my Graph Connection
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=DATABASE
)

def run_cypher(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def _json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, default=str, ensure_ascii=False)


@tool
def character_details(character_name: str) -> str:
    """Retrieve detailed information about a specific Star Wars character by exact name."""
    query = """
    MATCH (c:Characters {name: $characterName})
    RETURN c.name AS name,
           c.gender AS gender,
           c.skin_color AS skin_color,
           c.height AS height,
           c.weight AS weight,
           c.species AS species,
           c.homeworld AS homeworld,
           c.year_born AS yearBorn,
           c.year_died AS yearDied,
           c.description AS description
    """
    rows = run_cypher(query, {"characterName": character_name})
    return _json(rows)


@tool
def planet_details(planet_name: str) -> str:
    """Retrieve planet details by exact planet name."""
    query = """
    MATCH (p:Planets {name: $planetName})
    RETURN p.name AS name,
           p.climate AS climate,
           p.terrain AS terrain,
           p.population AS population,
           p.description AS description
    """
    rows = run_cypher(query, {"planetName": planet_name})
    return _json(rows)


@tool
def species_by_planet(planet_name: str) -> str:
    """List indigenous species from a specific planet by exact planet name."""
    query = """
    MATCH (s:Species)-[:ORIGINATED_FROM]->(p:Planets {name: $planetName})
    RETURN p.name AS planet,
           s.name AS species,
           s.classification AS classification,
           s.average_lifespan AS average_lifespan
    ORDER BY s.name
    """
    rows = run_cypher(query, {"planetName": planet_name})
    return _json(rows)


@tool
def characters_by_planet(planet_name: str) -> str:
    """List characters from a specific planet and the indigenous species there, by exact planet name."""
    query = """
    MATCH (c:Characters)-[:FROM]->(p:Planets {name: $planetName})
    OPTIONAL MATCH (s:Species)-[:ORIGINATED_FROM]->(p)
    RETURN c.name AS character,
           p.name AS homeworld,
           collect(DISTINCT s.name) AS indigenous_species
    ORDER BY c.name
    """
    rows = run_cypher(query, {"planetName": planet_name})
    return _json(rows)


@tool
def inspect_schema(_: str = "") -> str:
    """Return current Neo4j graph schema for Star Wars graph."""
    graph.refresh_schema()
    return graph.get_schema


@tool
def natural_language_to_cypher(question: str) -> str:
    """Translate natural language questions to Cypher when no specialized tool fits, execute the query, and return rows."""
    graph.refresh_schema()
    schema_text = graph.get_schema

    translation_prompt = (
        "You convert natural language into Cypher for a Star Wars Neo4j graph. "
        "Return only a single Cypher query with no markdown fences and no explanation. "
        "Use only labels/relationships/properties that exist in schema. "
        "Prefer MATCH/OPTIONAL MATCH/WHERE/RETURN/ORDER BY/LIMIT.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{question}"
    )

    cypher_text = llm.invoke(
        [
            SystemMessage(content="You are an expert Cypher translator."),
            HumanMessage(content=translation_prompt),
        ]
    ).content.strip()

    if cypher_text.startswith("```"):
        cypher_text = cypher_text.strip("`")
        cypher_text = cypher_text.replace("cypher", "", 1).strip()

    print("\n[NL -> Cypher translation]")
    print(f"Question: {question}")
    print("Generated Cypher:")
    print(cypher_text)

    rows = run_cypher(cypher_text)
    preview = rows[:5]
    print(f"Rows returned: {len(rows)}")
    if preview:
        print("Preview:")
        print(json.dumps(preview, default=str, indent=2, ensure_ascii=False))

    return _json(rows)


tools = [
    character_details,
    planet_details,
    species_by_planet,
    characters_by_planet,
    inspect_schema,
    natural_language_to_cypher,
]

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

assistant_system_prompt = """You are a helpful assistant for querying information from the Star Wars graph.
Provide accurate and concise answers based on the available data.
Use the specialized tools whenever possible.
If a question does not cleanly match a specialized tool, use natural_language_to_cypher.
Do not invent facts that are not returned by tools.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=assistant_system_prompt,
)


def _extract_last_text(agent_output: Any) -> str:
    if isinstance(agent_output, dict) and "messages" in agent_output and agent_output["messages"]:
        last_msg = agent_output["messages"][-1]
        content = getattr(last_msg, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return str(agent_output)


def ask_starwars(question: str) -> str:
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = _extract_last_text(response)
    print("\n[Agent answer]")
    print(answer)
    return answer