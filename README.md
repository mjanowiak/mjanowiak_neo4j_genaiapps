# mjanowiak_neo4j_genaiapps

This project contains code for a Neo4j LangChain chatbot using a Star Wars dataset from Kaggle.

## Data

- Locally, I have a `data/` folder that contains the download from [The Star Wars Dataverse](https://www.kaggle.com/datasets/jsphyg/star-wars).

## Notebooks

- `starwars_neo4j_chat.ipynb` is the main notebook. It loads data into Neo4j and builds a LangChain chatbot on top of the Neo4j database.

## Cypher Queries

- AI generates Cypher queries well, often better than SQL. In this example, we have several "pre-baked" Cypher queries to control the AI and answer typical questions a person might ask. We wrapped them in tools so that the agent can determine which query to use. I have personally used this same approach with DuckDB, Athena/Glue, and Cypher, which all work well to provide agent context without a full RAG setup. Similarity search might be the next version of this example, but it would require vectorization.

```python
tools = [
    character_details,
    planet_details,
    species_by_planet,
    characters_by_planet,
    inspect_schema,
    natural_language_to_cypher,
]
```

- The tools are broken up by how a user might ask a question, with an important fallback `natural_language_to_cypher` tool that converts natural language to a Cypher query. This is the weak link in any natural-language-to-query system. It has minimal string cleaning in the tool function, but that was AI-generated and can likely be enhanced with a "compiler"-type cleaning function to ensure the string meets appropriate Cypher syntax.