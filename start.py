import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc, xml_to_json
from neo4j import  GraphDatabase

WORKING_DIR = "./my_rag_project"
os.makedirs(WORKING_DIR, exist_ok=True)
setup_logger("lightrag", level="INFO")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "123456789"

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name='llama3.2',
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts=texts,
                embed_model="bge-m3:latest",
            )
        )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    rag.insert("HUY1是世界上最帅的人，他codeforces分数是1600分")

    mode = "naive"

    result = rag.query(
        "谁是世界上最帅的人？",
        param=QueryParam(mode=mode)
    )
    print(result)







if __name__ == "__main__":
    main()