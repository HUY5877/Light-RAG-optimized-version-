import asyncio
import os

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import setup_logger, EmbeddingFunc
from dotenv import load_dotenv
load_dotenv()
WORKING_DIR = "./my_rag_project"
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "123456789"

os.environ["https_proxy"] = "http://localhost:7890"
os.environ["http_proxy"] = "http://localhost:7890"

# Setup logger for LightRAG
setup_logger("lightrag", level="INFO")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-3.5-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("openai_api_key"),
        base_url="https://api.openai.com/v1",
        **kwargs
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # llm_model_func=ollama_model_complete,
        # llm_model_name='deepseek-r1:7b',
        # llm_model_func=llm_model_func,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts=texts,
                embed_model="bge-m3:latest",
            )
        ),
        graph_storage="Neo4JStorage",
        llm_model_max_async=1,#降低并发数量从而防止超出api请求频率
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag
if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    with open("pdf_data.txt", "r+",encoding="utf-8") as f:
        text = f.read()
    rag.insert(text,split_by_character="##########################################################################",
             file_paths="pdf_data.txt"
               )
    # mode = "local"
    #
    # result = rag.query(
    #     "HUY1是谁？",
    #     param=QueryParam(mode=mode,only_need_context=False,top_k=3)
    # )
    # print(result)

