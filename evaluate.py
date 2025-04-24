import asyncio
import csv
import io

import numpy as np
from boto3 import client
from lightrag import QueryParam
from lightrag.utils import setup_logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from startwith_neo4j import initialize_rag
from query_answer import test_query
import ollama
# def compute_similarity(expected, retrieved):
#     """计算文本相似度"""
#     vectorizer = TfidfVectorizer()
#     tfidf = vectorizer.fit_transform([expected, retrieved])
#     similarity_matrix = cosine_similarity(tfidf, tfidf)
#     return similarity_matrix[0, 1]
setup_logger("lightrag", level="INFO")
def compute_similarity(expected, retrieved):
    """计算预期答案与检索结果的相似度"""
    embedding_model = ollama.Client(host="127.0.0.1:11434")

    expected_embedding = embedding_model.embed(model="bge-m3:latest", input=expected).embeddings
    retrieved_embedding = embedding_model.embed(model="bge-m3:latest", input=retrieved).embeddings
    # print(expected_embedding,type(expected_embedding),sep="\n")
    return cosine_similarity(expected_embedding, retrieved_embedding)[0][0]

#计算精确率
def calculate_precision(retrieved, relevant, threshold=0.7):
    """计算精确率，当相似度超过阈值的时候就认为是正确的"""
    correct = 0
    for r in retrieved:
        for rel in relevant:
            similarity = compute_similarity(rel, r)
            if similarity >= threshold:
                correct += 1
                break
    return correct / len(relevant) if relevant else 0

#计算召回率
def calculate_recall(retrieved, relevant, threshold=0.7):
    """计算召回率，当相似度超过阈值的时候就认为是正确的"""
    correct = 0
    for rel in relevant:
        for r in retrieved:
            similarity = compute_similarity(rel, r)
            if similarity >= threshold:
                correct += 1
                break
    return correct / len(relevant) if relevant else 0




#精确率是在全部的检索结果中看有多少是正确的检索结果
#召回率是看在所有的相关文档中，有多少是跟检索结果匹配的
#他们的分母不一样

def calculate_f1(precision, recall):
    """计算F1值"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def str_to_list_with_highlevel(results : str):
    #定位最后一个模块
    sources_start_marker = "-----Sources-----"
    sources_start_index = results.find(sources_start_marker)
    #定位csv block
    csv_start_marker = "```csv\n"
    csv_block_start_index = results.find(csv_start_marker, sources_start_index)
    csv_end_marker = "\n```"
    csv_block_end_index = results.find(csv_end_marker, csv_block_start_index + len(csv_start_marker))

    #提取csv字符串
    csv_data_string = results[csv_block_start_index + len(csv_start_marker) : csv_block_end_index].strip()

    #解析csv
    csv_file = io.StringIO(csv_data_string)

    csv_reader = csv.DictReader(csv_file)
    content_list = []
    for row in csv_reader:
        if 'content' in row:
            if row['content'] != None:
                content_list.append(row['content'])
        else:
            print(f"Warning: 'content' column not found in row: {row}")
    return content_list

def str_to_list_with_lowlevel(formatted_string: str) -> list[str]:
    """
    从特定格式的字符串中提取每个 chunk 的 content 内容。

    Args:
        formatted_string: 包含多个 chunk 的格式化字符串。

    Returns:
        一个包含每个 chunk content 的字符串列表。
    """
    contents = []
    # 1. 按 "--New Chunk--" 分割字符串
    # 使用 '\n--New Chunk--\n' 作为分隔符，这样可以处理首尾 chunk
    chunk_blocks = formatted_string.strip().split('\n--New Chunk--\n') # strip() 去除首尾空白

    for block in chunk_blocks:
        block = block.strip() # 去除每个块的前后空白
        if not block: # 如果分割后出现空块，则跳过
            continue

        # 2. 找到 "File path: ..." 行后的第一个换行符
        first_newline_index = block.find('\n')

        if first_newline_index != -1:
            # 3. 提取换行符之后的内容
            content = block[first_newline_index + 1:].strip() # +1 跳过换行符本身
            contents.append(content)
        elif block.startswith("File path:"):
            # 特殊情况：如果块只有 "File path: ..." 行而没有内容（理论上不应发生）
            # 或者内容为空字符串，这里可以选择添加空字符串或跳过
            continue


    return contents


def evaluate_retrieval(query, expected_answers="", threshold=0.5, top_k=5):
    rag = asyncio.run(initialize_rag())
    mode = "naive"
    results = rag.query(
        query,
        QueryParam(mode=mode, only_need_context=True, top_k=top_k),
    )

    #relevant是检索结果，以列表形式返回
    retrieved_texts = str_to_list_with_lowlevel(results)
    precision = calculate_precision(retrieved_texts, expected_answers, threshold)
    recall = calculate_recall(retrieved_texts, expected_answers, threshold)
    f1 = calculate_f1(precision, recall)

    similarities = []
    for expected, retrieved in zip(expected_answers, retrieved_texts):
        similarities.append(compute_similarity(expected, retrieved))
    avg_similarity = np.mean(similarities) if similarities else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_similarity": avg_similarity,
        "retrieved_texts": retrieved_texts
    }


def main():

    evaluation_results = []

    for test_case in test_query:
        query = test_case["query"]
        expected_answers = test_case["expected_answers"]

        evaluation = evaluate_retrieval(query, expected_answers)
        evaluation_results.append({
            "query": query,
            "expected_answers": expected_answers,
            "evaluation": evaluation
        })

        print(f"Query: {query}")
        print(f"Expected Answers: {expected_answers}")
        print(f"Retrieved Results: {evaluation['retrieved_texts']}")
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
        print(f"F1 Score: {evaluation['f1']:.4f}")
        print(f"Average Similarity: {evaluation['avg_similarity']:.4f}")
        print("-" * 100)
    # 计算整体评估结果
    total_precision = sum(result["evaluation"]["precision"] for result in evaluation_results) / len(evaluation_results)
    total_recall = sum(result["evaluation"]["recall"] for result in evaluation_results) / len(evaluation_results)
    total_f1 = sum(result["evaluation"]["f1"] for result in evaluation_results) / len(evaluation_results)
    total_similarity = sum(result["evaluation"]["avg_similarity"] for result in evaluation_results) / len(
        evaluation_results)

    print("\n整体评估结果:")
    print(f"Average Precision: {total_precision:.4f}")
    print(f"Average Recall: {total_recall:.4f}")
    print(f"Average F1 Score: {total_f1:.4f}")
    print(f"Average Similarity: {total_similarity:.4f}")


main()













