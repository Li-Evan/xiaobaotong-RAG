# 首先，请确保你已经安装了必要的库
# pip install sentence-transformers

import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# 1. 定义我们的查询和文档库
query = "What is the capital of Germany?"

# 文档库中包含正确答案、相关但不精确的答案、以及不相关的答案
documents = [
    "Berlin is the capital of Germany.",  # 正确答案
    "Germany is a country in Central Europe.", # 相关，但不直接回答问题
    "The Brandenburg Gate is a famous landmark in Berlin.", # 包含关键词，但不是首都的定义
    "Paris is the capital of France.", # 不相关，但结构相似
    "What is the population of Berlin?" # 以问题的形式出现，包含关键词
]

print("--- 输入 ---")
print(f"查询: {query}")
print(f"文档库: {documents}\n")

# -----------------------------------------------------------
# 阶段一：召回 (使用 Bi-Encoder / Embedding 模型)
# -----------------------------------------------------------
# all-MiniLM-L6-v2 是一个轻量级且高效的Bi-Encoder模型
# 它会把查询和文档独立编码为向量
print("--- 阶段一：召回 (Bi-Encoder) ---")
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 独立编码查询和文档
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
doc_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)

# 使用余弦相似度计算得分
# 这是在向量数据库中发生的事情的模拟
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# 将得分与文档配对并排序
recall_results = sorted(zip(cosine_scores, documents), key=lambda x: x[0], reverse=True)

print("Bi-Encoder (Embedding) 相似度得分:")
for score, doc in recall_results:
    print(f"{score:.4f}\t{doc}")

# 取出召回阶段的Top-3结果，进入精排阶段
top_k = 3
rerank_candidates = [doc for score, doc in recall_results[:top_k]]
print(f"\n选择Top-{top_k}个召回结果进入下一阶段: {rerank_candidates}\n")

# -----------------------------------------------------------
# 阶段二：精排 (使用 Cross-Encoder / Rerank 模型)
# -----------------------------------------------------------
# ms-marco-MiniLM-L-6-v2 是一个为Rerank任务训练的强大的Cross-Encoder模型
print("--- 阶段二：精排 (Cross-Encoder) ---")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Cross-Encoder需要将查询和每个文档组成对
# 它不会生成向量，而是直接为每个对输出一个相关性分数
sentence_pairs = [[query, doc] for doc in rerank_candidates]

# 预测分数
cross_scores = cross_encoder.predict(sentence_pairs)

# 将分数与文档配对并排序
rerank_results = sorted(zip(cross_scores, rerank_candidates), key=lambda x: x[0], reverse=True)

print("Cross-Encoder (Rerank) 相关性得分:")
for score, doc in rerank_results:
    # Cross-Encoder的分数不是在0-1之间，它是一个logit值，越高越相关
    print(f"{score:.4f}\t{doc}")

print("\n--- 最终结论 ---")
print(f"召回阶段的Top-1: '{recall_results[0][1]}'")
print(f"精排阶段的Top-1: '{rerank_results[0][1]}'")