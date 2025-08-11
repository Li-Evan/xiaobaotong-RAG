from FlagEmbedding import FlagReranker
import torch
import torch.nn.functional as F

# 使用fp16以提升效率
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) 

# 计算单个查询与文档对的分数
#score = reranker.compute_score(['query', 'passage'])

# 批量计算多个数据对的分数
scores = reranker.compute_score([['特斯拉在中国市场的挑战是什么？', '特斯拉在中国市场面临着来自本土品牌的激烈竞争、供应链稳定性的考验以及不断变化的政策法规等多重挑战。'], ['特斯拉在中国市场的挑战是什么？', '特斯拉在中国市场的机遇巨大，包括广阔的消费市场、完善的充电设施网络以及消费者对电动汽车日益增长的热情。']])


for i in range(len(scores)):
    # 保留3位小数输出分数
    print("Doc", i, "score:", round(scores[i]/10, 3))