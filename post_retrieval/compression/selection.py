import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# --- 1. 初始化客户端 ---
# 代码会从环境变量中自动读取 'OPENAI_API_KEY' 和 'OPENAI_BASE_URL'
# 请确保在运行前已经设置好这两个环境变量
try:
    llm_client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
except Exception as e:
    print(os.getenv('OPENAI_API_KEY'))
    print(os.getenv('OPENAI_BASE_URL'))
    print(os.getenv('OPENAI_MODEL'))
    print("错误：无法初始化OpenAI客户端。")
    print("请确保您已经正确设置了 'OPENAI_API_KEY' 和 'OPENAI_BASE_URL' 环境变量。")
    print(f"具体错误: {e}")
    exit()

def get_relevance_score_from_llm(query: str, document: str) -> int:
    """
    调用LLM API，让模型评估文档与问题的相关性，并返回一个1-10的分数。
    """
    # 这是我们发送给LLM的指令 (Prompt)
    # 它要求LLM扮演一个评估者的角色，并以可预测的格式返回分数。
    system_prompt = """
    你是一个文档相关性评估助手。
    你的任务是根据用户提供的问题，评估一份文档的相关性。
    请只用一个1到10之间的整数来回答，其中1表示完全不相关，10表示高度相关。
    不要说任何别的话，不要加任何解释，只要一个数字。
    """
    
    user_prompt = f"""
    问题: "{query}"
    ---
    文档内容: "{document}"
    ---
    请根据以上问题，评估这份文档的相关性，并给出一个1到10之间的分数。
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL'),  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # 使用较低的温度确保输出的稳定性
        )
        
        # 解析LLM返回的分数
        score_text = response.choices[0].message.content.strip()
        return int(score_text)
        
    except Exception as e:
        print(f"调用API时出错: {e}")
        # 如果API调用失败或返回格式不正确，给一个最低分
        return 1

# --- 工作流程开始 ---

# 0. 原始问题
original_query = "LLM如何通过自我反思来优化和筛选检索到的文档？"

print(f"原始问题: '{original_query}'\n" + "="*50)

# 1. 初步检索 (Initial Retrieval)
# 假设这是我们第一步从数据库或搜索引擎中获取的候选文档列表。
retrieved_documents = [
    "文档A: LLM的自我反思是一种有效策略。它通过让模型自己评估每个文档与原始问题的相关性，从而过滤掉质量不高的信息，只保留最相关的内容用于最终的答案生成。",
    "文档B: 数据压缩技术，如JPEG和MP3，是计算机科学中的重要组成部分，它们致力于减少文件大小，方便存储和传输。",
    "文档C: 介绍一种简单的家庭烘焙食谱：巧克力饼干。您需要准备面粉、糖、鸡蛋和巧克力豆。制作过程非常有趣。",
    "文档D: 在一个典型的RAG（检索增强生成）流程中，可以增加一个评估步骤。在将检索到的片段送入生成器之前，先让一个LLM对这些片段进行打分和筛选，可以显著提升最终答案的准确性。"
]

print("【第1步：初步检索】\n获取了以下候选文档：")
for doc in retrieved_documents:
    print(f"  - {doc}")
print("\n" + "="*50)

# 2. 自我评估 (Self-Correction / Evaluation)
print("【第2步：LLM自我评估】\n正在调用API，让LLM为每个文档的相关性打分...")

scored_documents = []
for doc in retrieved_documents:
    # 真实调用LLM API进行打分
    score = get_relevance_score_from_llm(original_query, doc)
    scored_documents.append({"document": doc, "score": score})
    print(f"  - 评估文档: \"{doc[:30]}...\" | LLM给出的相关性得分: {score}/10")
print("\n" + "="*50)

# 3. 过滤 (Filtering)
score_threshold = 7  # 设定一个分数门槛，只保留得分大于等于7的文档
high_quality_documents = [
    item for item in scored_documents if item["score"] >= score_threshold
]

print(f"【第3步：过滤】\n只保留得分 >= {score_threshold} 的文档。")
if high_quality_documents:
    print("最终筛选出的高质量文档如下：")
    for item in high_quality_documents:
        print(f"  - (得分: {item['score']}/10) {item['document']}")
else:
    print("没有找到足够相关的文档。")
print("\n" + "="*50)

# 4. 最终答案生成阶段
print("【第4步：最终答案生成】")
print("接下来，LLM将只使用以上筛选出的高分文档作为上下文，来生成一个更精准的答案。")