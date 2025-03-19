from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

# 添加rag信息来源方法：https://github.com/run-llama/llama_index/issues/15321

documents = SimpleDirectoryReader(str(data_dir)).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query(
    "告诉我Chain-of-Thought Prompting Elicits Reasoning in Large Language Models这篇文章作者的名字是什么？"
)
print(response)
