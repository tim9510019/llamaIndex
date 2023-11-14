import os
import time
from pathlib import Path
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import OpenAI
from llama_index.embeddings import resolve_embed_model
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

openai_api_key = os.environ["OPENAI_API_KEY"]


srcDir = "./srcDataV1"
datetime = "2023-11-13"

queryWord = "Please answer with traditional Chinese. (ZH)"
queryWord += "Please organize the above text from an economic and political macro perspective and output it in the form of a paper."
queryWord += "Please help me deduce from all important dependencies, explain it, and make it relatively correct."
queryWord += """Please give as "1. Stocks or real estate:, 2. commodities:, 3. Dollar or Short-term bonds:, 4. Long-term bonds:"."""
queryWord += "Please add interesting questions for readers and provide answers."
queryWord += "Please include some economic formulas to support the viewpoints."


srcFiles = [os.path.join(srcDir, x) for x in os.listdir(srcDir)]

# Define Excel Reader
PandasExcelReader = download_loader("PandasExcelReader")
loader = PandasExcelReader(pandas_config={"header": 0, "usecols": ["answer"]})

# Define Node parser
node_parser = SimpleNodeParser.from_defaults(chunk_size=4096)

# Define embed. model
embed_model = resolve_embed_model("local:BAAI/bge-small-en")

# Define LLM model
llm = OpenAI(model="gpt-4-1106-preview", api_key=openai_api_key)

# Define service_context
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


""" 
Basic RAG Review
"""
# Load documents from excel files
docs = [loader.load_data(file=Path(x)) for x in srcFiles]

# Filt out special tokens from documents
docsMeta = []
for doc, src in zip(docs, srcFiles):
    doc[0].metadata = {"file_name": src, "date_time": src.split("_")[1]}
    doc[0].text = (
        doc[0]
        .text.replace("<|system|>", "")
        .replace("<|user|>", "")
        .replace("<|assistant|>", "")
        .replace("</s>", "")
        .replace("<s>", "")
    )
    docsMeta.append(doc)

print(docsMeta[0])

# Convert documents to nodes
base_nodes_list = [node_parser.get_nodes_from_documents(x) for x in docsMeta]

# Give nodes index names
base_nodes = []
nIdx = 0
for nodes in base_nodes_list:
    for node in nodes:
        node.id_ = f"node-{nIdx}"
        base_nodes.append(node)
        nIdx += 1

# Convert nodes to vector index and setup embed. / llm model
base_index = VectorStoreIndex(base_nodes, service_context=service_context)

# Retrieve top 3 similarity result of "certain" datetime
base_retriever = base_index.as_retriever(
    similarity_top_k=3,
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="date_time", value=datetime)]
    ),
)


print("=" * 20, "Base RAG Review", "=" * 20)
print("Base Node Num:", len(base_nodes))


timeBegin = time.time()

retrievals = base_retriever.retrieve(queryWord)

for n in retrievals:
    print(n)


query_engine_base = RetrieverQueryEngine.from_args(
    base_retriever, service_context=service_context
)

response = query_engine_base.query(queryWord)

print(str(response))

print("Elapsed Time: %.2f" % (time.time() - timeBegin))
