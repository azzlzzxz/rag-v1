from llama_index.core import Document

document = Document(
    text="""
    第一章：退款申请条件

    订单必须处于允许退款的状态。
    退款申请必须在规定时间内提交。
    退款金额不能超过订单剩余可退金额。

    第二章：退款申请步骤

    第一步，进入订单管理页面。
    第二步，选择需要退款的订单。
    第三步，填写退款原因并提交申请。
    提交后，可以在退款管理页面查看处理进度。

    第三章：退款失败处理

    如果退款失败，首先检查订单状态和支付渠道。
    如果订单状态正常但支付渠道异常，应联系相关服务方。
    不要在原退款请求状态未知时直接重复发起退款。
    """,
    metadata={
        "document_id": "refund_manual_001",
        "section": "退款管理",
        "version": 1,
    },
)
# print(document)

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)

# 从大块到小块，依次创建多层级节点
parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 128, 32]
)

# 返回的是包含各层级节点的扁平列表
nodes = parser.get_nodes_from_documents([document])

# 获取最底层、没有子节点的叶节点
leaf_nodes = get_leaf_nodes(nodes)

# 获取最顶层节点
root_nodes = get_root_nodes(nodes)

# print("全部节点数:", len(nodes))
# print("叶节点数:", len(leaf_nodes))
# print("根节点数:", len(root_nodes))

# for node in nodes[:10]:
#     parent = node.parent_node

#     print("-" * 40)
#     print("ID:", node.node_id)
#     print("文本:", node.text[:100])
#     print(
#         "父节点 ID:",
#         parent.node_id if parent else None
#     )
#     print("子节点数:", len(node.child_nodes))


from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever

# 1. 创建文档存储
docstore = SimpleDocumentStore()

# 保存全部层级节点，方便后续通过 ID 取回父节点
docstore.add_documents(nodes)

# 2. 创建 StorageContext
storage_context = StorageContext.from_defaults(
    docstore=docstore
)

# 3. 只把叶节点放入向量索引
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

# 4. 创建普通向量检索器
base_retriever = base_index.as_retriever(
    similarity_top_k=6
)

# 5. 包装成自动合并检索器
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    simple_ratio_thresh=0.5,
    verbose=True,
)

query = "退款失败以后应该怎么处理？"

# 原始向量检索结果
before = base_retriever.retrieve(query)

# 自动合并后的结果
after = retriever.retrieve(query)

print("\n=== 合并前 ===")
for item in before:
    print(item.node.node_id, item.score)
    print(item.node.text[:120])

print("\n=== 合并后 ===")
for item in after:
    print(item.node.node_id, item.score)
    print(item.node.text[:200])