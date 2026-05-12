

import asyncio
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent

from operator import add
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph

try:
    # 作为包运行（例如从项目上级目录执行）时使用
    from Cheer.config.load_key import load_key
    from Cheer.couplet_load import retrieve_couplet_samples
except ModuleNotFoundError:
    # 直接在 Cheer 目录执行脚本时使用
    from config.load_key import load_key
    from couplet_load import retrieve_couplet_samples

# 统一使用 ChatOpenAI，既能普通问答，也能给 ReAct Agent 绑定工具
llm = ChatOpenAI(
    api_key=load_key("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-flash",
    temperature=0
)

nodes = ["supervisor", "travel", "joke", "couplet", "other"]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str



def other_node(state: State):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    return {"messages": [HumanMessage(content="我无法回答这个问题")], "type": "other"}

def supervisor_node(state: State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})

    # 完全还原你原来的prompt和注释
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将其任务分给其他Agent执行。
如果用户的问题是和旅游路线规划相关的，那就返回 travel。
如果用户的问题是希望讲一个笑话，那就返回 joke 。
如果用户的问题是希望对一个对联，那就返回 couplet 。
如果是其他的问题，返回 other 。
除了这几个选项外，不要返回任何其他的内容。
"""
    user_content = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else state["messages"][0]
    prompts = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_content),
    ]

    # 还原你原来的逻辑和提示输出
    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']} 智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content.strip()
        writer({"supervisor_step": f"问题分类结果:{typeRes}"})
        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError("type is not in (travel,joke,other,couplet)")

def travel_node(state: State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>> travel_node"})

    systemPrompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划，使用中文回答，并返回一个不超过100字的规划结果。"

    user_content = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else state["messages"][0]

    # 高德地图的MCP配置信息
    client = MultiServerMCPClient({
        # "amap-amap-sse": {
        #     "url": "https://mcp.amap.com/sse?key=451ad40d0e39453600f2a305e31eabe4",
        #     "transport": "streamable_http"
        # },
        "amap-maps": {
            "command": "npx",
            "args": [
                "-y",
                "@amap/amap-maps-mcp-server"
            ],
            "env": {
                "AMAP_MAPS_API_KEY": load_key("AMAP_MAPS_API_KEY")
            },
            "transport": "stdio"
        }
    })

    tools = asyncio.run(client.get_tools())
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=systemPrompt
    )
    response = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": user_content}]})
    )
    final_message = response["messages"][-1].content
    if isinstance(final_message, list):
        final_message = "".join(
            item.get("text", "") for item in final_message if isinstance(item, dict)
        )
    writer({"travel_result": final_message})
    return {"messages": [HumanMessage(content=final_message)], "type": "travel"}

def joke_node(state: State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>> joke_node"})

    systemPrompt = "你是一个笑话大师，根据用户的问题，写一个不超过100个字的笑话。"

    user_content = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else state["messages"][0]

    prompts = [
        SystemMessage(content=systemPrompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(prompts)

    writer({"joke_result":response.content})

    return {"messages": [HumanMessage(content=response.content)], "type": "joke"}

def couplet_node(state: State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>> couplet_node"})

    user_content = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else state["messages"][0]

    try:
        samples = retrieve_couplet_samples(user_content, k=10)
        retrieval_source = samples[0].get("source", "unknown") if samples else "none"
        if samples:
            sample_lines = "\n".join(
                [
                    f"{idx + 1}. 上联：{item['upper']} | 下联：{item['lower']}"
                    for idx, item in enumerate(samples)
                ]
            )
        else:
            sample_lines = "（未检索到可用参考样本）"
    except Exception as e:
        retrieval_source = "failed"
        writer({"couplet_rag_error": str(e)})
        sample_lines = "（检索暂不可用，请仅依靠你的对联能力完成）"

    system_prompt = f"""你是一个专业的对联大师。
任务：根据用户给出的上联，写出工整、自然、有意境的下联。
要求：
1. 只输出下联，不要解释。
2. 尽量保证词性、结构、平仄风格协调。
3. 优先参考检索样本的风格，不要逐字照抄。

参考样本：
{sample_lines}
"""

    prompts = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(prompts)

    writer({"couplet_retrieval_source": retrieval_source})
    writer({"couplet_samples": sample_lines})
    writer({"couplet_result": response.content})

    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}

def routing_func(state: State):
    t = state["type"]
    if t == "travel":
        return "travel_node"
    elif t == "joke":
        return "joke_node"
    elif t == "couplet":
        return "couplet_node"
    elif t == END:
        return END
    else:
        return "other_node"

# 构建图
builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func)
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    for chunk in graph.stream({"messages": [HumanMessage(content="给我对一个下联：今天热得很")]},
            config,
            stream_mode="custom"):
        print(chunk)
