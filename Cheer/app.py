import random
import traceback

import gradio as gr
from langchain_core.messages import HumanMessage

from Test_in_local import graph


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)


def process_input(text: str):
    if not text or not text.strip():
        return "请输入问题后再提交。", "未执行"

    config = {
        "configurable": {
            "thread_id": str(random.randint(1, 1_000_000)),
        }
    }

    try:
        result = graph.invoke({"messages": [HumanMessage(content=text.strip())]}, config)
        messages = result.get("messages", [])
        answer = _to_text(messages[-1].content) if messages else "未获取到回复"
        debug = f"thread_id={config['configurable']['thread_id']}\nstatus=success"
        return answer, debug
    except Exception:
        err = traceback.format_exc()
        return "执行失败，请检查控制台日志。", err


CUSTOM_CSS = """
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --bg-gradient: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    --card-bg: rgba(255,255,255,0.06);
    --card-border: rgba(255,255,255,0.1);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent-green: #34d399;
    --accent-orange: #fb923c;
    --accent-pink: #f472b6;
}
body, .gradio-container { background: var(--bg-gradient) !important; min-height: 100vh; }
.gradio-container { max-width: 800px !important; margin: 0 auto !important; padding: 2rem 1rem !important; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
h1 { text-align: center; font-size: 2.2rem !important; font-weight: 700 !important; background: linear-gradient(135deg, #818cf8, #f472b6, #34d399); -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; margin-bottom: 0.3rem !important; }
.subtitle { text-align: center; color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1.5rem; }
.tag-row { display: flex; justify-content: center; gap: 8px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.tag { display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 500; border: 1px solid var(--card-border); background: var(--card-bg); color: var(--text-secondary); }
.tag.travel { border-color: #34d399; color: var(--accent-green); }
.tag.joke { border-color: #fb923c; color: var(--accent-orange); }
.tag.couplet { border-color: #f472b6; color: var(--accent-pink); }
.card { background: var(--card-bg) !important; border: 1px solid var(--card-border) !important; border-radius: 16px !important; backdrop-filter: blur(12px); padding: 1.2rem !important; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.gr-box { border: none !important; box-shadow: none !important; background: transparent !important; }
label { color: var(--text-primary) !important; font-weight: 600 !important; font-size: 0.9rem !important; }
textarea, input[type="text"] { background: #fff !important; border: 1px solid #d1d5db !important; border-radius: 12px !important; color: #1f2937 !important; font-size: 0.95rem !important; padding: 12px 16px !important; }
textarea:focus, input[type="text"]:focus { border-color: var(--primary-light) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important; }
button.primary { background: linear-gradient(135deg, var(--primary), #a855f7) !important; border: none !important; border-radius: 12px !important; color: white !important; font-weight: 600 !important; padding: 10px 32px !important; transition: all 0.2s !important; cursor: pointer; }
button.primary:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important; }
.output-text { background: #fff !important; border: 1px solid #d1d5db !important; border-radius: 12px !important; color: #1f2937 !important; font-size: 1rem !important; line-height: 1.6 !important; }
.debug-text { background: rgba(0,0,0,0.3) !important; border: 1px solid var(--card-border) !important; border-radius: 12px !important; color: #64748b !important; font-size: 0.8rem !important; font-family: 'JetBrains Mono', 'Cascadia Code', monospace !important; }
.examples-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 1rem; }
.examples-label { color: var(--text-secondary); font-size: 0.8rem; }
.example-btn { all: unset; cursor: pointer; padding: 6px 14px; border-radius: 8px; background: rgba(255,255,255,0.05); border: 1px solid var(--card-border); color: var(--text-secondary); font-size: 0.85rem; transition: all 0.15s; }
.example-btn:hover { background: rgba(99,102,241,0.15); border-color: var(--primary-light); color: var(--text-primary); }
footer { text-align: center; color: var(--text-secondary); font-size: 0.75rem; margin-top: 2rem; opacity: 0.6; }
"""

EXAMPLES_HTML = """
<div class="examples-row">
    <div class="examples-label">试试这些问题：</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="example-btn" onclick="document.querySelector('textarea').value='给我规划一个北京三日游路线';document.querySelector('textarea').dispatchEvent(new Event('input'))">🗺️ 北京三日游</button>
        <button class="example-btn" onclick="document.querySelector('textarea').value='讲个笑话';document.querySelector('textarea').dispatchEvent(new Event('input'))">😂 讲个笑话</button>
        <button class="example-btn" onclick="document.querySelector('textarea').value='上联：春风又绿江南岸';document.querySelector('textarea').dispatchEvent(new Event('input'))">🪄 对对联</button>
        <button class="example-btn" onclick="document.querySelector('textarea').value='今天天气怎么样';document.querySelector('textarea').dispatchEvent(new Event('input'))">💬 随便聊聊</button>
    </div>
</div>
"""


with gr.Blocks(css=CUSTOM_CSS, title="Cheer - 多智能体助手", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <h1>✨ Cheer</h1>
        <p class="subtitle">基于 LangGraph 的多智能体对话系统 · 智能路由你的问题</p>
        <div class="tag-row">
            <span class="tag travel">🗺️ 路线规划</span>
            <span class="tag joke">😂 讲笑话</span>
            <span class="tag couplet">🪄 对对联</span>
        </div>
    """)

    with gr.Column(elem_classes="card"):
        input_box = gr.Textbox(
            label="你的问题",
            placeholder="例如：给我对个下联，上联是：远上寒山石径斜",
            lines=3,
            elem_classes="input-text",
        )

        gr.HTML(EXAMPLES_HTML)

        submit_btn = gr.Button("发送", variant="primary", elem_classes="primary")

        output_box = gr.Textbox(
            label="回答",
            lines=4,
            elem_classes="output-text",
            interactive=False,
        )

        with gr.Accordion("🔍 调试信息", open=False):
            debug_box = gr.Textbox(
                label="",
                lines=6,
                elem_classes="debug-text",
                interactive=False,
            )

    gr.HTML("<footer>Cheer · LangGraph Multi-Agent Demo</footer>")

    submit_btn.click(
        fn=process_input,
        inputs=[input_box],
        outputs=[output_box, debug_box],
    )
    input_box.submit(
        fn=process_input,
        inputs=[input_box],
        outputs=[output_box, debug_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
