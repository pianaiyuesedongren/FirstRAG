import json
import os

def load_key(param: str) -> str:
    """从 Keys.json 文件中读取密钥，直接返回字符串"""
    # 定位 Keys.json 文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keys_path = os.path.join(current_dir, "Keys.json")

    # 读取并返回 key
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)

    if param not in keys:
        raise ValueError(f"Keys.json 中未找到键: {param}")
    return keys[param]