import json
import os

# ==========================================
# [設定] ここを書き換えて好きなシチュエーションを作ってください
# ==========================================
DATA_LIST = [
    {
        "filename": "my_scenario_01",  # 生成されるファイル名
        "scene": "I had a fight with my husband.",  # シーン設定 (Context)
        "first_task": "The person stands." # 最初の行動 (Task)
    },
    {
        "filename": "my_scenario_02",
        "scene": "I ran into a classmate unexpectedly at a store.",
        "first_task": "The person stands."
    },
    {
        "filename": "my_scenario_03",
        "scene": "I received a cake.",
        "first_task": "The person stands."
    }
]
# ==========================================

# パス設定
OUTPUT_JSON_DIR = "demo_inputs/samples"
OUTPUT_TXT_PATH = "demo_inputs/my_test_list.txt"

# ディレクトリがなければ作成
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

txt_content = []

for data in DATA_LIST:
    # JSON構造の作成（se1モード用最小構成）
    json_data = {
        "video_name": "dummy",
        "video_fps": 30,
        "contexts": [ data["scene"] ],  # ここが重要：シーン
        "actions": [
            {
                "start_time": 0,
                "end_time": 5,
                "start_frame": 0,
                "end_frame": 150,
                "captions": [
                    {
                        "detail": "dummy detail",
                        "executable_steps": "dummy steps",
                        "executable_simplified": [
                            data["first_task"]  # ここが重要：最初のタスク
                        ]
                    }
                ]
            }
        ]
    }

    # JSON書き出し
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{data['filename']}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Created: {json_path}")
    
    # TXTファイル用リストに追加
    txt_content.append(data["filename"])

# TXTファイル書き出し
with open(OUTPUT_TXT_PATH, 'w') as f:
    f.write("\n".join(txt_content))

print(f"Created List: {OUTPUT_TXT_PATH}")
print("Done.")