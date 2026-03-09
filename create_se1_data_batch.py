import json
import os

# ==========================================
# [設定] 入力と出力の設定
# ==========================================
# 読み込むテキストファイル（ここにシーンを1行ずつ書いておく）
INPUT_SCENE_FILE = "userstudy_scenarios.txt" 

# 固定するタスク
FIXED_TASK = "The person stands."

# 出力先の設定
OUTPUT_JSON_DIR = "demo_inputs/samples"
OUTPUT_TXT_PATH = "demo_inputs/userstudy_list.txt"

# ファイル名のプレフィックス（例: my_scenario_01.json, my_scenario_02.json...）
FILENAME_PREFIX = "userstudy_scenario"

# ==========================================

# ディレクトリがなければ作成
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# テキストファイルが存在するか確認（なければダミーを作成して警告）
if not os.path.exists(INPUT_SCENE_FILE):
    with open(INPUT_SCENE_FILE, "w", encoding="utf-8") as f:
        f.write("I had a fight with my husband.\n")
        f.write("I ran into a classmate unexpectedly at a store.\n")
        f.write("I received a cake.\n")
    print(f"[{INPUT_SCENE_FILE}] が見つからなかったため、ダミーファイルを作成しました。")

# 入力ファイルを読み込み
with open(INPUT_SCENE_FILE, "r", encoding="utf-8") as f:
    # 空行を除去してリスト化
    lines = [line.strip() for line in f.readlines() if line.strip()]

txt_content = []

print(f"--- Processing {len(lines)} scenes ---")

for i, scene_text in enumerate(lines):
    # ファイル名を連番で生成 (例: my_scenario_01)
    # i+1 なので 1からスタート、:02d で2桁埋め (01, 02, ...)
    filename = f"{FILENAME_PREFIX}_{i+1:02d}"

    # JSON構造の作成
    json_data = {
        "video_name": "dummy",
        "video_fps": 30,
        "contexts": [ scene_text ],  # 読み込んだ1行をここにセット
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
                            FIXED_TASK  # 固定タスクをここにセット
                        ]
                    }
                ]
            }
        ]
    }

    # JSON書き出し
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{filename}.json")
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"Created: {json_path} (Scene: {scene_text[:20]}...)")
    
    # TXTファイル用リストに追加
    txt_content.append(filename)

# TXTファイル書き出し
with open(OUTPUT_TXT_PATH, 'w', encoding="utf-8") as f:
    f.write("\n".join(txt_content))

print(f"\nCreated List: {OUTPUT_TXT_PATH}")
print("Done.")