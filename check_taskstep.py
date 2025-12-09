import os
import numpy as np

def inspect_se1_npy_file(file_path):
    """
    指定された単一のAvatarGPT se1生成結果(.npy)を読み込み、
    テキスト記述（Scene, Task, Step）を表示する。
    """
    
    if not os.path.exists(file_path):
        print(f"エラー: 指定されたファイルが見つかりません: {file_path}")
        return

    file_name = os.path.basename(file_path)
    print(f"Inspecting file: {file_name}\n")
    print("="*60)

    try:
        # .npyファイルのロード (辞書オブジェクトを含んでいるため allow_pickle=True が必須)
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Could not load {file_name}: {e}")
        return

    # --- データの抽出 ---
    # シーン記述 (リストに入っているので[0]を取得。なければデフォルト値)
    scene_desc = data.get("caption", ["No Scene Description"])[0]
    
    # アクション記述のリスト ({"task":..., "step":...} の辞書が入ったリスト)
    actions = data.get("actions", [])

    # --- 表示 ---
    print(f"Scene: {scene_desc}")
    
    if not actions:
        print("  (No actions found in this file)")
    
    current_task_text = None
    task_count = 0
    step_count_in_task = 0

    # タスクとステップを整理して表示
    for i, action in enumerate(actions):
        task_text = action.get("task", "No Task")
        step_text = action.get("step", "No Step")

        # タスクの内容が変わったら新しいタスクヘッダーを表示
        if task_text != current_task_text:
            task_count += 1
            step_count_in_task = 0 # ステップ番号リセット
            current_task_text = task_text
            print(f"\n  [Task {task_count}] {task_text}")
        
        step_count_in_task += 1
        print(f"    └ Step {step_count_in_task}: {step_text}")

    print("\n" + "="*60 + "\n")

# ==========================================
# 使い方: ここに対象のファイルパスを指定してください
# ==========================================
# 例: "./output/se1_p0/generated_000.npy" など
TARGET_FILE = "./demo_outputs/demo/output/se1_p0/B0002_T0000.npy" 

if __name__ == "__main__":
    inspect_se1_npy_file(TARGET_FILE)