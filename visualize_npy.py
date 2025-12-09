import os
import argparse
import glob
import numpy as np
import torch

# MDMのモジュールを読み込み
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil

def process_single_file(file_path):
    """
    単一のnpyファイルを処理してmp4を保存する関数
    元の visualize_npy のロジックをここに移動
    """
    print(f"\nProcessing: {file_path}")
    
    # ---------------------------------------------------------
    # 1. AvatarGPTのモーションデータをロード
    # ---------------------------------------------------------
    try:
        # 辞書としてロード
        raw_data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading npy file ({file_path}): {e}")
        return

    motion_data = None
    # print(f"  -> Keys found in NPY: {list(raw_data.keys())}") # ログが多すぎる場合はコメントアウト

    # ---------------------------------------------------------
    # 2. データの抽出 logic (pred -> body)
    # ---------------------------------------------------------
    # 'pred' キーを優先的に探索
    if 'pred' in raw_data:
        pred_content = raw_data['pred']
        
        # predが辞書で、かつbodyを持っている場合 (確認済みの構造)
        if isinstance(pred_content, dict) and 'body' in pred_content:
            # print("  -> Found structure: ['pred']['body']")
            motion_data = pred_content['body']
        
        # predが直接配列の場合
        elif isinstance(pred_content, (np.ndarray, list)):
             # print("  -> Found structure: ['pred'] (direct array)")
             motion_data = pred_content
    
    # 'gt' (正解データ) を可視化したい場合のフォールバック
    elif 'gt' in raw_data:
        print("  -> 'pred' not found. Using 'gt'.")
        if isinstance(raw_data['gt'], dict) and 'body' in raw_data['gt']:
            motion_data = raw_data['gt']['body']
        else:
            motion_data = raw_data['gt']

    if motion_data is None:
        # その他のキーを探索
        for key in ['motion', 'joints']:
            if key in raw_data:
                print(f"  -> Fallback: Using key '{key}'")
                motion_data = raw_data[key]
                break

    if motion_data is None:
        print(f"Skipping {file_path}: Could not find motion array in the file.")
        return

    # ---------------------------------------------------------
    # 3. 前処理 (型変換・シェイプ調整)
    # ---------------------------------------------------------
    try:
        # 強制的にfloat32配列へ変換
        motion_data = np.array(motion_data, dtype=np.float32)
    except:
        # ragged nested sequences対策
        motion_data = np.array(motion_data[0], dtype=np.float32)

    motion_tensor = torch.from_numpy(motion_data).float()
    
    # バッチ次元がない場合 (263, Length) -> (1, 263, Length) を追加
    if motion_tensor.ndim == 2:
        motion_tensor = motion_tensor.unsqueeze(0)

    # print(f"  -> Loaded Shape: {motion_tensor.shape}")

    # AvatarGPTの出力 (Batch, 263, Length) を MDM形式 (Batch, Length, 263) に変換
    if motion_tensor.shape[1] == 263:
        # print("  -> Transposing dimensions to (Batch, Length, 263)...")
        motion_tensor = motion_tensor.permute(0, 2, 1)

    # ---------------------------------------------------------
    # 4. 特徴量 -> XYZ座標変換
    # ---------------------------------------------------------
    # ※ここで逆正規化(Un-normalization)は行いません (Rawデータのため)
    # print("Recovering XYZ coordinates from RIC features (Skipping normalization)...")
    
    n_joints = 22 # HumanML3D standard
    
    # recover_from_ric は (Batch, Length, 263) を受け取ります
    try:
        xyz_motion = recover_from_ric(motion_tensor, n_joints)
        
        # バッチの0番目を取得 -> (Length, Joints, 3)
        xyz_motion = xyz_motion[0].numpy()
        
        frames_len = xyz_motion.shape[0]
        print(f"  -> Final XYZ Motion Shape: {xyz_motion.shape} (Frames: {frames_len})")

        # ---------------------------------------------------------
        # 5. 動画保存
        # ---------------------------------------------------------
        skeleton = paramUtil.t2m_kinematic_chain
        save_path = file_path.replace('.npy', '.mp4')
        
        print(f"Saving animation to: {save_path}")
        
        # dataset='humanml' を指定してプロット
        plot_3d_motion(save_path, skeleton, xyz_motion, 
                       dataset='humanml', title="", fps=20)
        print("Done.")
        
    except Exception as e:
        print(f"Error during processing/plotting {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main(args):
    input_path = args.input_path
    
    # 入力パスが存在するか確認
    if not os.path.exists(input_path):
        print(f"Error: Path does not exist -> {input_path}")
        return

    # 1. ディレクトリの場合
    if os.path.isdir(input_path):
        print(f"--- Batch Visualization Mode ---")
        print(f"Target Directory: {input_path}")
        
        # フォルダ内の .npy ファイルをすべて取得
        npy_files = glob.glob(os.path.join(input_path, "*.npy"))
        
        if not npy_files:
            print("No .npy files found in this directory.")
            return
            
        print(f"Found {len(npy_files)} files. Starting processing...\n")
        
        for npy_file in npy_files:
            process_single_file(npy_file)
            
    # 2. ファイルの場合
    elif os.path.isfile(input_path):
        print(f"--- Single File Visualization Mode ---")
        process_single_file(input_path)
        
    else:
        print("Invalid input path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input .npy file OR directory containing .npy files")
    args = parser.parse_args()

    main(args)