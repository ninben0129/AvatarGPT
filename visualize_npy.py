import os
import argparse
import numpy as np
import torch

# MDMのモジュールを読み込み
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil

def visualize_npy(args):
    # ---------------------------------------------------------
    # 1. 統計情報のロード (Mean/Std)
    # ---------------------------------------------------------
    mean_path = os.path.join(args.stat_path, 'Mean.npy')
    std_path = os.path.join(args.stat_path, 'Std.npy')
    
    print(f"Loading stats from: {args.stat_path}")
    try:
        mean = torch.from_numpy(np.load(mean_path)).float()
        std = torch.from_numpy(np.load(std_path)).float()
    except Exception as e:
        print(f"Error loading stats: {e}")
        return

    # ---------------------------------------------------------
    # 2. AvatarGPTのモーションデータをロード
    # ---------------------------------------------------------
    print(f"Loading motion from: {args.input_path}")
    try:
        # 辞書としてロード
        raw_data = np.load(args.input_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return

    motion_data = None
    print(f"  -> Keys found in NPY: {list(raw_data.keys())}")

    # ---------------------------------------------------------
    # 3. データの抽出 (pred -> body)
    # ---------------------------------------------------------
    # 指示通り 'pred' を優先的に使用します
    if 'pred' in raw_data:
        pred_content = raw_data['pred']
        
        # predが辞書で、かつbodyを持っている場合
        if isinstance(pred_content, dict) and 'body' in pred_content:
            print("  -> Found structure: ['pred']['body']")
            motion_data = pred_content['body']
        
        # predが直接配列の場合
        elif isinstance(pred_content, (np.ndarray, list)):
             print("  -> Found structure: ['pred'] (direct array)")
             motion_data = pred_content
    
    # 見つからない場合のフォールバック (一応残しておきます)
    if motion_data is None:
        print("  -> 'pred' not found or invalid. Searching other keys...")
        for key in ['motion', 'joints', 'gt']:
            if key in raw_data:
                print(f"  -> Fallback: Using key '{key}'")
                if isinstance(raw_data[key], dict) and 'body' in raw_data[key]:
                    motion_data = raw_data[key]['body']
                else:
                    motion_data = raw_data[key]
                break

    if motion_data is None:
        raise ValueError("Could not find motion array (checked ['pred']['body'] etc).")

    # ---------------------------------------------------------
    # 4. 前処理 (型変換・シェイプ調整)
    # ---------------------------------------------------------
    try:
        # 強制的にfloat32配列へ変換
        motion_data = np.array(motion_data, dtype=np.float32)
    except:
        # 配列の次元が不揃い(ragged)な場合の救済
        motion_data = np.array(motion_data[0], dtype=np.float32)

    motion_tensor = torch.from_numpy(motion_data).float()
    
    # バッチ次元がない場合 (263, Length) -> (1, 263, Length) を追加
    if motion_tensor.ndim == 2:
        motion_tensor = motion_tensor.unsqueeze(0)

    print(f"  -> Loaded Shape: {motion_tensor.shape}")

    # AvatarGPTの出力 (Batch, 263, Length) を MDM形式 (Batch, Length, 263) に変換
    if motion_tensor.shape[1] == 263:
        print("  -> Transposing dimensions to (Batch, Length, 263)...")
        motion_tensor = motion_tensor.permute(0, 2, 1)

    # ---------------------------------------------------------
    # 5. 逆正規化 (Un-normalization)
    # ---------------------------------------------------------
    # データの数値範囲を確認
    data_min = motion_tensor.min().item()
    data_max = motion_tensor.max().item()
    print(f"  -> Data Stats | Min: {data_min:.4f}, Max: {data_max:.4f}, Mean: {motion_tensor.abs().mean().item():.4f}")

    # 通常、正規化されたデータは小さい値(-5~5程度)です。
    # もし値が非常に大きい場合は既に正規化が解除されている可能性があります。
    if abs(data_max) > 10.0 or abs(data_min) > 10.0:
        print("  -> WARNING: Data values seem large. Skipping un-normalization.")
    else:
        print("  -> Applying un-normalization (X * Std + Mean)...")
        motion_tensor = motion_tensor.to(std.device) * std + mean

    # ---------------------------------------------------------
    # 6. 特徴量 -> XYZ座標変換
    # ---------------------------------------------------------
    print("Recovering XYZ coordinates from RIC features...")
    n_joints = 22 # HumanML3D standard
    
    # recover_from_ric は (Batch, Length, 263) を受け取ります
    # 出力は (Batch, Length, Joints, 3)
    xyz_motion = recover_from_ric(motion_tensor, n_joints)
    
    # バッチの0番目を取得 -> (Length, Joints, 3)
    xyz_motion = xyz_motion[0].numpy()
    
    frames_len = xyz_motion.shape[0]
    print(f"  -> Final XYZ Motion Shape: {xyz_motion.shape} (Frames: {frames_len})")

    # ---------------------------------------------------------
    # 7. 動画保存
    # ---------------------------------------------------------
    skeleton = paramUtil.t2m_kinematic_chain
    save_path = args.input_path.replace('.npy', '.mp4')
    
    print(f"Saving animation to: {save_path}")
    
    # タイトルを空にして余計なテキストを排除
    # fps=20 はHumanML3Dの標準
    try:
        plot_3d_motion(save_path, skeleton, xyz_motion, 
                       dataset='humanml', title="", fps=20)
        print("Done.")
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input .npy file")
    parser.add_argument("--stat_path", type=str, default='dataset/humanml', help="Path to Mean.npy/Std.npy directory")
    args = parser.parse_args()

    visualize_npy(args)