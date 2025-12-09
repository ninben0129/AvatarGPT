import numpy as np
import argparse

def inspect_motion_npy(file_path):
    try:
        # .npyファイルをロード (辞書形式で保存されているため allow_pickle=True が必要)
        data = np.load(file_path, allow_pickle=True).item()
        
        # 'pred' キーの中に生成結果が入っています
        # 構造: {'pred': {'body': array}, 'caption': ..., ...}
        if 'pred' in data and 'body' in data['pred']:
            motion_data = data['pred']['body']
            
            print(f"SUCCESS: ファイルを読み込みました: {file_path}")
            print(f"データの型: {type(motion_data)}")
            print(f"データのシェイプ (Shape): {motion_data.shape}")
            
            # 次元数の確認
            # AvatarGPTの出力は通常 (Batch, Dimension, Frames) の形式です
            batch_size, dimension, frames = motion_data.shape
            
            print("-" * 30)
            print(f"   - バッチサイズ (Batch): {batch_size}")
            print(f"   - 次元数 (Dimension)  : {dimension}")
            print(f"   - フレーム数 (Frames) : {frames}")
            print("-" * 30)
            
            if dimension == 263:
                print("判定: HumanML3D標準の「263次元」フォーマットです。")
            else:
                print(f"判定: 標準とは異なる次元数です ({dimension})。設定を確認してください。")
                
        else:
            print("Error: 予想されるキー ('pred' -> 'body') が見つかりません。")
            print(f"Keys found: {data.keys()}")

    except Exception as e:
        print(f"Error: ファイルの読み込みに失敗しました。\n{e}")

if __name__ == "__main__":
    # 使い方: python check_npy_shape.py --file demo_outputs/demo/output/se1_p0/B0000_T0000.npy
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .npy file")
    args = parser.parse_args()
    
    inspect_motion_npy(args.file)