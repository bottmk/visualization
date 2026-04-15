sample_inputs/ — 入力ファイルサンプル集
==========================================

このフォルダには各種入力ファイルのサンプルを格納する。

ファイル一覧
------------
device_xyz_sample.csv
  - 装置XYZ 固有フォーマット
  - タブ区切り、先頭5行ヘッダ、高さ単位 nm、グリッド 100×100
  - 対応ローダー: custom_surfaces/device_xyz.py (DeviceXyzSurface)
  - 使用 config: sample_inputs/config_device_xyz.yaml

今後追加予定のフォーマット
--------------------------
  - 装置ABC フォーマット（*.dat）
  - 干渉計出力（*.csv, *.txt）
  - AFM 出力（*.nid, *.sdf）

新しいフォーマットを追加する場合
---------------------------------
1. サンプルファイルをこのフォルダに追加する
2. custom_surfaces/<device_name>.py に MeasuredSurface サブクラスを実装する
3. このファイルにフォーマット説明を追記する
