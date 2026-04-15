sample_inputs/ — 入力ファイルサンプル集
==========================================

このフォルダには各種入力ファイルのサンプルを格納する。

ファイル一覧
------------
device_xyz_sample.csv
  - 装置XYZ 固有フォーマット（参考実装用）
  - タブ区切り、先頭5行ヘッダ、高さ単位 nm、グリッド 100×100
  - 対応ローダー: custom_surfaces/device_xyz.py (DeviceXyzSurface)
  - 使用 config: sample_inputs/config_device_xyz.yaml

device_vk6_sample.csv
  - Keyence VK-X シリーズ（VK-6000/VK-X1000 等）出力フォーマット
  - エンコード: Shift-JIS
  - ヘッダ 15 行（ピクセルサイズ・単位をヘッダから自動取得）
  - グリッド: 2048×1536、ピクセルサイズ: 0.136 μm、高さ単位: μm
  - 対応ローダー: custom_surfaces/device_vk6.py (DeviceVk6Surface)
  - 使用 config: sample_inputs/config_device_vk6.yaml

BRDF_BTDF_LightTools.bsdf
  - LightTools / MiniDiff Software 出力フォーマット（.bsdf）
  - エンコード: ASCII / UTF-8
  - ヘッダ: ScatterAzimuth 361（0–360°）、ScatterRadial 91（0–90°）
  - データ: 24 ブロック（BRDF/BTDF × AOI 4 種 × 波長 3 種）
  - ブロック構造: AOI/POI/Side/Wavelength/ScatterType/TIS + 361 行 × 91 列タブ区切り BSDF 値
  - 対応リーダー: custom_bsdf_readers/lightools_bsdf.py (LightToolsBsdfReader)
  - 使い方:
      bsdf simulate -c config.yaml --measured-bsdf sample_inputs/BRDF_BTDF_LightTools.bsdf
    または config.yaml の measured_bsdf: セクションで指定（config_device_vk6.yaml 参照）

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
