import streamlit as st

import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# チャンピオン名とIDの対応リストを辞書に変換
champion_dict = {
    "": 0,
    "アーゴット": 6,
    "アーリ": 103,
    "アカリ": 84,
    "アクシャン": 166,
    "アジール": 268,
    "アニビア": 34,
    "アニー": 1,
    "アフェリオス": 523,
    "アムム": 32,
    "アリスター": 12,
    "イブリン": 28,
    "イラオイ": 420,
    "イレリア": 39,
    "ウーコン": 62,
    "ウディア": 77,
    "エイトロックス": 266,
    "エコー": 245,
    "エズリアル": 81,
    "エリス": 60,
    "オラフ": 2,
    "オリアナ": 61,
    "オレリオン・ソル": 136,
    "オーン": 516,
    "カーサス": 30,
    "カイ＝サ": 145,
    "カ＝ジックス": 121,
    "カ＝サンテ": 897,
    "カサディン": 38,
    "カシオペア": 69,
    "カタリナ": 55,
    "カリスタ": 429,
    "カルマ": 43,
    "ガリオ": 3,
    "ガングプランク": 41,
    "ガレン": 86,
    "キヤナ": 246,
    "キンドレッド": 203,
    "グレイブス": 104,
    "ケイトリン": 51,
    "ケイル": 10,
    "ケイン": 141,
    "ケネン": 85,
    "コーキ": 42,
    "コグ＝マウ": 96,
    "サイオン": 14,
    "サイラス": 517,
    "サミーラ": 360,
    "シヴィア": 15,
    "シャコ": 35,
    "シヴァーナ": 102,
    "シェン": 98,
    "シンドラ": 134,
    "シンジド": 27,
    "シン・ジャオ": 5,
    "ジグス": 115,
    "ジリアン": 26,
    "ジャックス": 24,
    "ジャーヴァンⅣ": 59,
    "ジャンナ": 40,
    "ジン": 202,
    "ジンクス": 222,
    "スカーナー": 72,
    "スウェイン": 50,
    "スモルダー": 901,
    "スレッシュ": 412,
    "ゼド": 238,
    "ゼラス": 101,
    "ゼリ": 221,
    "ゾーイ": 142,
    "タリック": 44,
    "タリア": 163,
    "タム・ケンチ": 223,
    "タロン": 91,
    "ティーモ": 17,
    "ツイステッド・フェイト": 4,
    "トゥイッチ": 29,
    "トリスターナ": 18,
    "トリンダメア": 23,
    "トランドル": 48,
    "ナサス": 75,
    "ナフィーリ": 950,
    "ナミ": 267,
    "ニーコ": 518,
    "ニダリー": 76,
    "ヌヌ＆ウィルンプ": 20,
    "ノクターン": 56,
    "ノーチラス": 111,
    "ハイマーディンガー": 74,
    "パンテオン": 80,
    "フィオラ": 114,
    "フィドルスティックス": 9,
    "フィズ": 105,
    "ブラッドミア": 8,
    "ブラウム": 201,
    "ブリッツクランク": 53,
    "ブライアー": 233,
    "ブランド": 63,
    "ベイガー": 45,
    "ヘカリム": 120,
    "ベル＝ヴェス": 200,
    "ボリベア": 106,
    "マオカイ": 57,
    "マスター・イー": 11,
    "マルザハール": 90,
    "マルファイト": 54,
    "ミス・フォーチュン": 21,
    "ミリオ": 902,
    "モルガナ": 25,
    "モルデカイザー": 82,
    "ユーミ": 350,
    "ヨネ": 777,
    "ヨリック": 83,
    "ライズ": 13,
    "ラカン": 497,
    "ラムス": 33,
    "ランブル": 68,
    "リサンドラ": 127,
    "リヴェン": 92,
    "リリア": 876,
    "ルシアン": 236,
    "ルブラン": 7,
    "ルル": 117,
    "レク＝サイ": 421,
    "レネクトン": 58,
    "レナータ・グラスク": 888,
    "レオナ": 89,
    "レル": 526,
    "レンガー": 107,
    "ヴァイ": 254,
    "ヴァルス": 110,
    "ヴェイン": 67,
    "ヴェクス": 711,
    "ヴェル＝コズ": 161,
    "ヴィエゴ": 234,
    "ザイラ": 143,
    "ザック": 154,
    "ザヤ": 498,
    "ジリアン": 26,
    "ジンクス": 222
}


st.header('LOL勝利予想AI')

# チャンピオン名からIDを取得する関数
def get_champion_id(champion_name):
    return champion_dict.get(champion_name, "チャンピオンが見つかりません")

# モデルをロードする関数
def load_model(model_filename='lolperfect_AI.pkl'):
    return joblib.load(model_filename)

# 試合を予測する関数
def predict_match(loaded_model, champion_ids_100, champion_ids_200, X):
    new_match_data = {
        'champion_ids_100': champion_ids_100,
        'champion_ids_200': champion_ids_200
    }

    max_champions = 5
    for team in [100, 200]:
        key = f'champion_ids_{team}'
        for i in range(max_champions):
            new_match_data[f'champion_{team}_{i+1}'] = new_match_data[key][i]

    new_match_data.pop('champion_ids_100')
    new_match_data.pop('champion_ids_200')

    new_match_df = pd.DataFrame([new_match_data])

    # 特徴量名を学習時のものと一致させる
    new_match_df.columns = [
        'champion_100_1', 'champion_100_2', 'champion_100_3', 'champion_100_4', 'champion_100_5',
        'champion_200_1', 'champion_200_2', 'champion_200_3', 'champion_200_4', 'champion_200_5'
    ]

    # 足りない特徴量を補完
    for col in X.columns:
        if col not in new_match_df.columns:
            new_match_df[col] = 0

    new_match_df = new_match_df[X.columns]

    prediction = loaded_model.predict(new_match_df)
    win_probability = loaded_model.predict_proba(new_match_df)

    return prediction, win_probability


champion_ids_100 = []
champion_ids_200 = []

left_col, right_col = st.columns(2)

champion_names = list(champion_dict.keys())

with left_col:
    st.header("BLUEチーム")
    roles = ["TOP", "JG", "MID", "ADC", "SUP"]
    for i, role in enumerate(roles):
        champion_name = st.selectbox(f"{role}", champion_names, key=f"left{i}")
        champion_id = get_champion_id(champion_name)
        if champion_id != "チャンピオンが見つかりません":
            champion_ids_100.append(champion_id)

with right_col:
    st.header("REDチーム")
    roles = ["TOP", "JG", "MID", "ADC", "SUP"]
    for i, role in enumerate(roles):
        champion_name = st.selectbox(f"{role}", champion_names, key=f"right{i}")
        champion_id = get_champion_id(champion_name)
        if champion_id != "チャンピオンが見つかりません":
            champion_ids_200.append(champion_id)

if st.button("予測を実行"):
    model_filename = 'lolperfect_AI.pkl'
    loaded_model = load_model(model_filename)

    X = pd.DataFrame(columns=[
        'team_gold_100', 'team_gold_200', 'team_damage_100', 'team_damage_200',
        'team_vision_100', 'team_vision_200', 'team_kills_100', 'team_kills_200',
        'team_deaths_100', 'team_deaths_200', 'team_assists_100', 'team_assists_200',
                'champion_100_1', 'champion_100_2', 'champion_100_3', 'champion_100_4', 'champion_100_5',
        'champion_200_1', 'champion_200_2', 'champion_200_3', 'champion_200_4', 'champion_200_5'
    ])

    if len(champion_ids_100) == 5 and len(champion_ids_200) == 5:
        prediction, win_probability = predict_match(loaded_model, champion_ids_100, champion_ids_200, X)

        win_probability_reversed = win_probability[0][::-1]
        st.write(f'予測結果: {"Team BLUE Wins" if prediction[0] == 1 else "Team RED Wins"}')
        st.write(f'勝利確率: {win_probability_reversed, win_probability}')
    else:
        st.write("各チームに5つのチャンピオンIDを入力してください。")