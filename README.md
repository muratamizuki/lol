プロジェクト概要　設計ドキュメントに記載

リポジトリ概要
  lol.ipynb
  学習用のデータ取得
  streamlit.py
  アプリ本体
  .csv
  学習用データ
  .pkl
  学習済みモデル

環境構築
  pip install streamlit

  pip install pandas

  pip install joblib

  pip install scikit-learn

  もし動かなければstreamlitを以下のバージョンにしてください
  Streamlit 1.36.0

アプリ
  環境構築終了後streamlit run streamlit.pyで起動

データ収集や学習
  データ収集
    lol.ipynbのapi_keyを自身が取得したRIOTAPIKEYに置き換えてください。
  学習
    取得したデータもしくはすでにあるデータを使ってください、lol.ipynbの最後の方のコードを変えれば簡単にできます。

現在の問題点
  ~~キャリー系のAPジャングラー（ニダリー、カーサス等）がREDに存在するときRED側に非常に有利な結果が出ています。~~
  ↑修正済み
  
