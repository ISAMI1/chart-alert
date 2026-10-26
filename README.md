# chart-alert

平日16:05（日本時間）に GitHub Actions が watchlist.csv の銘柄の日足を判定し、latest_result.json に保存します。
Claude の定期タスク（平日16:40）がこのファイルを読み、シグナル日報に記録します。

- 銘柄の変更: watchlist.csv を編集（code=証券コード4桁, name=銘柄名）
- 手動実行: Actions タブ → daily-check → Run workflow
- 判定パターン: カップウィズハンドル / 逆三尊 / ダブルボトム / ゴールデンクロス(25日・75日) / 三角保ち合い上放れ
