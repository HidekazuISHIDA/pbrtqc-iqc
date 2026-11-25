# 実データでの再現と検証の手順（VS Code ＋ Jupyter）

## この研究のコードの作り方と確かめ方
| 段階 | 誰が | 何をするか | 使うデータ |
|---|---|---|---|
| 1. 構築 | AI（Claude）＋著者の確認 | 解析方法・モニター画面・検証方法を作り、合成データで動作を確かめて GitHub に公開する | 合成データ（`scripts/00_make_synthetic_raw.py`） |
| 2. 環境づくり | AI | 実データで同じコードを動かせる VS Code ＋ Jupyter の環境と手順を用意する | なし |
| 3. 再現 | **人（著者）** | 公開されたコードを自分の PC に取り込み、実データで全ステップを実行する | 実データ（PC の外に出さない） |
| 4. 検証 | 人 | 結果表を、以前の解析の結果表とセルごとに照合する（`06_verify`） | 集計表のみ |

コードは 1 つで、デモと実データの違いは「どのデータを読むか」だけです（`pbrtqc/config.py`）。

## 0. 守ること
- 実データ（LIS・IQC の出力）は**この PC の外に出さない**。クラウド・チャット・GitHub に置かない。
- 実データは**読むだけ**。キャッシュと結果は作業フォルダ `work/` にでき、Git に入らない（`.gitignore`）。
- コミット前に `tools/guard.py` が自動で動き、データ・キャッシュ・出力付きノートブック・ID らしい数字を含むファイルのコミットを止める。
- ノートブックには集計表だけを表示する。患者ごとの行を表示するコードは書かない。

## 1. コードを取り込む（初回のみ、5 分）
1. GitHub Desktop で **File → Clone repository** → `HidekazuISHIDA/pbrtqc-iqc` を選び、保存先を `~/pbrtqc-iqc` にする
   （AI の作業フォルダ `~/pbrtqc` とは別のフォルダにする。再現を独立にするため）。
2. VS Code で **File → Open Folder** → `~/pbrtqc-iqc` を開く。右下に推奨拡張機能（Python、Jupyter）の案内が出たら入れる。
3. **Terminal → Run Task → 「1. 環境を作る（初回のみ）」**。`.venv` に固定した版のパッケージが入り、コミットの見張りが有効になる。

## 2. デモで動作を確かめる（約 10 分）
1. `notebooks/00_setup.ipynb` を開き、右上の **Select Kernel → .venv** を選んで上から実行する。`profile=demo` と表示されることを確認。
2. `01` → `06` を順に開いて上から実行する（または **Run Task → 「2. デモ：全ステップを実行」** のあと **「3. デモ：照合」**）。
3. `06_verify` で **43 / 43 件が一致** すれば、環境は正しく作れている。

## 3. 実データで再現する（数時間かかることがあります。大半は待ち時間）
1. `00_setup.ipynb` の「2. 実データの場所を設定する」のセルで、パスを確認して `WRITE = True` にして実行 → カーネルを再起動。
   - `lis_raw_dir`：LIS の出力（`*.xlsx`）が入ったフォルダ
   - `qc_raw_file`：IQC の出力ファイル
   - `reference_tables`：照合に使う以前の結果表（AI が解析した `~/pbrtqc/outputs/tables`）
2. `00` を上から実行し直し、`profile=real` と入力ファイルの一覧を確認する。
3. `01` → `05` を順に実行する（または **Run Task → 「4. 実データ：全ステップを実行」**）。途中で止まっても、`python -m pbrtqc.pipeline --from <ステップ名>` で続きから実行できる。

## 4. 検証する
1. `06_verify.ipynb` を実行する（または **Run Task → 「5. 実データ：照合」**）。
2. 判定が全件「一致」なら、論文の数字を、公開コードと実データから独立に再現できたことになる。
3. 記録として残すもの（どれも集計のみ）：`work/verify_report.csv`、`work/run_log*.json`（コードの版、パッケージの版、各ステップの時間、入力ファイルの SHA-256）。
4. 一致しない表があれば、`内容` 列（列・行数・値のどれが違うか）と `run_log` を確認する。パッケージの版の違いが最も多い原因（`00` で確認）。

## 論文での書き方（案）
> The analysis code and the monitoring screen were developed with an AI coding assistant on synthetic data and are publicly available.
> One author (H.I.) independently re-ran the published code on the original data on a local computer, following the published instructions,
> and all result tables agreed cell by cell with those reported (relative tolerance 10⁻⁹).
