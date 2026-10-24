# local-music

ローカルで BGM や歌入りの曲を作る音楽生成サービス。[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) を省メモリ設定で動かし、
HTTP API・CLI・MCP サーバーで呼べるようにしたもの。画像生成の [local-imagegen](https://github.com/tamoco-mocomoco/local-imagegen) と同じ作りで、
組み合わせてゲームや動画の素材をローカルで完結させることを想定している。

```
run.py              起動スクリプト (標準ライブラリのみ)。ACE-Step の仮想環境で uvicorn を起動
server/app.py       FastAPI (GET /health, POST /generate, POST /load, POST /unload, /outputs)
server/engine.py    ACE-Step の読み込み・生成・自動解放 (省メモリ設定込み)
server/loop.py      ループ用の後処理と末尾の無音カット
gen_music.py        CLI (標準ライブラリのみ)
mcp_server.py       Claude Code / Claude Desktop 用 MCP サーバー
samples/            生成したサンプル音源 (MP3、CC0)
```

## サンプル

[samples/](samples/) に、このサービスで生成した曲を置いている (すべて AI 生成、CC0)。

- [カフェ風のジャズピアノトリオ](samples/cafe-jazz-trio.mp3) / [昼寝シーンのループ曲](samples/nap-lullaby-loop.mp3) / [短いジングル](samples/happy-jingle.mp3)
- [省メモリ設定の聴き比べ](samples/comparison/) (元の精度・省メモリ設定・言語モデル 0.6B / 1.7B)

曲ごとの caption・seed・設定は [samples/README.md](samples/README.md) にある。

## サービス共通の約束事

local-imagegen と揃えている。

- `GET /health` で状態 (読み込み済みか、メモリ、設定) を返す
- `POST /generate` は生成物をファイルに保存し、ファイル名・パス・seed などのメタ情報を返す (`return_base64` で中身も返す)
- `POST /load` / `POST /unload` でモデルの先読み・解放ができる
- モデルは要求時に読み込み、使われなければ自動で解放する (`IDLE_UNLOAD_SECONDS`、既定 300 秒)
- 重い処理は 1 件ずつ実行する

## メモリと速度 (M2 Max、30 秒の BGM)

| 状態 | メモリ |
|---|---|
| 待機中 (モデル未読み込み) | 約 0.1〜0.5GB |
| 読み込み後 | 約 12GB |
| 生成中のピーク | 約 15GB |
| 読み込みの瞬間 | 約 21GB (数十秒だけ) |

初回の読み込みは約 40 秒、以降は 1 曲約 9 秒。メモリ 24GB のマシンでは、画像生成サーバーのモデルを解放してから使うこと。

省メモリ設定 (`MUSIC_LEAN=1`、既定) の中身は次のとおり。ACE-Step 標準の設定では生成中に約 45GB まで膨らむ。

- PyTorch 側 (条件付け・テキストエンコーダ) は CPU に置く。重い計算は MLX が GPU で行う
- MLX へ変換した後、PyTorch 側の DiT decoder と VAE の重みを解放する
- MLX の DiT を bf16 にする (配布されている重みが bf16 なので劣化なし)
- MLX VAE を fp16 にし、分割サイズ 192 でデコードする (標準との差は SNR 約 52dB で聞き分けられない)
- MLX のキャッシュ上限を 1GB にする
- 言語モデルは 0.6B (1.7B より約 2GB 軽く速い)

## 準備

Apple Silicon の Mac を前提に検証している (MLX を使う)。

1. ACE-Step 1.5 を取得して依存を入れる

   ```bash
   git clone https://github.com/ace-step/ACE-Step-1.5.git
   cd ACE-Step-1.5 && uv sync --frozen
   ```

2. 重みを取得する (初回の起動時に自動で取得される分もある)。言語モデル 0.6B は次のコマンドで取れる

   ```bash
   HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='checkpoints/acestep-5Hz-lm-0.6B')"
   ```

3. このフォルダに `.env` を作って ACE-Step の場所を書く (`.env.example` を参照)

   ```bash
   cp .env.example .env   # ACESTEP_HOME=/path/to/ACE-Step-1.5 を書き換える
   ```

## 使い方

```bash
python3 run.py                     # http://127.0.0.1:8100
```

別ターミナルで:

```bash
python3 gen_music.py "cozy jazz piano trio for a cafe scene, instrumental" --bpm 120 -o cafe.wav
python3 gen_music.py "gentle felt piano and music box lullaby, instrumental, no drums" --loop -o nap.wav
python3 gen_music.py "upbeat kids pop song, ukulele, handclaps" --vocal en --lyrics lyrics.txt -o song.wav
python3 gen_music.py --status      # 状態とメモリ
python3 gen_music.py --unload      # すぐにメモリを空ける
```

- `--loop`: 末尾のフェードと無音を切り、冒頭と同じくらいの音量の位置で終わらせ、終端を始端にクロスフェードする。ゲームの BGM 向け
- 言語モデルを使うと曲が指定より短めに終わることがある。末尾の無音は自動で切り、実際の長さを返す。指定の長さを最後まで使い切りたいときは `--no-lm`
- caption は英語で、曲調・楽器・テンポ感・用途を書くと効く
- 言語モデルは生成の前に caption を書き換える。指定したジャンルや楽器から離れるときは `--no-rewrite` (HTTP / MCP では `rewrite_caption: false`) で書き換えを止める

## Claude から使う (MCP)

```bash
claude mcp add music -- uv run --no-project --with "mcp>=1.2" python "$PWD/mcp_server.py"
```

ツールは `generate_music` / `server_status` / `load_model` / `unload_model`。サーバー (`python3 run.py`) は別に起動しておく。

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `ACESTEP_HOME` | (必須) | ACE-Step 1.5 のチェックアウト。その中の `.venv` と `checkpoints/` を使う |
| `HOST` / `PORT` | `127.0.0.1` / `8100` | 待ち受け |
| `IDLE_UNLOAD_SECONDS` | `300` | 使われなければモデルを解放するまでの秒数。`0` で解放しない |
| `MUSIC_LEAN` | `1` | 省メモリ設定。`0` で ACE-Step 標準 |
| `MUSIC_LM` | `acestep-5Hz-lm-0.6B` | 言語モデル。`acestep-5Hz-lm-1.7B` や `none` |
| `MUSIC_INFERENCE_STEPS` | `8` | 拡散のステップ数 (turbo モデルの想定値) |
| `OUTPUT_DIR` | `./outputs` | 保存先 |
| `MUSIC_URL` | `http://localhost:8100` | CLI と MCP サーバーの接続先 |

## ライセンス

このリポジトリのコードは [MIT License](LICENSE)、[samples/](samples/) の音源は CC0 1.0。ACE-Step 1.5 本体とモデルの重みは同梱しておらず、各自が取得する。
ACE-Step 1.5 のライセンスと、生成した曲の利用条件は利用者が公式リポジトリで確認すること。
生成した曲の利用に関する責任は利用者にある。
