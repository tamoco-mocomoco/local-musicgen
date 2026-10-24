# サンプル音源

local-musicgen (ACE-Step 1.5) で生成した曲のサンプル。**すべて AI (ACE-Step 1.5) で生成したもの**で、人の手による作曲・演奏・編集はしていない
(ループ版のみ、サービスのループ加工を適用)。

生成環境: MacBook Pro (M2 Max / 64GB)、ACE-Step 1.5 (turbo、inference_steps 8)、MLX。MP3 (192kbps) に変換して置いている。

## 曲

| ファイル | 長さ | 内容 | 設定 |
|---|---|---|---|
| [cafe-jazz-trio.mp3](cafe-jazz-trio.mp3) | 26.8 秒 | カフェ風のジャズピアノトリオ | 言語モデル 0.6B (caption の書き換えなし)、bpm 120、F Major、seed 21。末尾の無音をカット |
| [nap-lullaby-loop.mp3](nap-lullaby-loop.mp3) | 22.0 秒 | 昼寝シーンのピアノとオルゴール (ループ加工版) | 言語モデル 0.6B、bpm 70、F Major、seed 42、`loop=true` |
| [happy-jingle.mp3](happy-jingle.mp3) | 8.9 秒 | 短いジングル | 言語モデル 0.6B、seed 3、長さ指定 10 秒 |

caption (曲の説明) はそれぞれ次のとおり。

- cafe-jazz-trio: `cozy jazz piano trio, warm acoustic piano melody, walking upright bass, soft brushed drums, relaxed medium swing, cafe afternoon, gentle, playful, instrumental`
- nap-lullaby-loop: `gentle lullaby, soft felt piano and music box, slow, calm, sleepy afternoon nap, warm, peaceful, minimal, instrumental, no drums`
- happy-jingle: `short happy jingle, glockenspiel and ukulele, instrumental`

ループ版の MP3 は聴いて確かめる用。MP3 は形式の性質上、頭と終わりにごく短い無音が入るので、ゲームで継ぎ目なくループさせるときはサービスが出力する WAV を使う。

## 省メモリ設定の聴き比べ ([comparison/](comparison/))

同じ caption (上の nap-lullaby と同じ)・bpm 70・F Major・seed 42・30 秒で、設定だけを変えたもの。

| ファイル | 設定 | 生成中のメモリのピーク |
|---|---|---|
| [1-original-precision-no-lm.mp3](comparison/1-original-precision-no-lm.mp3) | 元の精度 (音声復元 float32・一括)、言語モデルなし | 42.1GB |
| [2-lean-no-lm.mp3](comparison/2-lean-no-lm.mp3) | 省メモリ設定、言語モデルなし | 14.7GB |
| [3-lean-lm-0.6b.mp3](comparison/3-lean-lm-0.6b.mp3) | 省メモリ設定、言語モデル 0.6B (このサービスの既定) | 15.1GB |
| [4-lean-lm-1.7b.mp3](comparison/4-lean-lm-1.7b.mp3) | 省メモリ設定、言語モデル 1.7B | 17.4GB |

- 1 と 2 は言語モデルを使っていないので曲の中身は同じで、違いは音声復元の精度だけ (元の WAV どうしで SNR 約 52dB)
- 3 と 4 は言語モデルが曲の構成を考えるので、同じ seed でも別の曲になる
- 省メモリ設定の中身と計測の詳細はリポジトリの README を参照

## ライセンス

このフォルダの音源は [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.ja) とする。クレジット表記なしで、商用を含めて自由に使ってよい。

- 生成に使った ACE-Step 1.5 はコード・モデルとも MIT ライセンスで、生成した音楽の商用利用が認められている
- ACE-Step の README は、利用者に AI 生成であることの明示と、既存の作品に似ていないかの確認を求めている。使う場合も同様にしてほしい
- 既存の曲との類似について保証はしない。利用は自己責任で
