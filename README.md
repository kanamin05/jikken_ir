# wrs_algorithm

## 環境構築方法
必要なリポジトリのクローンする。
```bash
cd /workspace/src
git clone git@github.com:keio-smilab21/wrs_algorithm21.git
vcs import /workspace/src < /workspace/src/wrs_algorithm21/workspace.rosinstall
```

ビルドを実行する。
```
cd /workspace
catkin build
```

## 開始方法
```bash
rviz -d /workspace/src/wrs_algorithm21/config/wrs_visualize.rviz
/workspace/src/wrs_algorithm21/scripts/start_all.sh
```

シミュレータを起動した状態で `scripts/start_all.sh` を実行すると、点群処理・物体検出・
制御ノードが起動し、Task1 → Task2a（点群版DQN）→ Task2b の順に実行する。
Task2aは統合前の `task2a_precise_run.py --dqn-first --local-finish` と
`pointcloud_obstacles.py _self_filter_radius:=0.23` を使う。
深度点群から床と障害物の形状を抽出し、DQNのQ値と経路ガードで移動する。
Task2aでは物体検出ラベル・BBoxを使わず、`sponge`等のラベルで停止することはない。
Task1/Task2bの物体認識には従来の検出器を使用する。
Task1 が失敗・中断した場合は次のタスクへ進まない。
Task2a がエラーまたは失敗を返した場合は、点群による障害物判定を終了し、
台車の正面へ0.1 m/sで約5秒直進する（約0.5 m）。その後は点群やmove_baseの経路計画を使わず、
TFの現在位置をもとに `go_throw_2a`（2.0, 3.5, 90度）へ移動し、Task2bに進む。
直進後のゴール移動は90秒で打ち切る。TFが取れない場合や到着できない場合、ROS終了時は停止する。
点群ノード自体はTask2bの物体認識用に残す。Task2a単体のdry-runではこの代替移動を行わない。
Task2b は棚に移動した後、有効な物体名と左右を含む `/message` の指示を最大約30秒待つ。

点群版のコードと学習済みモデルはこのリポジトリに同梱している。
モデルの既定パスは `config/task2a_policy_numpy.npz`。
元のアルゴリズムと重みは変更せず、Pythonの兄弟モジュールのimportと末尾空行だけを調整している。
元ファイルのSHA256は `scripts/wrs_algorithm/task2a_cloud/source_manifest.json` に記録している。
別の場所にある場合は次のように指定する。

```bash
/workspace/src/wrs_algorithm21/scripts/start_all.sh weights:=/path/to/task2a_policy_numpy.npz
```

エラー後の直進距離・移動速度・ゴール移動の制限時間は、起動時に変更できる。

```bash
/workspace/src/wrs_algorithm21/scripts/start_all.sh fallback_forward_distance:=0.5 fallback_speed:=0.1 fallback_timeout:=90
```

認識ノードが起動済みの場合は `roslaunch wrs_algorithm start_task.launch` で制御だけ起動できる。
Task2a 単体の確認には `roslaunch wrs_algorithm task2a_dqn.launch dry_run:=true`、
Task2a 単体の走行には `dry_run:=false` を指定する。
Task2a開始時に点群ノードと走行プロセスを起動し、終了時に両方を停止してから
Task2bまたはエラー時の代替移動へ進む。以前の手動起動プロセスは終了してから使う。
実行ごとの診断結果は `~/.ros/log/task2a-cloud-*/summary.json` に保存する。
点群版の既存テストは次で実行できる。

```bash
PYTHONPATH=scripts python3 -B -m unittest discover -s tests -v
```
