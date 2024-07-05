# wrs_algorithm

## 環境構築方法
必要なリポジトリのクローンする。
```bash
cd /workspace/py3_ws/src
git clone git@github.com:keio-smilab21/wrs_algorithm21.git
vcs import /workspace/py3_ws/src < /workspace/py3_ws/src/wrs_algorithm21/py3_ws.rosinstall
```

ビルドを実行する。
```
cd /workspace/py3_ws
catkin build
```

## 開始方法
```bash
rviz -d /workspace/py3_ws/src/wrs_algorithm21/config/wrs_visualize.rviz
/workspace/py3_ws/src/wrs_algorithm21/scripts/start_all.sh
```
