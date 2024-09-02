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
