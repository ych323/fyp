# GitHub上传说明

只上传本仓库目录内容，不上传相邻的`local_assets`或总文件夹。建议使用GitHub Desktop选择本目录，检查Changes后再提交。网页上传不会替你执行`.gitignore`，不能随意拖入恢复后的整个目录。

上传前执行`python scripts/check_publish.py`。仓库准备时不包含原始评论、训练文件或权重；本地恢复后，这些文件会被`.gitignore`忽略。

如果使用命令行，先手动在GitHub创建仓库，再按照GitHub给出的连接命令操作。仓库地址、公开或私有和代码许可由你决定。此整理步骤没有创建远程仓库、提交或推送。

GitHub普通Git单文件上限为100 MiB；网页上传上限为25 MiB。权重不适合普通Git。将来如需分享，可在确认许可和存储限额后选择LFS或模型托管服务。本次保留checkpoint清单与SHA256，但不生成不存在的下载链接。

官方说明：https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

## 包含与不包含

本仓库可查看全部已有结果和中英文报告，保留抽样ID、下载清单和模型结构。`../local_assets/`保存原实验的数据与权重；`scripts/restore_local_assets.py`可校验后恢复。只克隆GitHub仓库不会获得已训练权重。

下载脚本仅恢复固定版本的公开上游资产，不能恢复本地训练权重。如果上游不可用，应自行取得合法访问，脚本不会绕过权限。

历史实验源码`src/`、配置、结果和ID文件按原字节保留。仓库整理仅调整安装脚本使用仓库内`.venv`，并将下载入口改为固定历史manifest。未修改历史指标或声称重新训练。
