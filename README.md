## OpenTS-Toolkit

本工具集旨在辅助处理 OpenTS 系列项目（包括 TFB、TAB 和 FoundTS）的运行结果。

> 📦 **按需下载组件：**  
> 如果你只需要其中的某个组件（例如TFB回归测试），请前往 [Releases 页面](https://github.com/yourname/OpenTS-Toolkit/releases) 下载，无需完整克隆整个仓库。

### TFB_regression_test 与 TAB_regression_test

`TFB_regression_test` 和 `TAB_regression_test` 目录分别用于对 TFB 和 TAB 项目的回归测试。使用方法如下：

- **放置位置**：请将对应目录放置于 TFB 或 TAB 项目的根目录下。
- **运行示例**（以 TFB 为例）：

```shell
(TFB) root@96444813d88c:/home/TFB# python TFB_regression_test/regression_test.py
```
> ⚠️ 请确保在TFB或TAB对应的虚拟环境中运行，并且运行目录为TFB或TAB根目录。

回归测试完成后，每次回归测试完成后，会在 `TFB_regression_test/val_result` 目录下生成一个以当前时间戳命名的子目录，包含以下内容：

- `{时间戳}.csv`：当前运行的回归测试结果。
- `{时间戳}_diff.csv`：本次运行结果与历史样板结果之间的差异。