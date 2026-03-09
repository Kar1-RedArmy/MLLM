# Windows 拉取后仍跑旧代码：排查与修复

如果你已经 `git pull`，但训练入口仍报旧错误（例如 `from peft import ...` 与 `Cache` 导入报错），通常是 **拉取分支不对** 或 **执行到了旧目录**。

## 1) 先确认当前仓库与分支
在 `Anaconda Prompt` 执行：

```bat
cd C:\work\MLLM\MLLM
git rev-parse --show-toplevel
git branch --show-current
git branch -a
git log --oneline -n 5
```

建议使用 `work` 分支：

```bat
git fetch origin
git checkout work
git pull --rebase origin work
```

## 2) 确认文件是否是新版本

```bat
type UReader-main\pipeline\train.py | findstr /N "from peft import from transformers.training_args"
type UReader-main\scripts\fix_env_win.bat | findstr /N "Script path Train entry path PYTHONPATH"
```

期望：
- `train.py` 命中 `from transformers.training_args import TrainingArguments`
- `fix_env_win.bat` 命中 `Script path` / `Train entry path` / `PYTHONPATH`

## 3) 统一用仓库根目录一键脚本

```bat
cd C:\work\MLLM\MLLM
fix_env_win.bat MLLM
```

这个入口会转发到 `UReader-main\scripts\fix_env_win.bat`，避免误执行旧目录脚本。

## 4) 若 `git pull` 网络失败
若报 `Failed to connect to github.com port 443`：

```bat
git config --global --unset http.proxy
git config --global --unset https.proxy
git pull --rebase origin work
```

若公司网络受限，请切换可访问 GitHub 的网络（如手机热点）后重试。

## 5) 仍失败时请回传这三段输出
1. `git branch --show-current` 与 `git log --oneline -n 5`
2. 两条 `findstr` 命令输出
3. `fix_env_win.bat` 第 `[6/7]` 与 `[7/7]` 全部输出
