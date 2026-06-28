# Git 提交流程（faceCopy）

以下命令按顺序执行即可完成一次日常提交与推送。

## 0. 进入项目目录
cd /Users/winssion/Desktop/faceCopy

## 1. 查看当前分支与状态
git branch -vv
git status

## 2. 拉取远程最新代码（避免冲突）
git pull --rebase origin main

## 3. 查看改动内容（可选但推荐）
git status
git diff

## 4. 添加要提交的文件

方式 A：提交全部改动（常用）
git add .

方式 B：只提交指定文件（更安全）
git add README.md face_swap.py requirements.txt

## 5. 再次检查将提交的内容
git status

## 6. 提交
示例：
git commit -m "docs: update README"

常用提交信息前缀建议：
- feat: 新功能
- fix: 修复问题
- docs: 文档更新
- refactor: 重构
- chore: 维护类改动

## 7. 推送到 GitHub
git push origin main

## 8. 验证是否推送成功
git log --oneline -n 5
git remote -v


# 常用补充命令

## 查看哪些文件已暂存
git diff --cached --name-only

## 查看最近一次提交内容
git show --name-only --oneline

## 撤销还没提交的暂存（保留文件修改）
git restore --staged <文件名>

## 放弃某个文件未提交修改（慎用）
git restore <文件名>


# 你的日常最简版（推荐记住这 6 行）
cd /Users/winssion/Desktop/faceCopy
git pull --rebase origin main
git add .
git commit -m "chore: update project"
git push origin main
git status
