# 评测报告目录

运行以下命令后会在本目录生成 `latest.json`：

```bash
python -m evaluation.run --all --output reports/latest.json
```

JSON 报告包含三组检索消融实验、Agent 指标、逐条明细和 Bad Case。运行结果属于本地数据，
已通过 `.gitignore` 排除；本目录只提交这份说明，不提交可能包含业务问题文本的实际报告。
