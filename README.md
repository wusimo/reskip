# ReSkip

本仓库用于实现和验证基于 Attention Residuals 的三条主线：

- `ReSkip`：基于块级路由的重要性跳过
- `ReLoop`：共享块循环与自适应深度
- `StarVLA + AttnRes`：在 VLA 骨干上的自适应计算

当前训练与评测主线已经迁移到：

- [flame](/home/user01/Minko/reskip2/reskip/flame)
- [flash-linear-attention](/home/user01/Minko/reskip2/reskip/flash-linear-attention)
- [starVLA](/home/user01/Minko/reskip2/reskip/starVLA)

## 文档入口

- 总实验规划、训练指令、评测指令、已实现内容、排错记录：见 [PLAN.md](/home/user01/Minko/reskip2/reskip/PLAN.md)
- 训练说明草稿与历史记录：见 [1232.txt](/home/user01/Minko/reskip2/reskip/1232.txt)

## 当前建议

- `ReSkip` 与 `ReLoop` 的 LM 训练统一使用仓库内这份 `flame`
- 不再使用单独拉下来的 `baseflame`
- 旧的异常 checkpoint 仅用于排查，不再作为正式结果

## 目录概览

```text
reskip/
├── flame/                    # flame 训练器与配置
├── flash-linear-attention/   # FLA 模型实现
├── experiments/              # 分析、导出、评测脚本
├── starVLA/                  # VLA 训练与部署
├── paper/                    # 论文与参考材料
├── PLAN.md                   # 统一实验计划
└── 1232.txt                  # 训练命令与补充说明
```
