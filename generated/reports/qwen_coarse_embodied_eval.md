# Qwen Coarse Embodied 评测报告

> 日期：2026-06-26 | Task 5：Qwen 生成粗粒度程序 → 验证 → 执行

## 总览

| 指标 | 结果 |
|------|:---:|
| 解析成功 | 10/10 (100%) |
| 验证通过 | 9/10 (90%) |
| 禁止操作 | 0/10 (0%) |
| BFS 重排触发 | 1/10 (10%) |
| BFS 重排成功 | 1/1 (100%) |
| 2-block 执行成功 | 22/25 (88%) |

---

## 逐程序结果

### 2-Block（已执行）

| ID | 指令 | 验证 | 执行 | 失败类型 |
|----|------|:---:|:---:|------|
| Q2_01 | 把 cube1 叠到 cube2 上 | ✅ | 5/5 | — |
| Q2_02 | 捡起 cube1 放到 cube2 上 | ✅ | 5/5 | — |
| Q2_03 | 把 cube2 放到 cube1 上面 | ✅ | 4/5 | 1 nocontact |
| Q2_04 | 把 cube1 移动到 cube2 上 | ✅ | 4/5 | 1 nocontact |
| Q2_05 | 把 cube1 卸下来放到桌上 | ✅ | 4/5 | 1 slip |

**2-block 总计：22/25 (88%)**

### 3-Block（仅验证）

| ID | 指令 | 验证 | 重排 | 备注 |
|----|------|:---:|:---:|------|
| Q3_01 | cube1 叠 cube2，然后 cube2 叠 cube3 | ❌ | ✅→bottom-up | Top-down 被捕获并修复 |
| Q3_02 | 搭塔：cube3 底座，cube2 中层，cube1 顶层 | ✅ | — | Qwen 正确生成 bottom-up |
| Q3_03 | 把所有方块叠起来，cube1 在最上面 | ✅ | — | 顺序正确 |
| Q3_04 | 先把 cube2 放 cube3 上，再把 cube1 放 cube2 上 | ✅ | — | 显式 bottom-up |
| Q3_05 | cube3 在桌上，cube1 在 cube3 上，cube2 在 cube1 上 | ✅ | — | 复杂排序正确 |

**3-block 验证：4/5 直接通过，1/5 BFS 修复通过**

---

## Qwen 实际输出示例

### Q2_01 "把 cube1 叠到 cube2 上"
```json
{"version":"1.0","steps":[
  {"op":"ensure","predicate":["clear","cube2"]},
  {"op":"pick","object":"cube1"},
  {"op":"place","object":"cube1","target":"cube2"}
]}
```
→ 执行：5/5 ✅

### Q3_02 "搭塔：cube3 底座，cube2 中层，cube1 顶层"
```json
{"version":"1.0","steps":[
  {"op":"ensure","predicate":["clear","cube3"]},
  {"op":"pick","object":"cube2"},
  {"op":"place","object":"cube2","target":"cube3"},
  {"op":"ensure","predicate":["clear","cube2"]},
  {"op":"pick","object":"cube1"},
  {"op":"place","object":"cube1","target":"cube2"}
]}
```
→ 验证：✅（Qwen 识别出"底座"应最先堆，"顶层"应最后堆）

### Q3_01（被拒绝 → BFS 修复）
Qwen 原始输出（top-down）：
```json
{"version":"1.0","steps":[
  {"op":"ensure","predicate":["clear","cube2"]},
  {"op":"pick","object":"cube1"},
  {"op":"place","object":"cube1","target":"cube2"},
  {"op":"ensure","predicate":["clear","cube3"]},
  {"op":"pick","object":"cube2"},                // ← 被拒绝：cube2 上已有 cube1
  {"op":"place","object":"cube2","target":"cube3"}
]}
```

BFS 修复后（bottom-up）：
```json
[{"op":"pick","object":"cube2"},
 {"op":"place","object":"cube2","target":"cube3"},
 {"op":"pick","object":"cube1"},
 {"op":"place","object":"cube1","target":"cube2"}]
```

---

## 关键发现

1. **Qwen 从不输出禁止操作**（0/10）——reach/grasp/lift 完全不出现，prompt 约束有效
2. **Qwen 自动添加 ensure 门**——每个 pick 前自动生成 precondition 验证
3. **Qwen 多数情况正确排序**——4/5 的 3-block 程序自动使用 bottom-up
4. **88% 执行成功率**——Qwen 生成的 2-block 程序可直驱动物理执行，接近 oracle 水平
5. **BFS 重排覆盖剩余 1 例**——top-down 错误被捕获并自动修复
6. **3-block 执行跳过**——AtomicSkill 层失败（详见主报告 §2.2），与 Qwen 生成质量无关
