# Coarse Reorder 评测报告

> 日期：2026-06-26 | Task 4：Verifier 拒绝 → BFS 重排修复

## 结论

**BFS reorder 成功将 top-down 3-block 顺序修复为 bottom-up。**

---

## 测试用例：Qwen Top-Down 3-Block

Qwen 对"把 cube1 叠到 cube2 上，再把 cube2 叠到 cube3 上"生成了 top-down 顺序：

```json
// 原始输出（被 Verifier 拒绝）
[{"op": "pick", "object": "cube1"},
 {"op": "place", "object": "cube1", "target": "cube2"},
 {"op": "pick", "object": "cube2"},           // ← 拒绝：cube2 上有 cube1，不 clear
 {"op": "place", "object": "cube2", "target": "cube3"}]
```

Verifier 拒绝原因：`step_3_pick: 'cube2' is not clear (has 'cube1' on top)`

## 修复管线

1. **提取目标栈**：on(cube1,cube2), on(cube2,cube3) → bottom→top = `["cube3", "cube2", "cube1"]`
2. **调用 BFS PlannerAdapter("bfs").solve(cubes, ["cube3","cube2","cube1"])**：
   - BFS 搜索到的最优计划：`MOVE CUBE2 TABLE CUBE3`, `MOVE CUBE1 TABLE CUBE2`
   - 搜索状态数：2
3. **转换为 coarse steps**：
```json
[{"op": "pick", "object": "cube2"},
 {"op": "place", "object": "cube2", "target": "cube3"},
 {"op": "pick", "object": "cube1"},
 {"op": "place", "object": "cube1", "target": "cube2"}]
```
4. **重新验证**：✅ CoarseOptionVerifier 通过

## 不可修复的情况

| 拒绝原因 | 重排结果 | 原因 |
|---------|:---:|------|
| step_1: self-place cube1 on cube1 | ❌ 不修复 | 结构错误，非排序问题 |
| step_0: 'reach' 是 phase-level，禁止 | ❌ 不修复 | 禁止操作，非排序问题 |
| step_0: unknown object 'cube99' | ❌ 不修复 | 对象不存在，非排序问题 |

## Bottom-Up 本已正确

Qwen 在 4/5 的 3-block 程序中已正确生成 bottom-up 顺序，无需 BFS 重排，直接通过验证。

## 验收

- [x] 至少修复 1 个 3-block top-down 排序问题：✅ Q3_01 修复成功
- [x] 报告 verifier 拒绝原因：✅ "cube2 is not clear (has cube1 on top)"
- [x] 报告 BFS 修复计划：✅ `[MOVE CUBE2 TABLE CUBE3, MOVE CUBE1 TABLE CUBE2]`
