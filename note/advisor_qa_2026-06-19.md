# 问答记录


---

## Q1：为什么不用扩散模型，直接用逆动力学解线性方程？

导师认为 PID 底层已经在做逆动力学，为什么不直接解方程 move it。

**答**：确实在做逆动力学——PID 外环输出 EEF 速度，robosuite OSC 内环用 Jacobian 伪逆+动力学模型转关节力矩。但"直接解方程"有三个前提不成立：(1) 机器人动力学不是线性方程 M(q)q̈+C+G=τ，M 随 q 变化；(2) 接触动力学（抓取摩擦、碰撞）不在方程里，Learned ID 模型验证了这个问题是 ill-posed 的多对一映射；(3) 实验证明无反馈的纯前馈不如 PID——ki=0 时 lift_failed 翻 4 倍。结论：我们已经在做逆动力学，外环 PID 提供的是纯前馈没有的纠错能力。

---

## Q2：PID 是什么，PID 外环是什么？

**答**：PID 是比例-积分-微分反馈控制器。P=误差越大输出越大，I=误差持续存在则累积纠正（消稳态误差），D=抑制震荡（你的系统里 kd=0 因为 robosuite 内部已有阻尼）。"外环"是相对于 robosuite 内置的 OSC 内环说的——外环输出 EEF 速度指令，内环把它转成关节力矩。纯前馈只用内环，加外环就是 PID 闭环控制。

---

## Q3：原论文为什么用扩散模型？

**答**：原论文没有 `get_object_pose`——他们只有 128×128 RGB 像素。扩散模型替代的是"从像素推断物体在哪+预测动作"整个管线。你的系统有 oracle 位姿，PID 可以直接算 error。不是扩散模型不如 PID，是对的任务难度不同。实测确实扩散模型 gate=0% 说明它在结构化任务上不如手工控制器。

---

## Q4：那只能在仿真里跑了吗？依赖 oracle 位姿？

**答**：不是。Stage 27 验证过——YOLO+双目视觉 vs Oracle GT，差距仅 1.7pp（统计不显著）。真正阻碍 sim-to-real 的不是物体检测，是夹爪触觉和相机标定精度——这两个是硬件问题不是架构问题。

---

## Q5：以后换成 YOLO+双目视觉，任务泛化做了没？

**答**：感知管线已有（detector_perception.py + dual_camera_triangulation.py），Stage 27 是离线验证，接入 PID 实时回路需要工程调试（YOLO 推理延迟、遮挡处理）。TaskSpec 统一架构完成了但执行质量有 gap——3-block 有死循环 bug，Hanoi grasp 步数过长，Drawer 未开始。

---

## Q6：TaskSpec 是什么，为什么要用？

**答**：之前每个任务一份独立硬编码 eval 脚本（stacking 700 行，Hanoi 250 行，3-block 250 行），核心逻辑完全一样但 6-7 处硬编码不同——这就是 C2/C3 等 bug 的根源。TaskSpec 是把"这个任务怎么配"打包成一个数据类，一份 `run_task()` 统一执行。修一次 bug 三个任务都受益，加新任务不需要重写 250 行脚本。

---

## Q7：BFS 是通用的为什么还需要 internal_hanoi？

**答**：BFS 只统一了堆叠任务——它的算子集是 `{MoveOn, MoveToTable}`，没有 peg 概念。Hanoi 需要 `MoveFromPeg(disk, src, dst)`，BFS 的世界模型表达不了。所以 `internal_hanoi` 是独立的递归算法。这不是设计缺陷——符号规划的每个 domain 本来就需要自己的算子集。论文应诚实写："BFS handles stacking; Hanoi uses specialized solver; new domains need new operators."

---

## Q8：有什么创新点？系统很单一，任务很简单。

**答**：不是范式革命，是三个工程上有用的发现 + 一个诚实的评估方法：(1) Gate attribution 作为评估指标——不报 success rate 报实际控制步数，这个指标本身可能比结果更有用；(2) 训练无关架构的可行性——PID+BFS+Qwen 零训练，三任务 80-93%；(3) OperatorMonitor 的实际价值——preplace→NOT_HOLDING 根因诊断。任务简单反而暴露了尴尬事实：26.8M 扩散模型 gate=0%，0 参数 PID 94%。

---

## Q9：对 gate=0% 持怀疑，推翻"一次规划一次执行"也不认可

**答**：怀疑是对的。Gate=0% 不是"扩散模型永远不行"，是"在这个具体设定下（2-block、200 demo、tight gate、零符号条件）VLA 做不出被 gate 接受的动作"。B5 有符号条件时 gate=47% 说明架构是能学的。70%+ 成功率是 scripted teacher 在兜底。"推翻一次规划一次执行"确实说过头了——3-block direct=0% 根因是 env horizon 没设对，horizon 修好直接 93%。Replan 的价值是工程上的——证明了 OperatorMonitor 可以检测崩塌并恢复。

---

## Q10：下一步计划应该先做复杂任务，同时让规划器作用更明显

**答**：当前规划器的问题：2-block 只需 1 个 action，Hanoi 用了专用递归算法，BFS 的价值没体现。需要设计一个"没有 BFS 就解不了"的任务。建议：(1) 4-block reverse stacking（初始已堆叠，目标不同顺序——需要先拆后堆，BFS 必须搜索反直觉路径）；(2) 定量对比 BFS vs InternalStackingPlanner vs 无规划（固定动作序列）在不同初始状态下的成功率。这样同时解决"任务简单"和"规划器隐形"两个问题。
