# 3D 位置估计方案：双相机 YOLO + 回归

> 从 NSM (Neuro-Symbolic Methods) 项目中独立迁移的 3D 物体位置估计模块。
> 用于在桌面操作场景中，通过两个普通 RGB 相机估计物体的 3D 世界坐标。

---

## 一、方案概述

### 设计思路

本方案**不依赖深度相机或立体视觉**，而是用监督学习将双视角的 2D 检测结果直接回归到 3D 世界坐标。

```
传统方案:   RGB-D 相机 → 点云 → 3D 定位
            或
           双目相机 → 立体匹配/深度估计 → 3D 定位

本方案:    双相机 RGB → YOLO 2D检测 → 14维特征向量 → 回归模型 → (x, y, z) 世界坐标
```

### 核心思想

在固定相机配置的桌面操作场景中，物体 3D 位置和其在两个相机图像中的 2D 检测结果之间存在可学习的映射关系。这个映射不仅依赖几何（如近大远小），还可以通过回归模型从数据中习得。

### 为什么可行

- **双相机提供视差信息**：同一物体在两个视角的 bbox 差异隐式编码了深度
- **bbox 尺寸编码距离**：越近物体 bbox 越大（在物体尺寸已知的前提下）
- **末端执行器位置辅助**：ee 坐标告诉模型腕部相机当前在哪里
- **桌面场景约束**：深度范围可控（通常 0.3–1.0m），不会出现极端远近

---

## 二、组件详解

### 组件 1：YOLOv8 目标检测

| 项目 | 详情 |
|------|------|
| 基座模型 | `yolov8s.pt`（Ultralytics 预训练权重） |
| 输入 | 256×256 RGB，两个相机各一张 |
| 输出 | 每个物体：`(cls_id, x, y, w, h, confidence)` 像素坐标 |
| 训练数据 | 仿真环境截图 + 人工标注 bbox |

**两个相机的配置：**

| 相机 | 位置 | 作用 |
|------|------|------|
| 相机 1（agentview） | 固定在世界坐标系中的俯视位置 | 提供稳定的全局视角 |
| 相机 2（eye_in_hand）| 安装在机器人腕部 | 随末端执行器移动，提供近距离视角和动态视差 |

### 组件 2：3D 坐标回归器

这是整个方案的**核心**。由三个独立的回归模型组成，输入相同的特征向量，分别预测 world_x、world_y、world_z。

**输入特征向量（14 维）：**

```
[cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z]
```

| 特征 | 来源 | 含义 |
|------|------|------|
| `cls_id` | YOLO 类别 ID | 让模型学习不同物体的空间分布差异 |
| `px1, py1` | 相机 1 bbox 中心 | 物体在全局视角中的像素位置 |
| `w1, h1` | 相机 1 bbox 宽高 | 隐式编码物体距离（近大远小） |
| `conf1` | 相机 1 检测置信度 | 让模型学会处理不确定的检测 |
| `px2, py2` | 相机 2 bbox 中心 | 物体在腕部视角中的像素位置 |
| `w2, h2` | 相机 2 bbox 宽高 | 腕部视角中的尺寸信息 |
| `conf2` | 相机 2 检测置信度 | 腕部检测可靠性 |
| `ee_x, ee_y, ee_z` | 机器人末端执行器 | 隐式编码腕部相机当前的世界位姿 |

**回归器类型选择：**

| 模型 | 优点 | 缺点 |
|------|------|------|
| **RandomForest**（原方案使用） | 无需调参、不易过拟合、训练快 | 外推能力差、模型较大 |
| **XGBoost** | 精度更高、支持外推 | 需要调参 |
| **GradientBoosting** | 精度高、平滑 | 训练慢 |
| **MLP（神经网络）** | 可端到端训练、外推好 | 需要更多数据 |

**Fallback 机制：** 当 14 维特征预测失败时（如 cls_id 未知），降级为 13 维特征，并对 X 坐标加 +0.03 偏置修正。

### 辅助组件：多目标跟踪与关系匹配

- **匈牙利算法匹配**：基于 IoU 代价矩阵关联连续帧中的检测
- **异常检测**：bbox 跳变 > 20px 或位置跳变 > 10cm 时标记为离群
- **遮挡处理**：抓取中的物体用夹爪位置替代 YOLO 估计
- **关系匹配**：通过空间关系图将 YOLO track ID 映射到语义 ID（如 "blue cube_0" → "cube1"）

---

## 三、文件结构

```
pose_estimation/
├── README.md                  # 本文件
├── executor.py                # 核心流水线：yolo_estimate(), pixel_to_world_dual(), 跟踪, 数据采集
├── object_metadata.py         # 物体关系匹配：YOLO track ID ↔ PDDL 语义 ID
├── train_regressor.py         # [新建] 回归器训练脚本（原项目缺失，此处补全）
├── config/
│   └── example.yaml           # 配置文件示例（Hanoi 任务）
└── yolo_training/
    ├── train_yolo.ipynb       # YOLOv8 训练 notebook（仿真数据标注 + 训练）
    └── data.yaml              # YOLO 数据集配置示例
```

---

## 四、完整训练流程

### 阶段 A：训练 YOLOv8 目标检测

**Step A1：采集和标注图像**

在仿真环境（robosuite/MuJoCo）中运行场景，从两个相机截图。需要采集：
- 不同物体位置、不同光照条件的场景
- Hanoi 任务原始数据：约 133 张图片，3 个类别

标注工具任选其一：

| 工具 | 特点 |
|------|------|
| [Roboflow](https://roboflow.com) | 在线平台，自动增强、版本管理（原方案使用） |
| [LabelImg](https://github.com/HumanSignal/labelImg) | 本地标注，导出 YOLO 格式 |
| [CVAT](https://www.cvat.ai) | 在线/自部署，适合大规模标注 |

标注格式（YOLOv8 normalized）：每张图片对应一个 `.txt`，每行：

```
<class_id> <x_center> <y_center> <width> <height>
# 所有值归一化到 [0, 1]
```

**Step A2：配置数据集**

参考 [yolo_training/data.yaml](yolo_training/data.yaml)：

```yaml
names:
- cube1
- cube2
- cube3
nc: 3
train: ../train/images
val: ../valid/images
test: ../test/images
```

**Step A3：数据增强（推荐）**

- 水平翻转、随机旋转 ±15°、随机剪切 ±5°
- 自动缩放至 640×640
- 原始 133 张图 → 增强后 80 train / 27 val / 27 test

**Step A4：训练**

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=800 plots=True
```

**Step A5：导出模型**

```bash
cp runs/detect/train/weights/best.pt models/yolo/hanoi_yolo.pt
```

### 阶段 B：采集回归器训练数据

这是整个流程中**最关键的一步**。`executor.py` 内置了数据采集功能：

**Step B1：修改 executor.py 开启采集模式**

在 `Executor_Diffusion.__init__()` 中设置：

```python
self.save_data = True  # 开启数据采集
```

或在调用 `yolo_estimate()` 时传入参数：

```python
self.yolo_estimate(image1, image2, save_video=True, cubes_obs=cubes_obs, ee_pos=ee_pos)
```

**Step B2：运行数据采集**

在仿真环境中执行各种操作，让物体出现在工作空间的不同位置：

```python
# 数据自动记录到 self.bboxes_centers
# 调用保存函数导出 CSV
executor.save_csv_yolo("yolo_data.csv")
```

**Step B3：CSV 格式说明**

| 列名 | 类型 | 说明 |
|------|------|------|
| `object_id` | str | 跟踪 ID，如 "blue cube_0" |
| `cls` | str | 类别名 |
| `px_cam1, py_cam1` | int | 相机 1 bbox 中心像素坐标 |
| `w_cam1, h_cam1` | int | 相机 1 bbox 宽高 |
| `conf_cam1` | float | 相机 1 检测置信度 |
| `px_cam2, py_cam2` | int | 相机 2 bbox 中心 |
| `w_cam2, h_cam2` | int | 相机 2 bbox 宽高 |
| `conf_cam2` | float | 相机 2 检测置信度 |
| `ee_x, ee_y, ee_z` | float | 末端执行器世界坐标 |
| `world_x, world_y, world_z` | float | ground truth 3D 坐标（来自仿真器） |

**Step B4：数据采集策略**

| 要点 | 建议 |
|------|------|
| 采样量 | 5000–20000 帧 |
| 覆盖范围 | 物体出现在工作空间各个位置，包括边缘和角落 |
| 相机状态 | 腕部相机在不同位置（通过移动机器人实现） |
| 多物体 | 包含单物体和多物体场景 |
| 噪声 | 可添加轻微随机扰动以提升鲁棒性 |

### 阶段 C：训练 3D 回归器

使用本项目提供的 [train_regressor.py](train_regressor.py)：

```bash
# 默认 RandomForest（与原方案一致）
python train_regressor.py --data_csv yolo_data.csv --output hanoi_regressor.pkl

# 使用 XGBoost（通常精度更高）
python train_regressor.py --data_csv yolo_data.csv --output hanoi_regressor.pkl \
    --model_type xgboost --n_estimators 300

# 跳过测试集，用全部数据训练
python train_regressor.py --data_csv yolo_data.csv --output hanoi_regressor.pkl --test_size 0
```

**训练输出示例：**

```
Loaded 15234 samples, 13 features
Feature columns: ['cls', 'px_cam1', 'py_cam1', 'w_cam1', 'h_cam1', 'conf_cam1',
                  'px_cam2', 'py_cam2', 'w_cam2', 'h_cam2', 'conf_cam2',
                  'ee_x', 'ee_y', 'ee_z']
Training completed (random_forest, 12187 samples)
Test set: 3047 samples
  MAE  X: 0.0062 m  (0.62 cm)
  MAE  Y: 0.0058 m  (0.58 cm)
  MAE  Z: 0.0081 m  (0.81 cm)
  Mean Euclidean error: 0.0118 m  (1.18 cm)
Model saved to hanoi_regressor.pkl
```

**精度目标：** 平均欧氏误差 < 2 cm 即可用于大多数桌面抓取任务。如果误差 > 5 cm，需要排查：

1. 数据量是否不足 → 增加采集帧数
2. 数据覆盖是否不全 → 检查工作空间边缘是否被覆盖
3. 相机视差是否不够 → 考虑增大两个相机之间的基线距离
4. 模型是否太弱 → 尝试 `--model_type xgboost`

### 阶段 D：配置与集成

**Step D1：编写配置文件**（参考 [config/example.yaml](config/example.yaml)）

```yaml
env_name: Hanoi
yolo_model: models/yolo/hanoi_yolo.pt
regressor_model: models/regressors/hanoi_regressor.pkl
# ... 其他策略配置
```

**Step D2：端到端推理**

```python
from executor import Executor_Diffusion
import joblib

# 加载模型
regressor = joblib.load("hanoi_regressor.pkl")

# 创建执行器
executor = Executor_Diffusion(
    yolo_model="hanoi_yolo.pt",
    regressor_model=regressor,
    image_size=256,
)

# 每帧推理
for step in range(max_steps):
    image1 = env.get_camera_image("agentview")   # 256x256 RGB
    image2 = env.get_camera_image("eye_in_hand") # 256x256 RGB
    ee_pos = env.get_ee_position()               # (x, y, z)

    objects_3d = executor.yolo_estimate(
        image1, image2,
        ee_pos=ee_pos,
        conf_threshold=0.7,
    )
    # objects_3d: {"blue cube_0": (x, y, z), "red cube_1": (x, y, z), ...}
```

---

## 五、适用性分析

### 适合的场景

- **桌面/工作台操作**：物体在有限空间内（0.3–1.0m 深度范围），位置分布可控
- **固定相机配置**：相机内外参不变，一旦标定好就不再移动
- **物体类别已知**：cls_id 能提供强先验（不同物体可能有不同的典型位置分布）
- **可以搭建仿真环境**：需要仿真器提供 ground truth 3D 坐标来训练回归器
- **只需要 3D 位置**：不需要完整的 6DOF 旋转姿态
- **硬件预算有限**：两个普通 RGB 相机即可，不需要深度相机

### 不适合的场景

- **需要 6DOF 位姿**：本方案只输出 (x, y, z)，不输出旋转
- **相机配置频繁变化**：每次改变相机位置都需要重新采集数据、重新训练
- **大范围场景**（> 3m）：bbox 尺寸对深度的区分度显著下降
- **物体尺寸差异极大**：如果不提供 cls_id，"近处小物体"和"远处大物体"在 bbox 上无法区分
- **无法搭建仿真环境**：没有 ground truth 3D 坐标来源（真机也可以标定采集，但成本极高）
- **高动态场景**：物体快速运动时 YOLO 检测不稳定

### 与其他方案的量化对比

| 方案 | 精度 | 硬件要求 | 迁移成本 | 需要训练数据 |
|------|------|---------|---------|-------------|
| **本方案（双相机+回归）** | 1–2 cm | 2 RGB 相机 | 高（每场景需重新采数据） | 仿真标注数据 |
| RGB-D 相机（如 RealSense）| < 5 mm | 深度相机 | 低（即插即用） | 无 |
| 双目立体视觉 | mm–cm 级 | 2 标定相机 | 中（需精确标定） | 无 |
| 单目深度估计 | cm 级 | 1 RGB 相机 | 中 | 大规模预训练 |
| 6DOF 位姿估计 | 旋转 ±5° | 1 RGB 相机 | 高 | 3D 模型 + 真实标注 |

### Sim-to-Real 迁移注意事项

YOLO 和回归器都在仿真中训练，迁移到真实场景时：

1. **YOLO sim-to-real**：使用 domain randomization（随机纹理、光照、背景）可以显著提升泛化能力；也可在少量真实图像上 fine-tune
2. **回归器 sim-to-real**：这是更大的挑战——需要确保仿真相机内外参与真实相机一致。建议：
   - 精确建模真实相机参数到仿真中
   - 或在真实场景采集少量 ground truth（如通过运动捕捉/Aruco 标记），对仿真训练的回归器进行校准修正

---

## 六、依赖

```
ultralytics          # YOLOv8
opencv-python        # 图像处理
numpy, pandas        # 数据处理
scikit-learn         # 回归模型训练
joblib               # 模型序列化
xgboost (可选)       # 更强的回归器
```

核心代码依赖（`executor.py` 还依赖以下原项目组件）：
- robosuite（仿真环境）
- diffusion_policy（底层策略执行）
- PDDL planner（符号规划）

如果你只需要 3D 位置估计部分，可以仅使用 `yolo_estimate()` 和 `pixel_to_world_dual()` 两个函数，它们相对独立。

---

## 七、参考

- 原项目：NSM — "Price is not Right: Neuro-Symbolic Methods Outperform VLAs"
- YOLOv8: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- 特征设计参考：`executor.py:154-174` `pixel_to_world_dual()`
- 数据采集参考：`executor.py:795-816`, `executor.py:999-1006`
