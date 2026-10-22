# corn_detect 接入 RealSense 类深度相机输出三维坐标——技术方案

> 项目：AIC-2026-46559581「玉米医生」边缘端 ROS 2 包 `corn_detect`
> 目标：用 RealSense D415（或同类 RGB-D 相机）提供彩色图 + 对齐深度图，病害识别后输出病斑目标的三维坐标，经串口下发单片机。

---

## 1. 现状与结论

现有 `corn_detect` 包**已经按 RealSense 接口设计完成**，链路为：

```
RealSense D415 (驱动 realsense2_camera, 开启 align_depth)
   ├── /camera/camera/color/image_raw                   彩色图
   ├── /camera/camera/aligned_depth_to_color/image_raw  对齐深度图 (mono16, 毫米)
   └── /camera/camera/color/camera_info                 彩色镜头内参 K
              │
        detect_node（MobileNetV2 ONNX 整图分类）
              │  中心像素 (cx_img, cy_img) 深度反投影
              ▼
        /target_3d  (geometry_msgs/PointStamped, 相机坐标系, 米)
              │
        serial_bridge_node
              ▼
        串口文本 "X:%.3f,Y:%.3f,Z:%.3f\r\n" @115200 → 下位单片机
```

由于数据集是**图片级标注（无病斑框）**，三维坐标目前取**画面正中心像素**的有效深度反投影，即"把镜头对准目标叶片，中心点即为采样点"。该方案实现简单、鲁棒，符合比赛演示场景；如后续升级检测框模型，只需把采样点换成框中心（见 §6）。

## 2. 硬件与软件环境

| 项 | 配置 |
|---|---|
| 深度相机 | Intel RealSense D415（推荐，精度高）；D435/D455/Orbbec 兼容方案见 §7 |
| 边缘计算 | 树莓派 4B/5（Ubuntu 22.04 + ROS 2 Humble） |
| 驱动 | `realsense2_camera` ROS 2 包（`sudo apt install ros-humble-realsense2-camera`） |
| 工作距离 | D415 最佳 0.3–1.5 m，叶片拍摄建议 0.3–0.6 m |
| 依赖 | onnxruntime、opencv-python、cv_bridge、pyserial |

## 3. RealSense 驱动配置要点

1. **必须开启深度对齐**：`align_depth:=true`。只有对齐后的 `aligned_depth_to_color` 深度图才能与彩色像素一一对应，这是反投影正确的前提。
2. **话题前缀**：Humble 版驱动默认前缀 `/camera/camera`；旧版本为 `/camera`。launch 文件已提供 `topic_prefix` 参数，按实际驱动版本传参即可，无需改代码。
3. **建议的图像设置**（可在 launch 中追加）：
   - 彩色流 640×480@30fps（够用且省算力，树莓派上不要开 1080p）；
   - 深度流 640×480@30fps；
   - 可选开启深度降噪/时间滤波（`depth_module.temporal_filter`）减少深度空洞。
4. **内参来源**：`/camera/camera/color/camera_info` 的 K 矩阵，取 `fx=K[0], fy=K[4], cx=K[2], cy=K[5]`。出厂标定即可，无需自行标定。

## 4. 三维坐标计算方法（detect_node 核心）

对像素 `(u, v)`（当前取画面中心 `w/2, h/2`）：

**① 取深度**

```
Z = depth[v][u] × 0.001      # mono16 原始值为毫米，换算成米
```

无效值处理：若该像素深度为 0（黑洞/反光），在半径 4 像素邻域内扩展搜索最近的有效深度；全无效则本帧不发布（`detect_node._sample_depth` 已实现）。

**② 针孔模型反投影**

```
X = (u − cx) × Z / fx
Y = (v − cy) × Z / fy
Z = Z
```

坐标系为相机光学坐标系：X 向右、Y 向下、Z 向前，单位米，参考系 `camera_color_optical_frame`。

**③ 发布**

`geometry_msgs/PointStamped`，header 带时间戳与参考系，供 serial_bridge 或其他节点（机械臂、底盘）消费。

## 5. 与单片机的串口协议

- 话题：`/target_3d` → `serial_bridge_node` → 串口
- 格式：每行一条 `X:%.3f,Y:%.3f,Z:%.3f\r\n`，单位米
- 容错：回调仅入队（容量 20，满则丢最旧），串口 IO 在独立线程；串口断开自动重连，节点不崩溃
- 参数：`serial_port`（默认 `/dev/ttyUSB0`）、`serial_baud`（默认 115200）

单片机端按行读取、以逗号分割解析三个 float 即可。

## 6. 一键启动与调参

```bash
# 安装依赖后，colcon 构建
cd ~/corn_ws && colcon build --symlink-install && source install/setup.bash

# 一键启动（驱动 + 检测 + 串口）
ros2 launch corn_detect detect.launch.py

# 常用参数
ros2 launch corn_detect detect.launch.py \
  topic_prefix:=/camera/camera \
  serial_port:=/dev/ttyACM0 serial_baud:=115200 \
  infer_interval:=0.5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_path` | 包内 data/best.onnx | ONNX 分类模型 |
| `img_size` | 224 | 模型输入边长 |
| `infer_interval` | 0.5 s | 推理节流，树莓派 CPU 建议 ≥0.5 |
| `depth_unit` | 0.001 | 深度原始值→米（D415 为毫米） |

**验证方法**：将镜头对准 0.5 m 处的白墙/标定板，`ros2 topic echo /target_3d`，Z 应约等于实测距离；左右平移相机观察 X 符号与方向（X 随目标右移增大）。

## 7. 换用其他深度相机（Orbbec / 小米 / 凯擎等）

`detect_node` 只依赖三个标准话题，任何 RGB-D 相机驱动只要能给出**彩色图 + 对齐深度图 + 内参**即可直接接入：

| 相机 | 驱动包 | 需注意 |
|---|---|---|
| RealSense D415/D435/D455 | `realsense2_camera` | 开 `align_depth:=true` |
| Orbbec 奥比中光 | `ros2_orbbec`（如 Gemini 2） | 开启 depth↔color 对齐；深度单位可能为 1 mm，核对后改 `depth_unit` |
| 凯擎 K4A/Azure Kinect | `k4a_ros2` | 深度单位可能为 0.001/0.00025 m，改 `depth_unit` 即可 |

## 8. 后续升级路线（可选）

1. **病斑框定位**：数据集改为病斑框标注后，把分类模型换成 YOLO 类检测模型，采样点取**检测框中心**（u=(x1+x2)/2, v=(y1+y2)/2）替代画面中心，其余链路完全不变。
2. **多目标输出**：`/target_3d` 改为自定义 msg 数组，按置信度排序串口下发最近目标。
3. **坐标系变换**：如需机械臂基座标系坐标，在 serial_bridge 前加 TF2 变换节点（`tf2_ros`），把 `camera_color_optical_frame` 变到基座标系。

---
*方案依据：`corn_ws/src/corn_detect`（detect_node.py / serial_bridge_node.py / detect.launch.py），2026-10-21*
