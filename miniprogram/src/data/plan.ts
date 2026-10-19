// 玉米医生 RealSense 三维定位方案结构化数据
// 来源：AIC-2026-46559581 RealSense三维定位方案.md（树莓派 ROS 2 Humble）

export interface FlowStep {
  tag: string;
  title: string;
  desc: string;
}

// 系统架构数据流（第 3 节）
export const architectureFlow: FlowStep[] = [
  { tag: '硬件', title: 'RealSense D415', desc: '主动双目 + RGB，0.3–3m 量程贴合"手持近景看叶"，出厂标定内参' },
  { tag: '驱动', title: 'realsense2_camera', desc: '官方 ROS 2 驱动，align_depth=true 输出对齐深度图' },
  { tag: '识别+定位', title: 'detect_node', desc: '① MobileNetV2 整图分类 (ONNX) ② 中心点邻域深度采样 ③ 针孔模型反投影' },
  { tag: '话题', title: '/target_3d (PointStamped)', desc: '发布目标三维坐标 X/Y/Z（米，相机光学系），2 Hz' },
  { tag: '串口', title: 'serial_bridge_node', desc: 'UART 115200，ASCII 行协议下发坐标' },
  { tag: '执行', title: '下位单片机', desc: '接收 X:%.3f,Y:%.3f,Z:%.3f 换行结尾，引导云台/执行器对准病叶' }
];

// 关键技术环节（第 5 节）
export interface TechItem {
  no: string;
  title: string;
  desc: string;
}

export const keyTechs: TechItem[] = [
  {
    no: '5.1',
    title: '深度—彩色对齐',
    desc: '开启 align_depth:=true 将深度图配准到彩色光轴，保证"彩色图上看到的点 = 深度图上同位置的值"，launch 已默认开启。'
  },
  {
    no: '5.2',
    title: '目标点选取：画面中心',
    desc: '数据集为图片级标注（无包围框），取画面正中心像素 (u,v)=(W/2,H/2) 作为病叶代表点，配合"对准即测"使用。'
  },
  {
    no: '5.3',
    title: '邻域扩环深度采样',
    desc: '中心 1×1 无效则逐级扩窗（3×3→…→9×9）取最近有效深度；整窗全无效则放弃本次发布并限频告警，避免噪声坐标下发。'
  },
  {
    no: '5.4',
    title: '针孔模型反投影',
    desc: '内参 (fx,fy,cx,cy) 从 /color/camera_info 的 K 矩阵实时读取（出厂标定，免手动标定），反投影得米制三维坐标。'
  },
  {
    no: '5.5',
    title: '病害识别（与训练端一致）',
    desc: 'MobileNetV2 ONNX（约 8.7MB），输入 224×224×3 RGB；Resize→/255→ImageNet 归一化→NCHW；类别顺序从模型元数据自动读取；2Hz 节流推理。'
  }
];

// 反投影公式（第 5.4 节）
export const projectFormula: string[] = [
  'X = (u − cx) · Z / fx',
  'Y = (v − cy) · Z / fy',
  'Z = Z'
];

// 坐标系定义（第 4 节）
export interface AxisItem {
  axis: string;
  dir: string;
  color: string;
}

export const axes: AxisItem[] = [
  { axis: 'X', dir: '画面向右', color: '#d64545' },
  { axis: 'Y', dir: '画面向下', color: '#2f9e44' },
  { axis: 'Z', dir: '沿光轴向前（目标距离）', color: '#3b7bd6' }
];

// ROS 话题接口（第 6.1 节）
export interface TopicItem {
  topic: string;
  dir: string;
  type: string;
  note: string;
}

export const rosTopics: TopicItem[] = [
  { topic: '/color/image_raw', dir: '订阅', type: 'sensor_msgs/Image', note: '彩色图（经 topic_prefix 重映射）' },
  { topic: '/aligned_depth_to_color/image_raw', dir: '订阅', type: 'sensor_msgs/Image', note: '对齐深度图 mono16' },
  { topic: '/color/camera_info', dir: '订阅', type: 'sensor_msgs/CameraInfo', note: '彩色相机内参' },
  { topic: '/target_3d', dir: '发布', type: 'geometry_msgs/PointStamped', note: '目标三维坐标（米），2 Hz' }
];

// 参数配置表（第 7 节）
export interface ParamItem {
  name: string;
  value: string;
  node: string;
  note: string;
}

export const paramTable: ParamItem[] = [
  { name: 'model_path', value: 'data/best.onnx', node: 'detect_node', note: 'ONNX 模型路径（约 8.7MB）' },
  { name: 'img_size', value: '224', node: 'detect_node', note: '模型输入边长' },
  { name: 'infer_interval', value: '0.5 s', node: 'detect_node', note: '推理节流间隔（2Hz）' },
  { name: 'depth_unit', value: '0.001', node: 'detect_node', note: '深度原始值 → 米换算（毫米制）' },
  { name: 'port', value: '/dev/ttyUSB0', node: 'serial_bridge_node', note: '串口设备' },
  { name: 'baudrate', value: '115200', node: 'serial_bridge_node', note: '串口波特率' },
  { name: 'topic_prefix', value: '/camera/camera', node: 'launch', note: '驱动话题前缀（老版本驱动为 /camera）' }
];

// 部署与运行步骤（第 8 节）
export interface DeployStep {
  comment: string;
  cmd: string[];
}

export const deploySteps: DeployStep[] = [
  {
    comment: '1) 安装依赖（Ubuntu 22.04 + ROS 2 Humble）',
    cmd: [
      'sudo apt install ros-humble-realsense2-camera \\',
      '  ros-humble-cv-bridge python3-pip ros-humble-rviz2',
      'pip3 install onnxruntime pyserial numpy \\',
      '  opencv-python-headless'
    ]
  },
  {
    comment: '2) 编译工作空间',
    cmd: ['cd ~/corn_ws', 'colcon build --symlink-install', 'source install/setup.bash']
  },
  {
    comment: '3) 串口权限（首次）',
    cmd: ['sudo usermod -aG dialout $USER && 重新登录']
  },
  {
    comment: '4) 一键启动（驱动 + 识别 + 串口桥接）',
    cmd: [
      'ros2 launch corn_detect detect.launch.py',
      '# 指定串口: ... serial_port:=/dev/ttyACM0 \\',
      '  serial_baud:=9600'
    ]
  },
  {
    comment: '5) 验证',
    cmd: [
      'ros2 topic echo /target_3d   # 持续输出 X/Y/Z',
      'rviz2                         # PointStamped 确认坐标'
    ]
  }
];

// 误差来源与对策（第 9 节）
export interface ErrorItem {
  source: string;
  impact: string;
  fix: string;
}

export const errorSources: ErrorItem[] = [
  { source: '手持抖动、瞄准偏移', impact: '坐标漂移', fix: '对准后稳 1s 读数；上云台；叠加十字准星' },
  { source: '叶面强反光/半透明', impact: '深度无效/跳变', fix: '邻域采样兜底；避免逆光；侧 45° 照射' },
  { source: '深度噪点', impact: '单帧跳变', fix: '下位机对连续 3 帧做中值/均值滤波' },
  { source: '坐标系不匹配', impact: '下位机解算错位', fix: '统一相机光学系 + 明确轴向；必要时加外参' }
];

// 风险与后续扩展（第 10 节）
export interface RiskItem {
  no: string;
  title: string;
  desc: string;
}

export const risks: RiskItem[] = [
  { no: '01', title: '病斑级定位升级', desc: '数据集补充包围框后，将"画面中心点"替换为"检测框中心点"反投影即可，/target_3d 接口与串口协议完全不变，下位机零改动。' },
  { no: '02', title: '多目标', desc: '可扩展为对每个病斑发布一帧 PointStamped（加编号字段），按距离最近的优先下发。' },
  { no: '03', title: '硬件备选', desc: 'D415 缺货可用任意"彩色+对齐深度+内参"深度相机（奥比中光等）；极端情况可用 USB 相机 + 固定工作距离（Z=常数）降级。' },
  { no: '04', title: '夜间/大棚弱光', desc: 'D415 为主动双目，弱光下深度可用；彩色通道补光即可保证分类精度。' }
];

// 系统组成（关于页）
export interface ModuleItem {
  name: string;
  desc: string;
}

export const systemModules: ModuleItem[] = [
  { name: '玉米病害数据集', desc: '8379 张田间照片，train/val/test 三分' },
  { name: 'PC 端训练与识别', desc: '自建 CNN 对比实验 → MobileNetV2 迁移学习 → Flask 网页 → ONNX 导出' },
  { name: '树莓派边缘部署', desc: 'corn_detect 深度定位 + corn_doctor_ros2 USB 相机手持机' }
];

// 相关链接
export const LINK_REPO = 'https://github.com/caic70456-glitch/corn-doctor';
export const LINK_PAGE = 'https://caic70456-glitch.github.io/corn-doctor/';
