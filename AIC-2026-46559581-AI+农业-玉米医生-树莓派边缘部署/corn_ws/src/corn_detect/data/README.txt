把病害识别模型 best.onnx 放到本目录（MobileNetV2 四分类 ONNX，输入 224x224，
类别 Blight / Common_Rust / Gray_Leaf_Spot / Healthy，输出已含 softmax）。
detect_node 运行时通过 get_package_share_directory("corn_detect")/data/best.onnx 自动加载。
模型来源：PC端训练与识别代码/06_ONNX模型导出.py 产物（本仓库已附带 best.onnx）。
