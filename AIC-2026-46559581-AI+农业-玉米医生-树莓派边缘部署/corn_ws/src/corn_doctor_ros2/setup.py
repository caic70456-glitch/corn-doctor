import os
from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "corn_doctor_ros2"

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [os.path.join("resource", PACKAGE_NAME)]),
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
        # launch 与模型文件随包安装到 install/share 下
        (os.path.join("share", PACKAGE_NAME, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", PACKAGE_NAME, "models"), glob(os.path.join("models", "*.onnx"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIC-2026-46559581",
    maintainer_email="13391151013@163.com",
    description="玉米医生：ROS2 Humble 相机采集 + ONNX 玉米病害识别",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = corn_doctor_ros2.camera_node:main",
            "detector_node = corn_doctor_ros2.detector_node:main",
        ],
    },
)
