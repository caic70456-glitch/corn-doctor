import os
from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "corn_detect"

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [os.path.join("resource", PACKAGE_NAME)]),
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
        # launch 与模型文件随包安装到 install/share（--symlink-install 下为软链接）
        (os.path.join("share", PACKAGE_NAME, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", PACKAGE_NAME, "data"), glob(os.path.join("data", "*"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIC-2026-46559581",
    maintainer_email="13391151013@163.com",
    description="RealSense D415 玉米病害检测 + 病斑三维坐标 + 串口输出",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detect_node = corn_detect.detect_node:main",
            "serial_bridge_node = corn_detect.serial_bridge_node:main",
        ],
    },
)
