from setuptools import setup, find_packages
from glob import glob

package_name = "me_act_inference_py"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),  # 自动发现包
    # 不要 package_dir 配置
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}", ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Pure Python ACT ROS2 inference node for JetArm deployment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "me_act_inference_node_py = me_act_inference_py.me_act_inference_node_py:main",
        ],
    },
)