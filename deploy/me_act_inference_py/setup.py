from setuptools import setup


package_name = "me_act_inference_py"


setup(
    name=package_name,
    version="0.0.1",
    py_modules=[
        "act_pipeline_py",
        "me_act_inference_node_py",
    ],
    package_dir={"": "scripts"},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/me_act_baseline_py.launch.py",
            "launch/me_act_memory_py.launch.py",
        ]),
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
            "me_act_inference_node_py = me_act_inference_node_py:main",
        ],
    },
)
