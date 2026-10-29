from setuptools import find_packages, setup

setup(
    name="legged_gym",
    version="2.0.0",
    author="Nikita Rudin",
    license="BSD-3-Clause",
    license_files=["LICENSE", "licenses/*", "resources/robots/UNITREE_LICENSE"],
    packages=find_packages(include=["legged_gym", "legged_gym.*"]) + ["legged_gym.resources"],
    package_dir={"legged_gym.resources": "resources"},
    author_email="rudinn@ethz.ch",
    description="MuJoCo batch locomotion environments with rsl-rl PPO on MPS/CUDA/CPU",
    python_requires=">=3.10,<3.14",
    include_package_data=True,
    package_data={
        "legged_gym.resources": [
            "robots/UNITREE_LICENSE",
            "robots/provenance.json",
            "robots/go2/go2_mjlab.xml",
            "robots/go2/unitree_mjlab.json",
            "robots/go2/assets/*",
            "robots/g1/g1_mjlab.xml",
            "robots/g1/unitree_mjlab.json",
            "robots/g1/meshes/*",
        ]
    },
    install_requires=[
        "mujoco==3.17.0",
        "rsl-rl-lib==5.0.1",
        "torch>=2.6.0",
        "numpy>=1.26",
        "tensordict>=0.7.0",
        "tensorboard",
        "viser>=1.1.1,<2",
        "mjviser==0.0.12",
    ],
    entry_points={
        "console_scripts": [
            "legged-train=legged_gym.scripts.train:main",
            "legged-play=legged_gym.scripts.play:main",
        ]
    },
)
