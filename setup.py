from setuptools import setup, find_packages

setup(
    name="cryoem-particle-picker",
    version="1.0.0",
    description="Cryo-EM Single Particle Intelligent Picking and Projection Unmixing Analysis System",
    author="Cryo-EM Supercomputing Center",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "mrcfile>=1.4.0",
        "onnxruntime-gpu>=1.15.0",
    ],
    entry_points={
        "console_scripts": [
            "cryoem-train=src.training.train:main",
            "cryoem-export=src.inference.export_onnx:main",
            "cryoem-serve=src.api.main:main",
        ],
    },
)
