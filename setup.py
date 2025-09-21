from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dopamine-research",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Computational models of dopamine function in addiction research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/dopamine-research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-prediction-error=scripts.run_prediction_error_experiment:main",
            "run-hedonic=scripts.run_hedonic_experiment:main",
            "run-incentive-salience=scripts.run_incentive_salience_experiment:main",
        ],
    },
)