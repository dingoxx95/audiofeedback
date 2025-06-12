from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audiofeedback",
    version="1.0.0",
    author="Audio Feedback Team",
    author_email="info@audiofeedback.dev",
    description="Professional audio analysis and feedback system using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audiofeedback",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "advanced": [
            "essentia>=2.1b6.dev1090",
            "mutagen>=1.45.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audiofeedback=audiofeedback.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "audiofeedback": ["*.md", "*.txt"],
    },
)
