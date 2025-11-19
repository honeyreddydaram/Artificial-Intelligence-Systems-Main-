from setuptools import setup, find_packages

setup(
    name="ecg-analysis",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "wfdb",
        "PyWavelets",
        "tsfel",
        "scikit-learn",
        "plotly",
        "altair",
        "joblib",
        "requests",
        "python-dotenv",
        "azure-storage-blob",
        "azure-identity",
    ],
    description="A tool for ECG signal analysis and visualization",
    keywords="ecg, analysis, signal processing, medicine, research",
    python_requires=">=3.8",
)