import setuptools


version = "0.0.1"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="data-drift-detector",
    version=version,
    author="Yang Dai",
    author_email="yang.dai@mediamonks.com",
    description="A data drift detection and schema validation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/superyang713/data-drift-detector/blob/main/README.md",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Software Development :: Build Tools',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
    ],
    install_requires=[
        "tensorflow-data-validation",
        "google-auth",
        "google-cloud-storage",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
    ],
)
