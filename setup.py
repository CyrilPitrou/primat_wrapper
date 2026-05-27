from setuptools import setup, find_packages

setup(
    name="primat_wrapper",
    version="0.1.0",
    description="Cobaya theory+likelihood wrapper for the PRIMAT BBN code",
    packages=find_packages(),
    # Include the YAML default files so Cobaya can read them via importlib.resources
    package_data={"primat_wrapper": ["*.yaml"]},
    install_requires=[
        "cobaya",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.10",
)
