from setuptools import setup, find_packages

setup(
    name="CustomerJourneyOptimizer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "CustomerJourneyOptimizer": ["data/*"],
    },
    # ... other setup parameters ...
)