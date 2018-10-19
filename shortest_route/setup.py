from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="shortest_route",
    version="0.1.0",
    author="Arjang Talattof",
    author_email="arjang@umich.edu",
    description="A small in-memory graph that can compute the shortest path between vertices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arjology/data_science/shortest_path/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["shortest_route"],
    entry_points={
        "console_scripts": ['shortest_route = shortest_route.shortest_route:main']
        },
    #package_data={'': ['shortest_path/setup.cfg']},
    include_package_data=True
)
