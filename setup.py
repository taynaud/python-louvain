from setuptools import setup

setup(
    name="python-louvain",
    version="0.2",
    author="Thomas Aynaud",
    author_email="thomas.aynaud@lip6.fr",
    description="Louvain algorithm for community detection",
    license="BSD",
    url="http://perso.crans.org/aynaud/communities/",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
    ],

    packages=['community'],
    install_requires=[
        "networkx",
    ],

    entry_points={
        'console_scripts': [
            'community = community:main',
        ]
    }
)
