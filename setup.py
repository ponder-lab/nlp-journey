import setuptools
import smartnlp

with open("smartnlp/README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'h5py', 'requests'
]

setuptools.setup(
    name="smartnlp",
    version=smartnlp.version,
    author="msgi(慢时光)",
    author_email="mayuan120226@sina.cn",
    description="Easy-to-use and Extendable package of deep learning based nlp (Natural Language Processing) "
                "tools with tensorflow 2.x .",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msgi/nlp-journey.git",
    # download_url='https://github.com/msgi/nlp-journey/tags',
    packages=setuptools.find_packages(
        exclude=["tests"]),
    python_requires=">=3.6",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "cpu": ["tensorflow>=2.0.0"],
        "gpu": ["tensorflow-gpu>=2.0.0"],
    },
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['nlp', 'natural language processing', "nlu", "natural language understanding"
                                                           'deep learning', 'tensorflow', 'keras'],
)
