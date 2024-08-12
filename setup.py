import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StreamingWavelet",
    version="1.0.6",
    author="Yu-Yang Qian",
    url='https://github.com/ZinYY/StreamingWavelet',
    author_email="qianyy@lamda.nju.edu.cn",
    install_requires=['numpy>=1.19.0'],
    license='MIT',
    description="This is an implementation for Streaming Wavelet Operator, "
                "which sequentially apply wavelet transform to a sequence efficiently.\n"
                "Reference: Qian et al., Efficient Non-stationary Online Learning by Wavelets\n"
                "with Applications to Online Distribution Shift Adaptation.\n"
                "In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    platforms='any',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
