from setuptools import find_packages, setup

setup(
    name="DeepBioisostere",
    version="0.1.0",
    packages=find_packages(),  # Tell setuptools to find packages under 'scripts'
    install_requires=[
        # List your project's dependencies here.
        # E.g., 'numpy', 'pandas', etc.
    ],
    python_requires=">=3.7",  # Minimum version requirement of the package
    author="Hyeongwoo Kim†, Seokhyun Moon†, Wonho Zhung, Sinwoo Kim, Jaechang Lim, and Woo Youn Kim*",
    author_email="novainco98@kaist.ac.kr;mshmjp@kaist.ac.kr",
    description="DeepBioisostere: Deep Learning-based Bioisosteric Replacements for Optimization of Multiple Molecular Properties",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # if your README is markdown
    url="https://github.com/Hwoo-Kim/DeepBioisostere.git",
)
