import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraspSamplerLearning",
    version="0.0.1",
    author="Donghyun Sung",
    author_email="dh-sung@naver.com, sdh1259@snu.ac.kr",
    description="learning based grasp sampler comparative study",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["gsl"],
    python_requires=">=3.6",
)
