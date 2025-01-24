from setuptools import setup, find_packages

setup(
    name='slalom-explanations',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'datasets >= 2.5.2',
        'numpy >= 1.23.1',
        'matplotlib >= 3.7',
        'transformers >= 4.30.0',
        'tqdm >= 4.65',
        'scipy >= 1.10.1',
        'scikit-learn >= 1.1.0'
    ],
    extras_require={
        'full': [
            'shap',
            'lime',
        ],
    },
    author='Tobias Leemann',
    author_email='tobias.leemann@uni-tuebingen.de',
    description='SLALOM is an method to explain transformer model for sequence classification problems.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tleemann/slalom_explanations',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8.1',
)