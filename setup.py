from setuptools import setup

setup(
    name='genai4seqcls',
    version='1.0.4',    
    description='The library is built as a subclass of Huggingface’s Trainer, tailored for encoder-style fine-tuning of generative transformer models on sequence classification tasks. It enhances the standard training loop with features like a dedicated classification head, RAG (Retrieval-Augmented Generation) integrated predictions, and label-balanced batch sampling. The framework also includes advanced callbacks for improved experiment monitoring and real-time notifications.',
    url='https://github.com/mbnczy/GenAI4SeqCls',
    author='Martin Balázs Bánóczy',
    author_email='banoczy.martin@icloud.com',
    license='MIT',
    packages=['genai4seqcls'],
    install_requires=[
        'setuptools',
        'wheel',
        'numpy',
        'pandas',
        'tqdm',
        'regex',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'datasets',
        'huggingface_hub',
        'transformers',
        'trl',
        'unsloth',
        'wandb',
        'tabulate',
        'openpyxl',
        'tf_keras',
        'faiss-cpu',
        'sentence-transformers',
        'slack_sdk',
        'gpustat',
        'nvidia-htop',
        'nvitop',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.11'
    ],
)
