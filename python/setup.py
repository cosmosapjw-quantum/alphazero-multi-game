import os
import re
import sys
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# Version information
VERSION = '1.0.0'

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the C++ extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Set build type based on environment variable or default to Release
        build_type = os.environ.get('BUILD_TYPE', 'Release')
        
        # Set CMake arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={build_type}',
            '-DALPHAZERO_BUILD_TESTS=OFF',
            '-DALPHAZERO_ENABLE_PYTHON=ON',
            '-DALPHAZERO_BUILD_EXAMPLES=OFF',
        ]

        # Set library output directory for Windows
        if platform.system() == "Windows":
            cmake_args += [
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{build_type.upper()}={extdir}'
            ]

        # Set other platform-specific settings
        if platform.system() == "Darwin":
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15']
        
        # Set CUDA options if available
        if os.environ.get('ALPHAZERO_ENABLE_GPU', '1') == '1':
            cmake_args += ['-DALPHAZERO_ENABLE_GPU=ON']
        else:
            cmake_args += ['-DALPHAZERO_ENABLE_GPU=OFF']

        # Configure build arguments
        build_args = ['--config', build_type]
        
        # Control parallelism
        if platform.system() == "Windows":
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j', str(multiprocessing.cpu_count())]

        # Create build directory if it doesn't exist
        os.makedirs(self.build_temp, exist_ok=True)
        
        # Run CMake configure and build steps
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# Get requirements from requirements.txt
def get_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if not line.startswith('#')]


setup(
    name='alphazero',
    version=VERSION,
    author='AlphaZero Team',
    author_email='alphazero@example.com',
    description='AlphaZero Multi-Game AI Engine',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alphazero/multi-game',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='alphazero, ai, reinforcement learning, mcts, chess, go, gomoku',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=get_requirements(),
    ext_modules=[CMakeExtension('pyalphazero')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
)