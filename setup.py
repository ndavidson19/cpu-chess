# Updated setup.py with all C files
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np
import platform

class BuildExt(build_ext):
    def run(self):
        super().run()
        print("---" * 20)
        print(f"Building cpu_chess with {self.compiler.compiler_type}")
        print(f"Build directory: {self.build_lib}")
        print(f"Platform: {platform.machine()}")
        print("---" * 20)

    def build_extensions(self):
        # Detect architecture
        is_arm = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')
        for ext in self.extensions:
            if is_arm:
                # ARM optimization flags
                ext.extra_compile_args = [
                    '-O3',
                    '-mcpu=native',
                    '-ffast-math'
                ]
                # Define flag to use non-SIMD code paths
                ext.define_macros = [('USE_ARM_NEON', '1')]
            else:
                # x86 optimization flags
                ext.extra_compile_args = [
                    '-mavx2',
                    '-O3',
                    '-march=native',
                    '-ffast-math'
                ]
        build_ext.build_extensions(self)

chess_engine = Extension(
    'lib.cpu_chess',
    sources=[
        'core/bitboard/bitboard_ops.c',
        'core/evaluation/eval_ops.c',
        'core/evaluation/patterns.c',
        'core/search/search_ops.c',
        'core/utils/utils.c',
        'core/move/move.c',
    ],
    include_dirs=[
        'core/bitboard',
        'core/evaluation',
        'core/search',
        'core/utils',
        'core/move',
        np.get_include()
    ],
    extra_compile_args=['-mavx2', '-O3', '-march=native', '-ffast-math', '-fopenmp'],
    extra_link_args=['-lm', '-fopenmp'],
)

setup(
    name='cpu-chess',
    version='0.1',
    packages=find_packages(),  # Automatically find packages
    ext_modules=[chess_engine],
    cmdclass={'build_ext': BuildExt},
    install_requires=['numpy'],
)