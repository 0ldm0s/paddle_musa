from paddle.utils.cpp_extension import CppExtension, MUSAExtension, setup

MUSA_LIB = "/usr/local/musa/lib"

setup(
    name='custom_setup_ops',
    ext_modules=MUSAExtension(
        sources=['kernels/flash_attn/flash_attn_kvcache_mate.cu'],
        library_dirs=[MUSA_LIB],
        libraries=[
            "mublas",
            "mublasLt",
            "mudnn_xmma",
        ],
        extra_link_args=[
            f"-Wl,-rpath,{MUSA_LIB}",
            "-Wl,--no-as-needed",
        ],
    )
)
