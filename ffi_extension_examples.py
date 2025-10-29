from csrc.kernels import add
from tokenizer import sum_as_string

if __name__ == '__main__':
    # Rust 
    x = sum_as_string(1, 2)
    print(type(x))

    # C++
    # Build example (TODO migrate to setuptools)
    # TODO https://docs.pytorch.org/docs/stable/cpp_extension.html 
    # uv run c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(uv run python -m pybind11 --includes) csrc/main.cpp -o csrc/kernels$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    print(add(1,2))