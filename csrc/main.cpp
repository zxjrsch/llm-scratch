#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(kernels, m) {
    m.doc() = "Example pybind11 module";
    m.def("add", &add, "Add two integers");
}

// uv run c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(uv run python -m pybind11 --includes) csrc/main.cpp -o csrc/kernels$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")