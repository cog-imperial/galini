#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace galini {

void init_problem(py::module&);
void init_expression(py::module&);

PYBIND11_MODULE(galini_core, m) {
  init_problem(m);
  init_expression(m);
}

}
