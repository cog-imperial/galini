/* Copyright 2018 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================== */
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;


// Ipopt SmartPtr does not have a .get() method so we need to implement
// the adapter
PYBIND11_DECLARE_HOLDER_TYPE(T, Ipopt::SmartPtr<T>);

namespace pybind11 {

namespace detail {

template<typename T>
struct holder_helper<Ipopt::SmartPtr<T>> {
static const T *get(const Ipopt::SmartPtr<T> &ptr) {
  return Ipopt::GetRawPtr(ptr);
}
};

} // namespace detail

} // namespace pybind11

namespace galini {

namespace ipopt {


class PythonJournal : public Ipopt::Journal {
public:
  PythonJournal(const std::string &name, Ipopt::EJournalLevel default_level, py::object stream)
    : Journal(name, default_level)
    , stream_(stream)
  {}
protected:
  virtual void PrintImpl(Ipopt::EJournalCategory category,
			 Ipopt::EJournalLevel level,
			 const char* str) override {
    auto write = stream_.attr("write");
    write(str);
  }

  virtual void PrintfImpl(Ipopt::EJournalCategory category,
			  Ipopt::EJournalLevel level,
			  const char* pformat,
			  va_list ap) override {
    // Define string
    static const int max_len = 8192;
    char s[max_len];

    if (vsnprintf(s, max_len, pformat, ap) > max_len) {
      PrintImpl(category, level, "Warning: not all characters of next line are printed to the file.\n");
    }
    PrintImpl(category, level, s);
  }

  virtual void FlushBufferImpl() override {
    stream_.attr("flush")();
}
private:
  py::object stream_;
};

void init_ipopt_bindings(py::module& m) {
  py::class_<Ipopt::Journal, Ipopt::SmartPtr<Ipopt::Journal>>(m, "Journal")
    .def_property_readonly("name", &Ipopt::Journal::Name);

  py::class_<PythonJournal, Ipopt::Journal, Ipopt::SmartPtr<PythonJournal>>(m, "PythonJournal")
    .def(py::init<const std::string&, Ipopt::EJournalLevel, py::object>());

  py::class_<Ipopt::Journalist, Ipopt::SmartPtr<Ipopt::Journalist>>(m, "Journalist")
    .def("add_journal", &Ipopt::Journalist::AddJournal)
    .def("delete_all_journals", &Ipopt::Journalist::DeleteAllJournals);

  py::enum_<Ipopt::EJournalLevel>(m, "EJournalLevel")
    .value("J_INSUPPRESIBLE", Ipopt::EJournalLevel::J_INSUPPRESSIBLE)
    .value("J_NONE", Ipopt::EJournalLevel::J_NONE)
    .value("J_ERROR", Ipopt::EJournalLevel::J_ERROR)
    .value("J_STRONGWARNING", Ipopt::EJournalLevel::J_STRONGWARNING)
    .value("J_SUMMARY", Ipopt::EJournalLevel::J_SUMMARY)
    .value("J_WARNING", Ipopt::EJournalLevel::J_WARNING)
    .value("J_ITERSUMMARY", Ipopt::EJournalLevel::J_ITERSUMMARY)
    .value("J_DETAILED", Ipopt::EJournalLevel::J_DETAILED)
    .value("J_MOREDETAILED", Ipopt::EJournalLevel::J_MOREDETAILED)
    .value("J_VECTOR", Ipopt::EJournalLevel::J_VECTOR)
    .value("J_MOREVECTOR", Ipopt::EJournalLevel::J_MOREVECTOR)
    .value("J_MATRIX", Ipopt::EJournalLevel::J_MATRIX)
    .value("J_MOREMATRIX", Ipopt::EJournalLevel::J_MOREMATRIX)
    .value("J_ALL", Ipopt::EJournalLevel::J_ALL)
    .value("J_LAST_LEVEL", Ipopt::EJournalLevel::J_LAST_LEVEL)
    .export_values();

  py::class_<Ipopt::OptionsList, Ipopt::SmartPtr<Ipopt::OptionsList>>(m, "OptionsList")
    .def("clear", &Ipopt::OptionsList::clear)
    .def("set_string_value", &Ipopt::OptionsList::SetStringValue)
    .def("set_numeric_value", &Ipopt::OptionsList::SetNumericValue)
    .def("set_integer_value", &Ipopt::OptionsList::SetIntegerValue)
    .def("set_string_value_if_unset", &Ipopt::OptionsList::SetStringValueIfUnset)
    .def("set_numeric_value_if_unset", &Ipopt::OptionsList::SetNumericValueIfUnset)
    .def("set_integer_value_if_unset", &Ipopt::OptionsList::SetIntegerValueIfUnset);

  py::class_<Ipopt::IpoptApplication, Ipopt::SmartPtr<Ipopt::IpoptApplication>>(m, "IpoptApplication")
    .def(py::init())
    .def("journalist", &Ipopt::IpoptApplication::Jnlst)
    .def("options", (Ipopt::SmartPtr<Ipopt::OptionsList> (Ipopt::IpoptApplication::*)(void)) &Ipopt::IpoptApplication::Options);
}

} // namespace ipopt

} // namespace galini
