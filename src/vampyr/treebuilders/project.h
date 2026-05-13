#pragma once

#include <complex>

#include <MRCPP/utils/CompFunction.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include "PyProjectors.h"

namespace vampyr {

using ComplexDouble = std::complex<double>;
template <int D> void project(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    m.def(
        "ZeroTree",
        [](const MultiResolutionAnalysis<D> &mra, const std::string &name) {
            auto out = std::make_unique<FunctionTree<D, double>>(mra, name);
            out->setZero();
            return out;
        },
        "mra"_a,
        "name"_a = "nn");

    // Complex ZeroTree
    m.def(
        "ZeroTree_Complex",
        [](const MultiResolutionAnalysis<D> &mra, const std::string &name) {
            auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(mra, name);
            out->setZero();
            return out;
        },
        "mra"_a,
        "name"_a = "nn");

    py::class_<PyScalingProjector<D>>(m, (std::string("ScalingProjector") + std::to_string(D) + "D").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, double>(), "mra"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<D> &, int>(), "mra"_a, "scale"_a)
        .def(
            "__call__", [](PyScalingProjector<D> &P, RepresentableFunction<D, double> &func) { return P(func); }, "func"_a)
        .def(
            "__call__",
            [](PyScalingProjector<D> &P, std::function<double(const Coord<D> &r)> func) {
                try {
                    // When the analytic function func is badly defined, it kills the kernel
                    // of Notebooks. This evaluates func in a point, and if it is not successful
                    // it throws an error instead of killing the kernel.
                    auto arr = std::array<double, D>();
                    arr.fill(111111.111); // A number which hopefully does not divide by zero
                    func(arr);
                } catch (py::cast_error &e) {
                    py::print("Error: Invalid definition of analytic function");
                    throw;
                }
                auto old_threads = mrcpp_get_num_threads();
                set_max_threads(1);
                auto out = P(func);
                set_max_threads(old_threads);
                return out;
            },
            "func"_a);

    py::class_<PyWaveletProjector<D>>(m, (std::string("WaveletProjector") + std::to_string(D) + "D").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, int>(), "mra"_a, "scale"_a)
        .def(
            "__call__", [](PyWaveletProjector<D> &P, RepresentableFunction<D, double> &func) { return P(func); }, "func"_a)
        .def(
            "__call__",
            [](PyWaveletProjector<D> &P, std::function<double(const Coord<D> &r)> func) {
                try {
                    auto arr = std::array<double, D>();
                    arr.fill(111111.111); // A number which hopefully does not divide by zero
                    func(arr);
                } catch (py::cast_error &e) {
                    py::print("Error: Invalid definition of analytic function");
                    throw;
                }

                auto old_threads = mrcpp_get_num_threads();
                set_max_threads(1);
                auto out = P(func);
                set_max_threads(old_threads);
                return out;
            },
            "func"_a);

    // Complex ScalingProjector
    py::class_<PyScalingProjectorComplex<D>>(m, (std::string("ScalingProjector") + std::to_string(D) + "D_Complex").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, double>(), "mra"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<D> &, int>(), "mra"_a, "scale"_a)
        .def(
            "__call__",
            [](PyScalingProjectorComplex<D> &P, std::function<ComplexDouble(const Coord<D> &r)> func) {
                try {
                    auto arr = std::array<double, D>();
                    arr.fill(111111.111);
                    func(arr);
                } catch (py::cast_error &e) {
                    py::print("Error: Invalid definition of analytic function");
                    throw;
                }
                auto old_threads = mrcpp_get_num_threads();
                set_max_threads(1);
                auto out = P(func);
                set_max_threads(old_threads);
                return out;
            },
            "func"_a);

    // Complex WaveletProjector
    py::class_<PyWaveletProjectorComplex<D>>(m, (std::string("WaveletProjector") + std::to_string(D) + "D_Complex").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, int>(), "mra"_a, "scale"_a)
        .def(
            "__call__",
            [](PyWaveletProjectorComplex<D> &P, std::function<ComplexDouble(const Coord<D> &r)> func) {
                try {
                    auto arr = std::array<double, D>();
                    arr.fill(111111.111);
                    func(arr);
                } catch (py::cast_error &e) {
                    py::print("Error: Invalid definition of analytic function");
                    throw;
                }
                auto old_threads = mrcpp_get_num_threads();
                set_max_threads(1);
                auto out = P(func);
                set_max_threads(old_threads);
                return out;
            },
            "func"_a);
}

template <int D> void advanced_project(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    m.def("project",
          [](double prec, FunctionTree<D, double> &out, RepresentableFunction<D, double> &inp, int max_iter, bool abs_prec) {
              mrcpp::project<D, double>(prec, out, inp, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);

    m.def(
        "project",
        [](double prec,
           FunctionTree<D, double> &out,
           std::function<double(const Coord<D> &r)> inp,
           int max_iter,
           bool abs_prec) {
            auto old_threads = mrcpp_get_num_threads();
            mrcpp::set_max_threads(1);
            mrcpp::project<D>(prec, out, inp, max_iter, abs_prec);
            mrcpp::set_max_threads(old_threads);
        },
        "prec"_a = -1.0,
        "out"_a,
        "inp"_a,
        "max_iter"_a = -1,
        "abs_prec"_a = false);
    // Project lambda onto CompFunction<3>: 3D-only because the MRCPP
    //  declarations in CompFunction.h are non-templated and 3D-only.
    //     We bind project_real and project_cplx as separate names because
    //     overloading "project" causes pybind11 to silently coerce complex
    //     return values to double through std::function<double(...)>.
    if constexpr (D == 3) {
        m.def("project_real",
              [](CompFunction<3> &out,
                 std::function<double(const Coord<3> &)> f,
                 double prec) {
                  mrcpp::project_real(out, f, prec);
              },
              "out"_a, "inp"_a, "prec"_a,
              "Project a real-valued analytic function onto a CompFunction<3>.");

        m.def("project_cplx",
              [](CompFunction<3> &out,
                 std::function<ComplexDouble(const Coord<3> &)> f,
                 double prec) {
                  mrcpp::project_cplx(out, f, prec);
              },
              "out"_a, "inp"_a, "prec"_a,
              "Project a complex-valued analytic function onto a CompFunction<3>.");
    }
}
} // namespace vampyr
