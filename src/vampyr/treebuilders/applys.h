#pragma once

#include <complex>

#include <MRCPP/utils/CompFunction.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MRCPP/treebuilders/apply.h>

namespace vampyr {

using ComplexDouble = std::complex<double>;

template <int D> void applys(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    // Real divergence
    m.def(
        "divergence",
        [](DerivativeOperator<D> &oper, std::vector<FunctionTree<D, double> *> &inp) {
            std::unique_ptr<FunctionTree<D, double>> out{nullptr};
            if (inp.size() == (size_t)D) {
                out = std::make_unique<FunctionTree<D, double>>(inp[0]->getMRA());
                divergence<D, double>(*out, oper, inp);
            }
            return out;
        },
        "oper"_a,
        "inp"_a);

    // Complex divergence
    m.def(
        "divergence",
        [](DerivativeOperator<D> &oper, std::vector<FunctionTree<D, ComplexDouble> *> &inp) {
            std::unique_ptr<FunctionTree<D, ComplexDouble>> out{nullptr};
            if (inp.size() == (size_t)D) {
                out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp[0]->getMRA());
                divergence<D, ComplexDouble>(*out, oper, inp);
            }
            return out;
        },
        "oper"_a,
        "inp"_a);

    // Real gradient
    m.def(
        "gradient",
        [](DerivativeOperator<D> &oper, FunctionTree<D, double> &inp) {
            auto tmp = mrcpp::gradient<D, double>(oper, inp);
            std::vector<std::unique_ptr<FunctionTree<D, double>>> out;
            for (size_t i = 0; i < tmp.size(); i++) {
                auto *tmp_p = std::get<1>(tmp[i]);
                out.push_back(std::unique_ptr<FunctionTree<D, double>>(tmp_p));
            }
            mrcpp::clear(tmp, false);
            return out;
        },
        "oper"_a,
        "inp"_a);

    // Complex gradient - implemented using apply() since MRCPP doesn't instantiate gradient<D, ComplexDouble>
    m.def(
        "gradient",
        [](DerivativeOperator<D> &oper, FunctionTree<D, ComplexDouble> &inp) {
            std::vector<std::unique_ptr<FunctionTree<D, ComplexDouble>>> out;
            for (int dir = 0; dir < D; dir++) {
                auto component = std::make_unique<FunctionTree<D, ComplexDouble>>(inp.getMRA());
                mrcpp::apply<D, ComplexDouble>(*component, oper, inp, dir);
                out.push_back(std::move(component));
            }
            return out;
        },
        "oper"_a,
        "inp"_a);
}

// Direct bindings to MRCPP functionality
template <int D> void advanced_applys(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    // Real apply with ConvolutionOperator
    m.def(
        "apply",
        [](double prec, FunctionTree<D, double> &out, ConvolutionOperator<D> &oper, FunctionTree<D, double> &inp, int max_iter, bool abs_prec) {
            mrcpp::apply<D, double>(prec, out, oper, inp, max_iter, abs_prec);
        },
        "prec"_a,
        "out"_a,
        "oper"_a,
        "inp"_a,
        "max_iter"_a = -1,
        "abs_prec"_a = false);

    // Complex apply with ConvolutionOperator
    m.def(
        "apply",
        [](double prec, FunctionTree<D, ComplexDouble> &out, ConvolutionOperator<D> &oper, FunctionTree<D, ComplexDouble> &inp, int max_iter, bool abs_prec) {
            mrcpp::apply<D, ComplexDouble>(prec, out, oper, inp, max_iter, abs_prec);
        },
        "prec"_a,
        "out"_a,
        "oper"_a,
        "inp"_a,
        "max_iter"_a = -1,
        "abs_prec"_a = false);

    // Real apply with DerivativeOperator
    m.def("apply",
          [](FunctionTree<D, double> &out, DerivativeOperator<D> &oper, FunctionTree<D, double> &inp, int dir) {
              mrcpp::apply<D, double>(out, oper, inp, dir);
          },
          "out"_a,
          "oper"_a,
          "inp"_a,
          "dir"_a = -1);

    // Complex apply with DerivativeOperator
    m.def("apply",
          [](FunctionTree<D, ComplexDouble> &out, DerivativeOperator<D> &oper, FunctionTree<D, ComplexDouble> &inp, int dir) {
              mrcpp::apply<D, ComplexDouble>(out, oper, inp, dir);
          },
          "out"_a,
          "oper"_a,
          "inp"_a,
          "dir"_a = -1);

        //  Apply ConvolutionOperator to a CompFunction<D>
    //     C++ signature has a metric parameter for multi-component math;
    //     we hide it by always using mrcpp::defaultMetric.
    m.def("apply",
          [](double prec,
             CompFunction<D> &out,
             ConvolutionOperator<D> &oper,
             const CompFunction<D> &inp,
             int max_iter,
             bool abs_prec) {
              mrcpp::apply<D>(prec, out, oper, inp,
                              mrcpp::defaultMetric, max_iter, abs_prec);
          },
          "prec"_a,
          "out"_a,
          "oper"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);
    // apply DerivativeOperator to a CompFunction<D>.
    //     Mirrors the existing FunctionTree DerivativeOperator apply at line ~123.
    if constexpr (D == 3) {
        m.def("apply",
              [](CompFunction<D> &out,
                 DerivativeOperator<D> &oper,
                 CompFunction<D> &inp,
                 int dir) {
                  mrcpp::apply<D>(out, oper, inp, dir, mrcpp::defaultMetric);
              },
              "out"_a, "oper"_a, "inp"_a, "dir"_a = -1,
              "Apply a DerivativeOperator natively to a CompFunction. (3D only)");
    }
}

} // namespace vampyr
