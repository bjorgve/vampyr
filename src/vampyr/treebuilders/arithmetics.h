#pragma once

#include <complex>

#include <MRCPP/utils/CompFunction.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>   
#include <MRCPP/treebuilders/add.h>
#include <MRCPP/treebuilders/multiply.h>

namespace vampyr {

using ComplexDouble = std::complex<double>;
template <int D> void arithmetics(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    m.def(
        "sum",
        [](std::vector<FunctionTree<D, double> *> &inp) {
            auto out = std::unique_ptr<FunctionTree<D, double>>(nullptr);
            if (inp.size() > 0) {
                auto &mra = inp[0]->getMRA();
                out = std::make_unique<FunctionTree<D, double>>(mra);
                FunctionTreeVector<D, double> vec;
                for (auto* tree : inp) vec.push_back({1.0, tree});
                build_grid(*out, vec);
                add(-1.0, *out, vec);
            }
            return out;
        },
        "inp"_a);

    m.def(
        "sum",
        [](std::vector<std::tuple<double, FunctionTree<D, double> *>> &inp) {
            auto out = std::unique_ptr<FunctionTree<D, double>>(nullptr);
            if (inp.size() > 0) {
                auto &mra = std::get<1>(inp[0])->getMRA();
                out = std::make_unique<FunctionTree<D, double>>(mra);
                FunctionTreeVector<D, double> vec;
                for (auto& t : inp) vec.push_back({std::get<0>(t), std::get<1>(t)});
                build_grid(*out, vec);
                add(-1.0, *out, vec);
            }
            return out;
        },
        "inp"_a);



    m.def("dot", [](FunctionTree<D, double> &bra, FunctionTree<D, double> &ket) { return mrcpp::dot<D, double>(bra, ket); }, "bra"_a, "ket"_a);

    // Complex dot products
    m.def("dot", [](FunctionTree<D, ComplexDouble> &bra, FunctionTree<D, ComplexDouble> &ket) {
        return mrcpp::dot<D, ComplexDouble>(bra, ket);
    }, "bra"_a, "ket"_a);

    // Mixed-type dot products (real-complex and complex-real)
    m.def("dot", [](FunctionTree<D, double> &bra, FunctionTree<D, ComplexDouble> &ket) {
        return mrcpp::dot<D, double, ComplexDouble>(bra, ket);
    }, "bra"_a, "ket"_a);

    m.def("dot", [](FunctionTree<D, ComplexDouble> &bra, FunctionTree<D, double> &ket) {
        return mrcpp::dot<D, ComplexDouble, double>(bra, ket);
    }, "bra"_a, "ket"_a);

    // Conjugate function for complex trees
    m.def("conj",
          [](FunctionTree<D, ComplexDouble> *inp) {
              // First make a deep copy, then conjugate in place
              auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp->getMRA());
              copy_grid(*out, *inp);
              copy_func(*out, *inp);
              // Now conjugate the coefficients in place
              for (int i = 0; i < out->getNEndNodes(); i++) {
                  auto &out_node = out->getEndMWNode(i);
                  int n_coefs = out_node.getNCoefs();
                  for (int j = 0; j < n_coefs; j++) {
                      out_node.getCoefs()[j] = std::conj(out_node.getCoefs()[j]);
                  }
              }
              out->mwTransform(BottomUp);
              out->calcSquareNorm();
              return out;
          },
          "inp"_a);

    m.def(
        "dot",
        [](std::vector<FunctionTree<D, double> *> &inp_a, std::vector<FunctionTree<D, double> *> &inp_b) {
            auto out = std::unique_ptr<FunctionTree<D, double>>(nullptr);
            if ((inp_a.size() > 0) && (inp_b.size() == inp_a.size())) {
                auto &mra = inp_a[0]->getMRA();
                out = std::make_unique<FunctionTree<D, double>>(mra);
                auto out_vec = FunctionTreeVector<D, double>();
                for (size_t i = 0; i < inp_a.size(); ++i) {
                    auto *prod = new FunctionTree<D, double>(mra);
                    build_grid(*prod, *inp_a[i]);
                    build_grid(*prod, *inp_b[i]);
                    build_grid(*prod, 1);
                    multiply(-1.0, *prod, 1.0, *inp_a[i], *inp_b[i]);
                    out_vec.push_back({1.0, prod});
                }
                build_grid(*out, out_vec); // Union grid
                add(-1.0, *out, out_vec);
                clear(out_vec, true);
            }
            return out;
        },
        "inp_a"_a,
        "inp_b"_a);

    m.def(
        "prod",
        [](std::vector<FunctionTree<D, double> *> &inp) {
            auto out = std::unique_ptr<FunctionTree<D, double>>(nullptr);
            if (inp.size() > 0) {
                auto &mra = inp[0]->getMRA();
                out = std::make_unique<FunctionTree<D, double>>(mra);
                FunctionTreeVector<D, double> vec;
                for (auto* tree : inp) vec.push_back({1.0, tree});
                build_grid(*out, vec); // Union grid
                build_grid(*out, 1);   // One extra refinement
                multiply(-1.0, *out, vec);
            }
            return out;
        },
        "inp"_a);

    m.def(
        "prod",
        [](std::vector<std::tuple<double, FunctionTree<D, double> *>> &inp) {
            auto out = std::unique_ptr<FunctionTree<D, double>>(nullptr);
            if (inp.size() > 0) {
                auto &mra = std::get<1>(inp[0])->getMRA();
                out = std::make_unique<FunctionTree<D, double>>(mra);
                FunctionTreeVector<D, double> vec;
                for (auto& t : inp) vec.push_back({std::get<0>(t), std::get<1>(t)});
                build_grid(*out, vec); // Union grid
                build_grid(*out, 1);   // One extra refinement
                multiply(-1.0, *out, vec);
            }
            return out;
        },
        "inp"_a);
}

template <int D> void advanced_arithmetics(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    m.def("add",
          [](double prec, FunctionTree<D, double> &out, double a, FunctionTree<D, double> &inp_a, double b, FunctionTree<D, double> &inp_b, int max_iter, bool abs_prec) {
              mrcpp::add<D, double>(prec, out, a, inp_a, b, inp_b, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "a"_a = 1.0,
          "inp_a"_a,
          "b"_a = 1.0,
          "inp_b"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);

    m.def("add",
          [](double prec, FunctionTree<D, double> &out, std::vector<FunctionTree<D, double> *> &inp, int max_iter, bool abs_prec) {
              FunctionTreeVector<D, double> vec;
              for (auto* tree : inp) vec.push_back({1.0, tree});
              mrcpp::add<D, double>(prec, out, vec, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);

    m.def("add",
          [](double prec, FunctionTree<D, double> &out, std::vector<std::tuple<double, FunctionTree<D, double> *>> &inp, int max_iter, bool abs_prec) {
              FunctionTreeVector<D, double> vec;
              for (auto& t : inp) vec.push_back({std::get<0>(t), std::get<1>(t)});
              mrcpp::add<D, double>(prec, out, vec, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);

    m.def("multiply",
          [](double prec, FunctionTree<D, double> &out, double c, FunctionTree<D, double> &inp_a, FunctionTree<D, double> &inp_b, int max_iter, bool abs_prec, bool use_max_norms) {
              mrcpp::multiply<D, double>(prec, out, c, inp_a, inp_b, max_iter, abs_prec, use_max_norms);
          },
          "prec"_a = -1.0,
          "out"_a,
          "c"_a = 1.0,
          "inp_a"_a,
          "inp_b"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false,
          "use_max_norms"_a = false);

    m.def("multiply",
          [](double prec, FunctionTree<D, double> &out, std::vector<FunctionTree<D, double> *> &inp, int max_iter, bool abs_prec, bool use_max_norms) {
              FunctionTreeVector<D, double> vec;
              for (auto* tree : inp) vec.push_back({1.0, tree});
              mrcpp::multiply<D, double>(prec, out, vec, max_iter, abs_prec, use_max_norms);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false,
          "use_max_norms"_a = false);

    m.def("multiply",
          [](double prec, FunctionTree<D, double> &out, std::vector<std::tuple<double, FunctionTree<D, double> *>> &inp, int max_iter, bool abs_prec, bool use_max_norms) {
              FunctionTreeVector<D, double> vec;
              for (auto& t : inp) vec.push_back({std::get<0>(t), std::get<1>(t)});
              mrcpp::multiply<D, double>(prec, out, vec, max_iter, abs_prec, use_max_norms);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false,
          "use_max_norms"_a = false);

    m.def(
        "dot",
        [](double prec, FunctionTree<D, double> &out, std::vector<FunctionTree<D, double>*> &inp_a, std::vector<FunctionTree<D, double>*> &inp_b, int maxIter, bool abs_prec) {
            FunctionTreeVector<D, double> vec_a, vec_b;
            for (auto* t : inp_a) vec_a.push_back({1.0, t});
            for (auto* t : inp_b) vec_b.push_back({1.0, t});
            mrcpp::dot<D, double>(prec, out, vec_a, vec_b, maxIter, abs_prec);
        },
        "prec"_a,
        "out"_a,
        "inp_a"_a,
        "inp_b"_a,
        "maxIter"_a = -1,
        "abs_prec"_a = false);

    m.def("power",
          [](double prec, FunctionTree<D, double> &out, FunctionTree<D, double> &inp, double pow, int max_iter, bool abs_prec) {
              mrcpp::power<D, double>(prec, out, inp, pow, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "pow"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);

    m.def("square",
          [](double prec, FunctionTree<D, double> &out, FunctionTree<D, double> &inp, int max_iter, bool abs_prec) {
              mrcpp::square<D, double>(prec, out, inp, max_iter, abs_prec);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);
     // ========================================================================
    // CompFunction free-function bindings 
    //
    // MRCPP currently instantiates most CompFunction free functions only for
    // D=3 (intended for 3D quantum chemistry codes).  We guard accordingly.
    // ========================================================================
    if constexpr (D == 3) {
        // Deep copy
        m.def("deep_copy",
              [](CompFunction<D> &out, const CompFunction<D> &inp) {
                  mrcpp::deep_copy<D>(out, inp);
              },
              "out"_a, "inp"_a,
              "Deep-copy a CompFunction into a (possibly differently-allocated) output. (3D only)");

        // Inner product <bra|ket>
        m.def("dot",
              [](CompFunction<D> bra, CompFunction<D> ket) {
                  return mrcpp::dot<D>(bra, ket);
              },
              "bra"_a, "ket"_a,
              "Compute the L2 inner product <bra|ket> of CompFunctions. Returns complex. (3D only)");

        // Weighted addition: out = a*inp_a + b*inp_b
        m.def("add",
              [](CompFunction<D> &out,
                 ComplexDouble a, CompFunction<D> inp_a,
                 ComplexDouble b, CompFunction<D> inp_b,
                 double prec, bool conjugate) {
                  mrcpp::add<D>(out, a, inp_a, b, inp_b, prec, conjugate);
              },
              "out"_a, "a"_a, "inp_a"_a, "b"_a, "inp_b"_a,
              "prec"_a = -1.0, "conjugate"_a = false,
              "Compute out = a*inp_a + b*inp_b for CompFunctions. (3D only)");

        // Linear combination: out = sum_i c_i * inp_i
        m.def("linear_combination",
              [](CompFunction<D> &out,
                 const std::vector<ComplexDouble> &c,
                 std::vector<CompFunction<D>> &inp,
                 double prec, bool conjugate) {
                  mrcpp::linear_combination<D>(out, c, inp, prec, conjugate);
              },
              "out"_a, "c"_a, "inp"_a,
              "prec"_a = -1.0, "conjugate"_a = false,
              "Compute out = sum_i c[i] * inp[i] for a list of CompFunctions. (3D only)");
    }
}
} // namespace vampyr
