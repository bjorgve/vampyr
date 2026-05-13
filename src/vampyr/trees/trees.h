#pragma once

#include <complex>
#include <filesystem>
#include <type_traits>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl/filesystem.h>

#include <MRCPP/utils/CompFunction.h>
#include <MRCPP/trees/FunctionNode.h>
#include <MRCPP/trees/FunctionTree.h>
#include <MRCPP/trees/MWNode.h>
#include <MRCPP/trees/MWTree.h>
#include <MRCPP/trees/TreeIterator.h>

namespace vampyr {

using ComplexDouble = std::complex<double>;

// Helper to generate type suffix for class names
template <typename T>
constexpr const char* type_suffix() {
    if constexpr (std::is_same_v<T, double>) {
        return "";
    } else {
        return "_Complex";
    }
}

// Helper to get dtype string for Python
template <typename T>
constexpr const char* dtype_string() {
    if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else {
        return "complex128";
    }
}
template <int D>
auto impl__add__(mrcpp::FunctionTree<D, double> *inp_a, mrcpp::FunctionTree<D, double> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp_a->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({1.0, inp_a});
    vec.push_back({1.0, inp_b});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl__sub__(mrcpp::FunctionTree<D, double> *inp_a, mrcpp::FunctionTree<D, double> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp_a->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({1.0, inp_a});
    vec.push_back({-1.0, inp_b});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl__mul__(mrcpp::FunctionTree<D, double> *inp_a, mrcpp::FunctionTree<D, double> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp_a->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({1.0, inp_a});
    vec.push_back({1.0, inp_b});
    build_grid(*out, vec);
    build_grid(*out, 1);
    multiply(-1, *out, vec);
    return out;
};

template <int D>
auto impl__mul__(mrcpp::FunctionTree<D, double> *inp_a, double c) -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp_a->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({c, inp_a});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D> auto impl__pos__(mrcpp::FunctionTree<D, double> *inp) -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp->getMRA());
    copy_grid(*out, *inp);
    copy_func(*out, *inp);
    return out;
};

template <int D> auto impl__neg__(mrcpp::FunctionTree<D, double> *inp) -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({-1.0, inp});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl__truediv__(mrcpp::FunctionTree<D, double> *inp, double c) -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp->getMRA());
    FunctionTreeVector<D, double> vec;
    vec.push_back({1.0 / c, inp});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D> auto impl__pow__(mrcpp::FunctionTree<D, double> *inp, double c) -> std::unique_ptr<mrcpp::FunctionTree<D, double>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, double>>(inp->getMRA());
    copy_grid(*out, *inp);
    copy_func(*out, *inp);
    refine_grid(*out, 1);
    out->power(c);
    return out;
};

// Complex arithmetic operators
template <int D>
auto impl_complex__add__(mrcpp::FunctionTree<D, ComplexDouble> *inp_a, mrcpp::FunctionTree<D, ComplexDouble> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp_a->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({ComplexDouble(1.0, 0.0), inp_a});
    vec.push_back({ComplexDouble(1.0, 0.0), inp_b});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl_complex__sub__(mrcpp::FunctionTree<D, ComplexDouble> *inp_a, mrcpp::FunctionTree<D, ComplexDouble> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp_a->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({ComplexDouble(1.0, 0.0), inp_a});
    vec.push_back({ComplexDouble(-1.0, 0.0), inp_b});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl_complex__mul__(mrcpp::FunctionTree<D, ComplexDouble> *inp_a, mrcpp::FunctionTree<D, ComplexDouble> *inp_b)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp_a->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({ComplexDouble(1.0, 0.0), inp_a});
    vec.push_back({ComplexDouble(1.0, 0.0), inp_b});
    build_grid(*out, vec);
    build_grid(*out, 1);
    multiply(-1, *out, vec);
    return out;
};

template <int D>
auto impl_complex__mul__(mrcpp::FunctionTree<D, ComplexDouble> *inp_a, ComplexDouble c)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp_a->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({c, inp_a});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl_complex__pos__(mrcpp::FunctionTree<D, ComplexDouble> *inp)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp->getMRA());
    copy_grid(*out, *inp);
    copy_func(*out, *inp);
    return out;
};

template <int D>
auto impl_complex__neg__(mrcpp::FunctionTree<D, ComplexDouble> *inp)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({ComplexDouble(-1.0, 0.0), inp});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D>
auto impl_complex__truediv__(mrcpp::FunctionTree<D, ComplexDouble> *inp, ComplexDouble c)
    -> std::unique_ptr<mrcpp::FunctionTree<D, ComplexDouble>> {
    using namespace mrcpp;
    auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp->getMRA());
    FunctionTreeVector<D, ComplexDouble> vec;
    vec.push_back({ComplexDouble(1.0, 0.0) / c, inp});
    build_grid(*out, vec);
    add(-1.0, *out, vec);
    return out;
};

template <int D> void trees(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<MWTree<D, double>>(m, (std::string("MWTree") + std::to_string(D) + "D").c_str())
        .def("MRA", &MWTree<D, double>::getMRA, py::return_value_policy::reference_internal)
        .def("nNodes", &MWTree<D, double>::getNNodes)
        .def("nEndNodes", &MWTree<D, double>::getNEndNodes)
        .def("nRootNodes", &MWTree<D, double>::getNRootNodes)
        .def("fetchEndNode",
             [](MWTree<D, double>& tree, int i) -> MWNode<D, double>& { return tree.getEndMWNode(i); },
             py::return_value_policy::reference_internal)
        .def("fetchRootNode",
             [](MWTree<D, double>& tree, int i) -> MWNode<D, double>& { return tree.getRootMWNode(i); },
             py::return_value_policy::reference_internal)
        .def("rootScale", &MWTree<D, double>::getRootScale)
        .def("depth", &MWTree<D, double>::getDepth)
        .def("setZero",
             [](MWTree<D, double> *out) {
                 out->setZero();
                 return out;
             })
        .def("clear", &MWTree<D, double>::clear)
        .def("setName", &MWTree<D, double>::setName)
        .def("name", &MWTree<D, double>::getName)
        .def("fetchNode",
             [](MWTree<D, double>& tree, NodeIndex<D> idx) -> MWNode<D, double>& { return tree.getNode(idx); },
             py::return_value_policy::reference_internal)
        .def("squaredNorm", &MWTree<D, double>::getSquareNorm)
        .def("norm",
             [](MWTree<D, double> &tree) {
                 auto sqNorm = tree.getSquareNorm();
                 return (sqNorm >= 0.0) ? std::sqrt(sqNorm) : -1.0;
             })
        .def("__str__", [](MWTree<D, double> &tree) {
            std::ostringstream os;
            os << tree;
            return os.str();
        });

    auto ft_real = py::class_<FunctionTree<D, double>, MWTree<D, double>, RepresentableFunction<D, double>>(m, (std::string("FunctionTree") + std::to_string(D) + "D").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, const std::string &>(), "mra"_a, "name"_a = "nn")
        .def("nGenNodes", &FunctionTree<D, double>::getNGenNodes)
        .def("deleteGenerated", &FunctionTree<D, double>::deleteGenerated)
        .def("integrate", &FunctionTree<D, double>::integrate)
        .def("normalize",
             [](FunctionTree<D, double> *out) {
                 out->normalize();
                 return out;
             })
        .def(
            "saveTree",
            [](FunctionTree<D, double> &obj, const std::string &filename) {
                namespace fs = std::filesystem;
                obj.saveTree(filename);
                return fs::absolute(fs::path(filename + ".tree"));
            },
            "filename"_a)
        .def("loadTree", &FunctionTree<D, double>::loadTree, "filename"_a)
        .def(
            "crop",
            [](FunctionTree<D, double> *out, double prec, bool abs_prec) {
                out->crop(prec, 1.0, abs_prec);
                return out;
            },
            "prec"_a,
            "abs_prec"_a = false)
        .def("deepCopy",
             [](FunctionTree<D, double> *inp) {
                 auto out = std::make_unique<FunctionTree<D, double>>(inp->getMRA());
                 copy_grid(*out, *inp);
                 copy_func(*out, *inp);
                 return out;
             })
        .def("__call__", [](FunctionTree<D, double> &func, const Coord<D> &r) { return func.evalf_precise(r); })
        .def_property_readonly("dtype", [](FunctionTree<D, double>&) { return "float64"; })
        .def_property_readonly("is_complex", [](FunctionTree<D, double>&) { return false; })
        .def("to_complex",
             [](FunctionTree<D, double> *inp) {
                 FunctionTree<D, ComplexDouble> *out = nullptr;
                 inp->CopyTreeToComplex(out);
                 return std::unique_ptr<FunctionTree<D, ComplexDouble>>(out);
             })
        .def("__pos__", &impl__pos__<D>, py::is_operator())
        .def("__neg__", &impl__neg__<D>, py::is_operator())
        .def("__add__", &impl__add__<D>, py::is_operator())
        .def("__iadd__", &impl__add__<D>, py::is_operator())
        .def("__sub__", &impl__sub__<D>, py::is_operator())
        .def("__isub__", &impl__sub__<D>, py::is_operator())
        .def("__mul__", py::overload_cast<FunctionTree<D, double> *, FunctionTree<D, double> *>(&impl__mul__<D>), py::is_operator())
        .def("__mul__", py::overload_cast<FunctionTree<D, double> *, double>(&impl__mul__<D>), py::is_operator())
        .def("__imul__", py::overload_cast<FunctionTree<D, double> *, FunctionTree<D, double> *>(&impl__mul__<D>), py::is_operator())
        .def("__imul__", py::overload_cast<FunctionTree<D, double> *, double>(&impl__mul__<D>), py::is_operator())
        .def("__rmul__", py::overload_cast<FunctionTree<D, double> *, double>(&impl__mul__<D>), py::is_operator())
        .def("__truediv__", &impl__truediv__<D>, py::is_operator())
        .def("__itruediv__", &impl__truediv__<D>, py::is_operator())
        .def("__pow__", &impl__pow__<D>, py::is_operator())
        .def("__ipow__", &impl__pow__<D>, py::is_operator());

    // 1D scalar evaluation: tree(0.5) instead of tree([0.5])
    if constexpr (D == 1) {
        ft_real.def("__call__", [](FunctionTree<D, double> &self, double x) {
            return self.evalf_precise({x});
        });
    }

    py::class_<MWNode<D, double>>(m, (std::string("MWNode") + std::to_string(D) + "D").c_str())
        .def("depth", &MWNode<D, double>::getDepth)
        .def("scale", &MWNode<D, double>::getScale)
        .def("nCoefs", &MWNode<D, double>::getNCoefs)
        .def("nChildren", &MWNode<D, double>::getNChildren)
        .def("index",
             [](MWNode<D, double>& node) -> const NodeIndex<D>& { return node.getNodeIndex(); },
             py::return_value_policy::reference_internal)
        .def("norm",
             [](MWNode<D, double> &node) {
                 auto sqNorm = node.getSquareNorm();
                 return (sqNorm >= 0.0) ? std::sqrt(sqNorm) : -1.0;
             })
        .def("squaredNorm", &MWNode<D, double>::getSquareNorm)
        .def("scalingNorm", &MWNode<D, double>::getScalingNorm)
        .def("waveletNorm", &MWNode<D, double>::getWaveletNorm)
        .def("componentNorm", &MWNode<D, double>::getComponentNorm)
        .def("isAllocated", &MWNode<D, double>::isAllocated)
        .def("isRootNode", &MWNode<D, double>::isRootNode)
        .def("isEndNode", &MWNode<D, double>::isEndNode)
        .def("isLeafNode", &MWNode<D, double>::isLeafNode)
        .def("isBranchNode", &MWNode<D, double>::isBranchNode)
        .def("isGenNode", &MWNode<D, double>::isGenNode)
        .def("hasParent", &MWNode<D, double>::hasParent)
        .def("hasCoefs", &MWNode<D, double>::hasCoefs)
        .def("center", &MWNode<D, double>::getCenter)
        .def("upperBounds", &MWNode<D, double>::getUpperBounds)
        .def("lowerBounds", &MWNode<D, double>::getLowerBounds)
        .def("__str__", [](MWNode<D, double> &node) {
            std::ostringstream os;
            os << node;
            return os.str();
        });

    py::class_<FunctionNode<D, double>, MWNode<D, double>, std::unique_ptr<FunctionNode<D, double>, py::nodelete>>(m, (std::string("FunctionNode") + std::to_string(D) + "D").c_str())
        .def("integrate", &FunctionNode<D, double>::integrate);

    py::class_<NodeIndex<D>>(m, (std::string("NodeIndex") + std::to_string(D) + "D").c_str())
        .def(py::init<int, const std::array<int, D>>(), "scale"_a = 0, "translation"_a = std::array<int, D>{})
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("child", &NodeIndex<D>::child)
        .def("parent", &NodeIndex<D>::parent)
        .def("scale", &NodeIndex<D>::getScale)
        .def("setScale", &NodeIndex<D>::setScale)
        .def("translation", py::overload_cast<>(&NodeIndex<D>::getTranslation, py::const_))
        .def("translation", py::overload_cast<int>(&NodeIndex<D>::getTranslation, py::const_))
        .def("setTranslation", py::overload_cast<const std::array<int, D> &>(&NodeIndex<D>::setTranslation))
        .def("__str__", [](NodeIndex<D> &idx) {
            std::ostringstream os;
            os << idx;
            return os.str();
        });

    py::class_<TreeIterator<D, double>>(m, (std::string("TreeIterator") + std::to_string(D) + "D").c_str())
        .def(py::init<Traverse, Iterator>(), "traverse"_a = TopDown, "iterator"_a = Lebesgue)
        .def(py::init<MWTree<D, double> &, Traverse, Iterator>(), "tree"_a, "traverse"_a = TopDown, "iterator"_a = Lebesgue)
        .def("setReturnGenNodes", &TreeIterator<D, double>::setReturnGenNodes)
        .def("setMaxDepth", &TreeIterator<D, double>::setMaxDepth)
        .def("setTraverse", &TreeIterator<D, double>::setTraverse)
        .def("setIterator", &TreeIterator<D, double>::setIterator)
        .def("get", [](TreeIterator<D, double>& it) -> MWNode<D, double>& { return it.getNode(); }, py::return_value_policy::reference_internal)
        .def("init", &TreeIterator<D, double>::init)
        .def("next", &TreeIterator<D, double>::next);

    // Complex MWTree bindings (all dimensions)
    py::class_<MWTree<D, ComplexDouble>>(m, (std::string("MWTree") + std::to_string(D) + "D_Complex").c_str())
        .def("MRA", &MWTree<D, ComplexDouble>::getMRA, py::return_value_policy::reference_internal)
        .def("nNodes", &MWTree<D, ComplexDouble>::getNNodes)
        .def("nEndNodes", &MWTree<D, ComplexDouble>::getNEndNodes)
        .def("nRootNodes", &MWTree<D, ComplexDouble>::getNRootNodes)
        .def("fetchEndNode",
             [](MWTree<D, ComplexDouble>& tree, int i) -> MWNode<D, ComplexDouble>& { return tree.getEndMWNode(i); },
             py::return_value_policy::reference_internal)
        .def("fetchRootNode",
             [](MWTree<D, ComplexDouble>& tree, int i) -> MWNode<D, ComplexDouble>& { return tree.getRootMWNode(i); },
             py::return_value_policy::reference_internal)
        .def("rootScale", &MWTree<D, ComplexDouble>::getRootScale)
        .def("depth", &MWTree<D, ComplexDouble>::getDepth)
        .def("setZero",
             [](MWTree<D, ComplexDouble> *out) {
                 out->setZero();
                 return out;
             })
        .def("clear", &MWTree<D, ComplexDouble>::clear)
        .def("setName", &MWTree<D, ComplexDouble>::setName)
        .def("name", &MWTree<D, ComplexDouble>::getName)
        .def("fetchNode",
             [](MWTree<D, ComplexDouble>& tree, NodeIndex<D> idx) -> MWNode<D, ComplexDouble>& { return tree.getNode(idx); },
             py::return_value_policy::reference_internal)
        .def("squaredNorm", &MWTree<D, ComplexDouble>::getSquareNorm)
        .def("norm",
             [](MWTree<D, ComplexDouble> &tree) {
                 auto sqNorm = tree.getSquareNorm();
                 return (sqNorm >= 0.0) ? std::sqrt(sqNorm) : -1.0;
             })
        .def("__str__", [](MWTree<D, ComplexDouble> &tree) {
            std::ostringstream os;
            os << tree;
            return os.str();
        });

    // Complex FunctionTree bindings (all dimensions)
    auto ft_complex = py::class_<FunctionTree<D, ComplexDouble>, MWTree<D, ComplexDouble>>(m, (std::string("FunctionTree") + std::to_string(D) + "D_Complex").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, const std::string &>(), "mra"_a, "name"_a = "nn")
        .def("nGenNodes", &FunctionTree<D, ComplexDouble>::getNGenNodes)
        .def("deleteGenerated", &FunctionTree<D, ComplexDouble>::deleteGenerated)
        .def("integrate", &FunctionTree<D, ComplexDouble>::integrate)
        .def("normalize",
             [](FunctionTree<D, ComplexDouble> *out) {
                 out->normalize();
                 return out;
             })
        .def(
            "crop",
            [](FunctionTree<D, ComplexDouble> *out, double prec, bool abs_prec) {
                out->crop(prec, 1.0, abs_prec);
                return out;
            },
            "prec"_a,
            "abs_prec"_a = false)
        .def("deepCopy",
             [](FunctionTree<D, ComplexDouble> *inp) {
                 auto out = std::make_unique<FunctionTree<D, ComplexDouble>>(inp->getMRA());
                 copy_grid(*out, *inp);
                 copy_func(*out, *inp);
                 return out;
             })
        .def("__call__", [](FunctionTree<D, ComplexDouble> &func, const Coord<D> &r) { return func.evalf_precise(r); })
        .def_property_readonly("dtype", [](FunctionTree<D, ComplexDouble>&) { return "complex128"; })
        .def_property_readonly("is_complex", [](FunctionTree<D, ComplexDouble>&) { return true; })
        // Type conversion methods
        .def("real",
             [](FunctionTree<D, ComplexDouble> *inp) {
                 return std::unique_ptr<FunctionTree<D, double>>(inp->Real());
             })
        .def("imag",
             [](FunctionTree<D, ComplexDouble> *inp) {
                 return std::unique_ptr<FunctionTree<D, double>>(inp->Imag());
             })
        .def("conj",
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
             })
        // Arithmetic operators for complex trees
        .def("__pos__", &impl_complex__pos__<D>, py::is_operator())
        .def("__neg__", &impl_complex__neg__<D>, py::is_operator())
        .def("__add__", &impl_complex__add__<D>, py::is_operator())
        .def("__iadd__", &impl_complex__add__<D>, py::is_operator())
        .def("__sub__", &impl_complex__sub__<D>, py::is_operator())
        .def("__isub__", &impl_complex__sub__<D>, py::is_operator())
        .def("__mul__", py::overload_cast<FunctionTree<D, ComplexDouble> *, FunctionTree<D, ComplexDouble> *>(&impl_complex__mul__<D>), py::is_operator())
        .def("__mul__", py::overload_cast<FunctionTree<D, ComplexDouble> *, ComplexDouble>(&impl_complex__mul__<D>), py::is_operator())
        .def("__imul__", py::overload_cast<FunctionTree<D, ComplexDouble> *, FunctionTree<D, ComplexDouble> *>(&impl_complex__mul__<D>), py::is_operator())
        .def("__imul__", py::overload_cast<FunctionTree<D, ComplexDouble> *, ComplexDouble>(&impl_complex__mul__<D>), py::is_operator())
        .def("__rmul__", py::overload_cast<FunctionTree<D, ComplexDouble> *, ComplexDouble>(&impl_complex__mul__<D>), py::is_operator())
        .def("__truediv__", &impl_complex__truediv__<D>, py::is_operator())
        .def("__itruediv__", &impl_complex__truediv__<D>, py::is_operator());

    // 1D scalar evaluation: complex_tree(0.5) instead of complex_tree([0.5])
    if constexpr (D == 1) {
        ft_complex.def("__call__", [](FunctionTree<D, ComplexDouble> &self, double x) {
            return self.evalf_precise({x});
        });
    }

    // Complex MWNode bindings (all dimensions)
    py::class_<MWNode<D, ComplexDouble>>(m, (std::string("MWNode") + std::to_string(D) + "D_Complex").c_str())
        .def("depth", &MWNode<D, ComplexDouble>::getDepth)
        .def("scale", &MWNode<D, ComplexDouble>::getScale)
        .def("nCoefs", &MWNode<D, ComplexDouble>::getNCoefs)
        .def("nChildren", &MWNode<D, ComplexDouble>::getNChildren)
        .def("index",
             [](MWNode<D, ComplexDouble>& node) -> const NodeIndex<D>& { return node.getNodeIndex(); },
             py::return_value_policy::reference_internal)
        .def("norm",
             [](MWNode<D, ComplexDouble> &node) {
                 auto sqNorm = node.getSquareNorm();
                 return (sqNorm >= 0.0) ? std::sqrt(sqNorm) : -1.0;
             })
        .def("squaredNorm", &MWNode<D, ComplexDouble>::getSquareNorm)
        .def("scalingNorm", &MWNode<D, ComplexDouble>::getScalingNorm)
        .def("waveletNorm", &MWNode<D, ComplexDouble>::getWaveletNorm)
        .def("componentNorm", &MWNode<D, ComplexDouble>::getComponentNorm)
        .def("isAllocated", &MWNode<D, ComplexDouble>::isAllocated)
        .def("isRootNode", &MWNode<D, ComplexDouble>::isRootNode)
        .def("isEndNode", &MWNode<D, ComplexDouble>::isEndNode)
        .def("isLeafNode", &MWNode<D, ComplexDouble>::isLeafNode)
        .def("isBranchNode", &MWNode<D, ComplexDouble>::isBranchNode)
        .def("isGenNode", &MWNode<D, ComplexDouble>::isGenNode)
        .def("hasParent", &MWNode<D, ComplexDouble>::hasParent)
        .def("hasCoefs", &MWNode<D, ComplexDouble>::hasCoefs)
        .def("center", &MWNode<D, ComplexDouble>::getCenter)
        .def("upperBounds", &MWNode<D, ComplexDouble>::getUpperBounds)
        .def("lowerBounds", &MWNode<D, ComplexDouble>::getLowerBounds)
        .def("__str__", [](MWNode<D, ComplexDouble> &node) {
            std::ostringstream os;
            os << node;
            return os.str();
        });

    // Complex FunctionNode bindings (all dimensions)
    py::class_<FunctionNode<D, ComplexDouble>, MWNode<D, ComplexDouble>, std::unique_ptr<FunctionNode<D, ComplexDouble>, py::nodelete>>(m, (std::string("FunctionNode") + std::to_string(D) + "D_Complex").c_str())
        .def("integrate", &FunctionNode<D, ComplexDouble>::integrate);

        // ----------------------------------------------------------------------
    // CompFunction<D>  --  multi-component function wrapper from MRCPP.
    // Holds either real or complex FunctionTree(s) internally; switch with
    // .defreal() / .defcomplex().
    // ----------------------------------------------------------------------
    auto comp_name = std::string("CompFunction") + std::to_string(D) + "D";

    py::class_<CompFunction<D>>(m, comp_name.c_str())
        // (G) constructors
        .def(py::init<MultiResolutionAnalysis<D> &>(), "mra"_a,
             py::keep_alive<1, 2>())
        .def(py::init<>())
        .def(py::init<int>(), "n1"_a)
        .def(py::init<int, bool>(), "n1"_a, "share"_a)

        // (C) data accessors -- inline in CompFunction.h lines 113-119
        .def("Ncomp",      &CompFunction<D>::Ncomp)
        .def("rank",       &CompFunction<D>::rank)
        .def("conj",       &CompFunction<D>::conj)
        .def("isreal",     &CompFunction<D>::isreal)
        .def("iscomplex",  &CompFunction<D>::iscomplex)
        .def("share",      &CompFunction<D>::share)

        // (C) type-switching
        .def("defreal",    &CompFunction<D>::defreal)
        .def("defcomplex", &CompFunction<D>::defcomplex)

        // (C) multiplicative scalar factor
        .def("getFac", &CompFunction<D>::getFac)
        .def("setFac", &CompFunction<D>::setFac, "fac"_a)

        // (C) integrals & norms
        .def("integrate",     &CompFunction<D>::integrate)
        .def("norm",          &CompFunction<D>::norm)
        .def("getSquareNorm", &CompFunction<D>::getSquareNorm)

        // (C) allocation
        .def("alloc",      &CompFunction<D>::alloc,
             "nalloc"_a = 1, "zero"_a = true)
        .def("alloc_comp", &CompFunction<D>::alloc_comp, "i"_a = 0)

        // (C) sizes
        .def("getSizeNodes", &CompFunction<D>::getSizeNodes)
        .def("getNNodes",    &CompFunction<D>::getNNodes)

        // (C) modifiers
        .def("rescale", &CompFunction<D>::rescale, "c"_a)
        .def("crop",    &CompFunction<D>::crop,
             "prec"_a, "absPrec"_a = true)
        .def("free",    py::overload_cast<>(&CompFunction<D>::free))

        // (G) public name string
          // (C) P2.2 -- complex conjugation in place.  CompFunction.h:159 declares
        //     `void dagger();`.  This flips the sign of the imag parts.
        .def("dagger", &CompFunction<D>::dagger,
             "Conjugate the function in place: f -> conj(f).")

        // (C) P2.1 -- access underlying FunctionTrees.  Mirrors the binding
        //     style of FunctionTree<D, ComplexDouble>::real() in this same file.
        //     Returns a reference into the CompFunction; lifetime tied to self.
        .def("real",
             [](CompFunction<D> &self, int i) -> FunctionTree<D, double> & {
                 return self.real(i);
             },
             py::arg("i") = 0,
             py::return_value_policy::reference_internal,
             "Get a reference to the i-th real-valued component tree.")

        .def("complex",
             [](CompFunction<D> &self, int i) -> FunctionTree<D, ComplexDouble> & {
                 return self.complex(i);
             },
             py::arg("i") = 0,
             py::return_value_policy::reference_internal,
             "Get a reference to the i-th complex-valued component tree.")
        .def_readwrite("name", &CompFunction<D>::name);
}
} // namespace vampyr
