// cutegen_oracle — nanobind binding over the BSD cutegen layout algebra.
//
// The vendored cutlass_compiler's cute dialect uses cutegen (header-only,
// BSD-3) as its layout/shape type engine; the closed-source official
// nvidia-cutlass-dsl links this same library. This binding exposes the
// algebra in-process so the object model asks ONE oracle (the library
// itself) instead of shelling out to the verifier binary.
//
// Semantics: text in (cute grammar, '?' for dynamic), text out. The
// default dynamic_traits_t is stateless — '?' leaves are anonymous;
// identity tracking requires an emission backend (see mlir_dynamic.hpp),
// which is out of scope here (documented in the report).
#include <cutegen/cutegen.hpp>
#include <cutegen/cutegen_base_dynamic.hpp>
#include <cutegen/layout.hpp>
#include <cutegen/visitors.hpp>
#include <chrono>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
namespace cg = cutegen;
using cg::layout;

static double g_total_ms = 0.0;
static long long g_calls = 0;

struct Timer {
    std::chrono::steady_clock::time_point t0;
    Timer() : t0(std::chrono::steady_clock::now()) {}
    ~Timer() {
        g_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        g_calls += 1;
    }
};

static layout must_parse(const std::string& text, const char* what) {
    auto v = cg::from_string<layout>(text);
    if (!v.has_value())
        throw nb::value_error(
            ("failed to parse " + std::string(what) + ": " + text).c_str());
    return v.value();
}

static std::string composition(const std::string& a, const std::string& b) {
    Timer _;
    return cg::to_string(composition(must_parse(a, "layout A"),
                                     must_parse(b, "layout B")));
}

static std::string coalesce(const std::string& a) {
    Timer _;
    return cg::to_string(coalesce(must_parse(a, "layout")));
}

static std::string flatten(const std::string& a) {
    Timer _;
    return cg::to_string(flatten(must_parse(a, "layout")));
}

// kind selects the cutegen overload: the dialect's zipped_divide/logical_
// divide take layout | tile | shape second operands with distinct
// semantics; the caller derives the kind from the operand's MLIR type.
static std::string zipped_divide(const std::string& a, const std::string& t,
                                 const std::string& kind) {
    Timer _;
    auto la = must_parse(a, "layout");
    if (kind == "shape") {
        auto sh = cg::from_string<cg::shape>(t);
        if (sh.has_value())
            return cg::to_string(zipped_divide(la, sh.value()));
    } else if (kind == "tile") {
        auto tile = cg::from_string<cg::tile>(t);
        if (tile.has_value())
            return cg::to_string(zipped_divide(la, tile.value()));
    } else {
        auto lb = cg::from_string<cg::layout>(t);
        if (lb.has_value())
            return cg::to_string(zipped_divide(la, lb.value()));
    }
    // The kind is authoritative — no silent cross-kind fallback (the
    // grammars overlap and the overloads have distinct semantics).
    throw nb::value_error(("zipped_divide: operand not a parseable " +
                           kind + ": " + t).c_str());
}

static std::string logical_divide(const std::string& a, const std::string& t,
                                  const std::string& kind) {
    Timer _;
    auto la = must_parse(a, "layout");
    if (kind == "shape") {
        auto sh = cg::from_string<cg::shape>(t);
        if (sh.has_value())
            return cg::to_string(logical_divide(la, sh.value()));
    } else if (kind == "tile") {
        auto tile = cg::from_string<cg::tile>(t);
        if (tile.has_value())
            return cg::to_string(logical_divide(la, tile.value()));
    } else {
        auto lb = cg::from_string<cg::layout>(t);
        if (lb.has_value())
            return cg::to_string(logical_divide(la, lb.value()));
    }
    throw nb::value_error(("logical_divide: unparsable operand " + t).c_str());
}

static long long count_dynamics(const std::string& a) {
    Timer _;
    std::vector<cg::dynamic_t> dyns;
    cg::collect_dynamics(dyns, must_parse(a, "layout"));
    return (long long)dyns.size();
}

static std::string op_slice(const std::string& crd, const std::string& lay) {
    Timer _;
    auto c = cg::from_string<cg::shape>(crd);
    if (!c.has_value())
        throw nb::value_error(("failed to parse coord: " + crd).c_str());
    return cg::to_string(cutegen::slice(c.value(), must_parse(lay, "layout")));
}

static std::string selfcheck() {
    Timer _;
    auto r = composition(must_parse("(4,8):(8,1)", "a"),
                         must_parse("(32,4):(4,1)", "b"));
    return cg::to_string(r);   // expected "(32,4):(1,8)"
}

NB_MODULE(_cutegen_oracle, m) {
    m.def("composition", &composition);
    m.def("coalesce", &coalesce);
    m.def("flatten", &flatten);
    m.def("zipped_divide", &zipped_divide);
    m.def("logical_divide", &logical_divide);
    m.def("slice", &op_slice);
    m.def("count_dynamics", &count_dynamics);
    m.def("selfcheck", &selfcheck);
    m.def("stats", []() {
        return std::make_pair(g_calls, g_total_ms);
    });
    m.attr("__doc__") = "in-process cutegen layout algebra (BSD-3)";
}
