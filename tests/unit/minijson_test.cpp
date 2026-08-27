#include "loaders/minijson.hpp"

#include <cstring>

#include "common/test.hpp"

using dgpp::minijson::parse;
using dgpp::minijson::Value;

DGPP_TEST(minijson_object_scalars) {
  const char* json = R"({"a":1,"b":-2,"c":3.5,"d":true,"e":false,"f":null})";
  auto r = parse(json);
  const Value& o = r.root;
  if (!o.is_object()) throw std::runtime_error("not object");
  if (o.at("a").as_int() != 1) throw std::runtime_error("a");
  if (o.at("b").as_int() != -2) throw std::runtime_error("b");
  if (o.at("c").as_double() != 3.5) throw std::runtime_error("c");
  if (!o.at("d").as_bool()) throw std::runtime_error("d");
  if (o.at("e").as_bool(true)) throw std::runtime_error("e");
  if (!o.at("f").is_null()) throw std::runtime_error("f");
}

DGPP_TEST(minijson_nested_arrays_and_views) {
  // Shapes come from safetensors headers: array of ints under string keys.
  const char* json =
      R"({"t":{"dtype":"BF16","shape":[2,3,4],"data_offsets":[0,48]}})";
  auto r = parse(json);
  const Value& shape = r.root.at("t").at("shape");
  if (!shape.is_array() || shape.items().size() != 3)
    throw std::runtime_error("shape");
  if (shape.items()[2].as_int() != 4) throw std::runtime_error("elem");
  if (std::string_view(r.root.at("t").at("dtype").as_string()) != "BF16")
    throw std::runtime_error("dtype view");
}

DGPP_TEST(minijson_escape_decode_produces_owned_storage) {
  const char* json = "{\"k\\n\\u0041\" : \"v\\t\\\"q\\\"\"}";
  auto r = parse(json);
  if (r.root.at(std::string_view("k\nA")).as_string() !=
      std::string_view("v\t\"q\""))
    throw std::runtime_error("escape decode mismatch");
}

DGPP_TEST(minijson_trailing_whitespace_and_consumed_count) {
  auto r = parse("  [1, 2, 3]   \n\t ");
  if (!r.root.is_array() || r.root.items().size() != 3)
    throw std::runtime_error("array");
  if (r.consumed != std::strlen("  [1, 2, 3]"))
    throw std::runtime_error("consumed count");
}

DGPP_TEST(minijson_malformed_throws) {
  bool threw = false;
  try {
    parse("{\"a\":}");
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("should throw on malformed");
}
