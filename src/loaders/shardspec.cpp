#include "loaders/shardspec.hpp"

#include <format>

namespace dgpp {

ShardedCheckpoint::ShardedCheckpoint(const Config& c) : cfg_(c) {
  std::string json = [&] {
    FILE* f = fopen(cfg_.spec_path.c_str(), "rb");
    if (!f)
      throw std::runtime_error("cannot open shardspec " + cfg_.spec_path);
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(static_cast<size_t>(len), '\0');
    if (fread(s.data(), 1, static_cast<size_t>(len), f) !=
        static_cast<size_t>(len)) {
      fclose(f);
      throw std::runtime_error("short read on shardspec");
    }
    fclose(f);
    return s;
  }();

  auto parsed = minijson::parse(json);
  model_id_ = std::string(parsed.root.at("model").as_string());
  for (const auto& sh : parsed.root.at("shards").items()) {
    ShardInfo si;
    si.filename = std::string(sh.at("file").as_string());
    si.size_bytes = static_cast<uint64_t>(sh.at("size_bytes").as_int());
    shards_.push_back(std::move(si));
  }
  for (const auto& m : parsed.root.at("tensors").members()) {
    SpecTensor st;
    st.shard = static_cast<uint32_t>(
        m.value.at("shard").as_int());
    auto ds = m.value.at("dtype").as_string();
    auto dt = dtype_from_string(ds);
    if (!dt)
      throw std::runtime_error(
          std::format("shardspec {}: bad dtype {}", cfg_.spec_path, ds));
    st.dtype = *dt;
    for (const auto& d : m.value.at("shape").items())
      st.shape.push_back(d.as_int());
    st.role = std::string(m.value.at("role").as_string(""));
    st.nbytes = static_cast<uint64_t>(m.value.at("nbytes").as_int(0));
    tensor_list_.emplace_back(m.key, st);
  }
  tensor_list_.shrink_to_fit();
  for (auto& [name, st] : tensor_list_) by_name_[name] = st;

  if (cfg_.eager_bind) bind_all();
}

void ShardedCheckpoint::bind_all() {
  namespace fs = std::filesystem;
  bound_.reserve(shards_.size());
  for (const auto& sh : shards_) {
    fs::path p = fs::path(cfg_.dir) / sh.filename;
    bound_.push_back(SafetensorsFile::open(p.string()));
  }
}

const SpecTensor* ShardedCheckpoint::find(std::string_view name) const {
  auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &it->second;
}

const void* ShardedCheckpoint::tensor_data(
    std::string_view name, DType expect_dtype,
    std::initializer_list<int64_t> expect_shape) const {
  const SpecTensor* st = find(name);
  if (!st)
    throw std::runtime_error(
        std::format("tensor '{}' not in shardspec", name));
  if (st->dtype != expect_dtype)
    throw std::runtime_error(std::format(
        "tensor '{}' dtype {} != expected {}", name, dtype_name(st->dtype),
        dtype_name(expect_dtype)));
  if (expect_shape.size() && expect_shape.size() != st->shape.size())
    throw std::runtime_error(
        std::format("tensor '{}' rank mismatch", name));
  const SafetensorsFile& f = *bound_.at(st->shard);
  const TensorInfo* ti = f.find(name);
  if (!ti)
    throw std::runtime_error(
        std::format("tensor '{}' absent in bound shard {}", name, f.path()));
  return ti->data;
}

}  // namespace dgpp
