#include <fstream>
#include <iostream>

#include "common/base64.hpp"
#include "serve/image_inputs.hpp"
int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    std::ifstream f(argv[1], std::ios::binary);
    std::string bytes(std::istreambuf_iterator<char>(f), {});
    auto image = dgpp::serve::prepare_glm_image(
        dgpp::minijson::Value::make_object(
            {{"url", dgpp::minijson::Value::make_owned_string("data:image/jpeg;base64," +
                                                              dgpp::encode_base64(bytes))}}),
        "image_url");
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(image.rgb.data()), image.rgb.size());
    if (!out) return 1;
    uint64_t hash = 14695981039346656037ull;
    for (auto b : image.rgb) {
      hash ^= b;
      hash *= 1099511628211ull;
    }
    std::cout << image.width << ' ' << image.height << ' ' << image.tokens << ' ' << hash << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
