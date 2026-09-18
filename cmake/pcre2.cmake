# Prefix-aware regular expressions power JSON Schema string constraints and
# custom-tool grammars. Prefer the system development package; otherwise
# build a pinned, static PCRE2 (also keeps release peers dependency-free).
find_path(PCRE2_INCLUDE_DIR pcre2.h)
find_library(PCRE2_LIBRARY NAMES pcre2-8)
if(PCRE2_INCLUDE_DIR AND PCRE2_LIBRARY)
  add_library(dgpp_pcre2 INTERFACE)
  target_include_directories(dgpp_pcre2 INTERFACE ${PCRE2_INCLUDE_DIR})
  target_link_libraries(dgpp_pcre2 INTERFACE ${PCRE2_LIBRARY})
else()
  include(FetchContent)
  set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PCRE2_STATIC_PIC ON CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(pcre2
    URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.45/pcre2-10.45.tar.gz
    URL_HASH SHA256=0e138387df7835d7403b8351e2226c1377da804e0737db0e071b48f07c9d12ee
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(pcre2)
  # The server links the core archive statically. Keep PCRE2's standalone
  # install rules (POSIX archive, headers, tools and manuals) out of DGPP's
  # runtime package; only the linked targets need to be built. Using the
  # directory property also works with our CMake 3.25 minimum, before
  # FetchContent_Declare gained its EXCLUDE_FROM_ALL option in 3.28.
  set_property(DIRECTORY "${pcre2_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  add_library(dgpp_pcre2 ALIAS pcre2-8-static)
endif()
