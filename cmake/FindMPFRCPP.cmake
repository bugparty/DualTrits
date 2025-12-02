# Note: Search system locations for most up-to-date mpreal.h
# If not found, use the (older) vendored version in include/3rdparty
# as a fall-back.
find_path(MPFRCPP_INCLUDES
  NAMES
  mpreal.h
  PATHS
  ${MPFRCPP_ROOT}
  ${INCLUDE_INSTALL_DIR}
  ${CMAKE_SOURCE_DIR}/include/3rdparty
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFRCPP DEFAULT_MSG MPFRCPP_INCLUDES)
mark_as_advanced(MPFRCPP_INCLUDES)
