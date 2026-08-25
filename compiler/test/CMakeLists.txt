# LIT configuration for selfcute tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-selfcute-lit
  "Running the selfcute dialect LIT tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS selfcute-opt
)
set_target_properties(check-selfcute-lit PROPERTIES FOLDER "Tests")
