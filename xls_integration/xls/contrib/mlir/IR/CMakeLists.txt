# ===----------------------------------------------------------------------===//
#
# Super hacky XLS dialect integration into Dynamatic.
#
# Since XLS uses a completely different file naming convention, directory
# structure, and build system, this cannot simply make use of standard LLVM
# cmake integration.
#
# All tablegen files are picked out directly from the XLS checkout, while
# source files are symlinked into this directory.
#
# Note also that because the two projects are organised in such a different
# fashion and rely on LLVM versions that are around 2 years apart, this uses
# a fork of XLS with slight modifications to the XLS dialect definition.
#
# These are (mostly) selected by the `DYNAMATIC_INTEROP` macro.
#
# ===----------------------------------------------------------------------===//

#
# TableGen
#

# Location of XLS checkout:
get_filename_component(xls_root_dir_abs "../../../../../xls" REALPATH)

# Location of XLS dialect ODS:
set(dialect_file ${xls_root_dir_abs}/xls/contrib/mlir/IR/xls_ops.td)
get_filename_component(dialect_dir "${dialect_file}" DIRECTORY)

# Run tablegen for XLS ODS:
set(LLVM_TARGET_DEFINITIONS ${dialect_file})
mlir_tablegen(xls_ops.h.inc -gen-op-decls EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops.cc.inc -gen-op-defs EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_attrs.h.inc -gen-attrdef-decls EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_attrs.cc.inc -gen-attrdef-defs EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_enums.h.inc -gen-enum-decls EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_enums.cc.inc -gen-enum-defs EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_typedefs.h.inc -gen-typedef-decls -typedefs-dialect=xls EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
mlir_tablegen(xls_ops_typedefs.cc.inc -gen-typedef-defs -typedefs-dialect=xls EXTRA_INCLUDES ${dialect_dir} ${xls_root_dir_abs})
add_public_tablegen_target(MLIRXlsIncGen)
add_dependencies(mlir-headers MLIRXlsIncGen)
add_dependencies(dynamatic-headers MLIRXlsIncGen)

# Location of XLS interfaces description:
set(interface_file ${xls_root_dir_abs}/xls/contrib/mlir/IR/interfaces.td)

# Run tablegen for XLS interfaces:
set(LLVM_TARGET_DEFINITIONS ${interface_file})
mlir_tablegen(interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(interfaces.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRXlsInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRXlsInterfaceIncGen)
add_dependencies(dynamatic-headers MLIRXlsInterfaceIncGen)

#
# Lib
#

set_property(GLOBAL APPEND PROPERTY DYNAMATIC_DIALECT_LIBS XlsMlir)

add_mlir_library(XlsMlir
  xls_ops.cc
  assembly_format.cc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
  MLIRAffineDialect

  DEPENDS
  MLIRXlsIncGen
  MLIRXlsInterfaceIncGen
  dynamatic-headers
)

target_include_directories(XlsMlir PUBLIC ${CMAKE_BINARY_DIR}/xls_integration/)
target_include_directories(XlsMlir PUBLIC ${CMAKE_SOURCE_DIR}/xls_integration/)
