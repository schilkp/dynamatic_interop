add_dynamatic_library(DynamaticHandshakeToXls
  HandshakeToXls.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupportRTL
  DynamaticHandshake
  XlsMlir
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
