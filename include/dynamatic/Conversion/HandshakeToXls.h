//===- HandshakeToXls.h - Convert Handshake to Xls --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-handshake-to-xls conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_HANDSHAKETOXLS_H
#define DYNAMATIC_CONVERSION_HANDSHAKETOXLS_H

#include "dynamatic/Support/DynamaticPass.h"

#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {
/// Forward declare the Xls dialect which the pass depends on.
class XlsDialect;
} // namespace mlir::xls

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKETOXLS
#define GEN_PASS_DEF_HANDSHAKETOXLS
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeToXlsPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETOXLS_H
