//===- FtdCfToHandhsake.h - Convert func/cf to handhsake dialect -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --ftd-lower-cf-to-handshake conversion pass
// along with a helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
#define DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Conversion/FtdMemoryInterface.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Convert a func-level function into an handshake-level function. A custom
/// behavior is defined so that the functionalities of the `fast delivery token`
/// methodology can be implemented.
class FtdLowerFuncToHandshake : public LowerFuncToHandshake {
public:
  // Use the same constructors from the base class
  FtdLowerFuncToHandshake(ControlDependenceAnalysis &cda, gsa::GSAAnalysis &gsa,
                          NameAnalysis &namer, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, ctx, benefit), cdAnalysis(cda),
        gsaAnalysis(gsa) {};

  FtdLowerFuncToHandshake(ControlDependenceAnalysis &cda, gsa::GSAAnalysis &gsa,
                          NameAnalysis &namer,
                          const TypeConverter &typeConverter, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, typeConverter, ctx, benefit),
        cdAnalysis(cda), gsaAnalysis(gsa) {};

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Store the control dependency analysis over the input function
  ControlDependenceAnalysis cdAnalysis;

  /// Store the GSA analysis over the input function
  gsa::GSAAnalysis gsaAnalysis;

  LogicalResult ftdVerifyAndCreateMemInterfaces(
      handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
      MemInterfacesInfo &memInfo, const BlockIndexing &bi) const;

  void exportGsaGatesInfo(handshake::FuncOp funcOp) const;

  /// Given a list of operations, return the list of memory dependencies for
  /// each block. This allows to build the group graph, which allows to
  /// determine the dependencies between memory access inside basic blocks.
  // Two types of hazards between the predecessors of one LSQ node:
  // (1) WAW between 2 Store operations,
  // (2) RAW and WAR between Load and Store operations
  void identifyMemoryDependencies(const SmallVector<Operation *> &operations,
                                  SmallVector<ProdConsMemDep> &allMemDeps,
                                  const mlir::CFGLoopInfo &li,
                                  const BlockIndexing &bi) const;

  /// Convers arith-level constants to handshake-level constants. Constants are
  /// triggered by the start value of the corresponding function. The FTD
  /// algorithm is then in charge of connecting the constants to the rest of the
  /// network, in order for them to be re-generated
  LogicalResult convertConstants(ConversionPatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp) const;

  /// Converts undefined operations (LLVM::UndefOp) with a default "0"
  /// constant triggered by the start signal of the corresponding function.
  LogicalResult convertUndefinedValues(ConversionPatternRewriter &rewriter,
                                       handshake::FuncOp &funcOp) const;

  /// When the consumer is in a loop while the producer is not, the value must
  /// be regenerated as many times as needed. This function is in charge of
  /// adding some merges to the network, to that this can be done. The new
  /// merge is moved inside of the loop, and it works like a reassignment
  /// (cfr. FPGA'22, Section V.C).
  LogicalResult addRegen(ConversionPatternRewriter &rewriter,
                         handshake::FuncOp &funcOp) const;

  /// Given each pairs of producers and consumers within the circuit, the
  /// producer might create a token which is never used by the corresponding
  /// consumer, because of the control decisions. In this scenario, the token
  /// must be suprressed. This function inserts a `SUPPRESS` block whenever it
  /// is necessary, according to FPGA'22 (IV.C and V)
  LogicalResult addSupp(ConversionPatternRewriter &rewriter,
                        handshake::FuncOp &funcOp,
                        ControlDependenceAnalysis::BlockControlDepsMap &cda,
                        const ftd::BlockIndexing &bi) const;
};
#define GEN_PASS_DECL_FTDCFTOHANDSHAKE
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake();

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
