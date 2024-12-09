//===- FtdImplementation.h --- FTD conversion support -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_SUPPORT_H
#define DYNAMATIC_SUPPORT_FTD_SUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Analysis/GSAAnalysis.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Given a set of values defining the same value in different blocks of a
/// CFG, modify the SSA representation to connect the values through some
/// merges. Replace the input uses with the correct value coming from the
/// network.
LogicalResult createPhiNetwork(Region &funcRegion,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value> &vals,
                               SmallVector<OpOperand *> &toSubstitue);

/// Add some regen multiplexers between an opearation and one of its operands
void addRegenOperandConsumer(ConversionPatternRewriter &rewriter,
                             dynamatic::handshake::FuncOp &funcOp,
                             Operation *consumerOp, Value operand);

/// Add suppression mechanism to all the inputs and outputs of a producer
void addSuppOperandConsumer(ConversionPatternRewriter &rewriter,
                            handshake::FuncOp &funcOp, Operation *consumerOp,
                            Value operand);

/// When the consumer is in a loop while the producer is not, the value must
/// be regenerated as many times as needed. This function is in charge of
/// adding some merges to the network, to that this can be done. The new
/// merge is moved inside of the loop, and it works like a reassignment
/// (cfr. FPGA'22, Section V.C).
void addRegen(handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter);

/// Given each pairs of producers and consumers within the circuit, the
/// producer might create a token which is never used by the corresponding
/// consumer, because of the control decisions. In this scenario, the token
/// must be suprressed. This function inserts a `SUPPRESS` block whenever it
/// is necessary, according to FPGA'22 (IV.C and V)
void addSupp(handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter);

/// Starting from the information collected by the gsa analysis pass,
/// instantiate some merge operations at the beginning of each block which
/// work as explicit phi functions.
LogicalResult addGsaGates(Region &region, ConversionPatternRewriter &rewriter,
                          const gsa::GSAAnalysis &gsa, Backedge startValue,
                          bool removeTerminators = true);

/// Use the GSA analysis to replace each non-init merge in the IR with a
/// multiplexer.
LogicalResult replaceMergeToGSA(handshake::FuncOp funcOp,
                                ConversionPatternRewriter &rewriter);

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_SUPPORT_H
