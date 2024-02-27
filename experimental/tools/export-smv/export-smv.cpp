//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental tool that exports nuXmv-compatible format from a netlist-level
// IR expressed in the handshake dialect
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstddef>
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

// for parametrized units, we encode the parameter in the type of the unit and
// later generate a unit for each parametrization used by the model
LogicalResult getNodeType(Operation *op, mlir::raw_indented_ostream &os) {

  std::string type =
      llvm::TypeSwitch<Operation *, std::string>(op)
          .Case<handshake::OEHBOp>([&](handshake::OEHBOp oehbOp) {
            return "oehb_" + std::to_string(oehbOp.getSlots()) + "s_" +
                   std::to_string(op->getNumOperands()) + "_" +
                   std::to_string(op->getNumResults());
          })
          .Case<handshake::TEHBOp>([&](handshake::TEHBOp tehbOp) {
            return "tehb_" + std::to_string(tehbOp.getSlots()) + "s_" +
                   std::to_string(op->getNumOperands()) + "_" +
                   std::to_string(op->getNumResults());
          })
          .Default([&](auto) {
            return "unknownop" + std::to_string(op->getNumOperands()) + "_" +
                   std::to_string(op->getNumResults());
          });

  os << type;
  return success();
}

// prints the declaration for the unit
LogicalResult printUnit(Operation *op, mlir::raw_indented_ostream &os) {
  StringRef unitName = getUniqueName(op);

  os << "VAR " << unitName << " : ";
  if (failed(getNodeType(op, os))) {
    return failure();
  }
  os << " (";

  for (auto [idx, arg] : llvm::enumerate(op->getOperands())) {
    os << unitName << "_dataIn" << idx << ", " << unitName << "_pValid" << idx;
    if (idx < op->getNumOperands() - 1) {
      os << ", ";
    }
  }

  if (op->getNumOperands() > 0 && op->getNumResults() > 0)
    os << ", ";

  for (auto [idx, arg] : llvm::enumerate(op->getResults())) {
    os << unitName << "_nReady" << idx;
    if (idx < op->getNumResults() - 1) {
      os << ", ";
    }
  }
  os << ");\n";
  return success();
}

static size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  llvm_unreachable("value should exist in range");
}

// prints the declaration for the channels, one line per each data, ready and
// valid signal
LogicalResult printEdge(OpOperand &oprd, mlir::raw_indented_ostream &os) {
  Value val = oprd.get();
  Operation *src = val.getDefiningOp();
  Operation *dst = oprd.getOwner();

  // Locate value in source results and destination operands
  const size_t resIdx = findIndexInRange(src->getResults(), val);
  const size_t argIdx = findIndexInRange(dst->getOperands(), val);

  os << "DEFINE " << getUniqueName(src) << "_nReady" << resIdx << " = "
     << getUniqueName(dst) << "_ready" << argIdx << ";\n";

  os << "DEFINE " << getUniqueName(dst) << "_pValid" << argIdx << " = "
     << getUniqueName(src) << "_valid" << resIdx << ";\n";

  os << "DEFINE " << getUniqueName(dst) << "_dataIn" << argIdx << " = "
     << getUniqueName(src) << "_dataOut" << resIdx << ";\n";

  return success();
}

LogicalResult writeSmv(mlir::ModuleOp mod) {

  mlir::raw_indented_ostream stdOs(llvm::outs());
  auto funcs = mod.getOps<handshake::FuncOp>();
  if (++funcs.begin() != funcs.end()) {
    mod->emitOpError()
        << "we currently only support one handshake function per module";
    return failure();
  }
  NameAnalysis nameAnalysis = NameAnalysis(mod);
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  nameAnalysis.nameAllUnnamedOps();

  handshake::FuncOp funcOp = *funcs.begin();

  stdOs << "MODULE main\n";
  stdOs << "\n-- units\n";

  for (auto &op : funcOp.getOps()) {
    if (failed(printUnit(&op, stdOs)))
      return failure();
  }

  stdOs << "\n\n-- channels\n";

  for (auto &op : funcOp.getOps()) {
    for (OpResult res : op.getResults()) {
      for (OpOperand &val : res.getUses()) {
        if (failed(printEdge(val, stdOs)))
          return failure();
      }
    }
  }

  stdOs << "\n\n-- formal properties\n";

  return success();
}

int main(int argc, char **argv) {

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a SMV model corresponding to the module"
      "The tool only supports exporting the graph of a single Handshake "
      "function at the moment, and will fail"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
                      handshake::HandshakeDialect, math::MathDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> mod(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!mod)
    return 1;
  return failed(writeSmv(*mod));
}
