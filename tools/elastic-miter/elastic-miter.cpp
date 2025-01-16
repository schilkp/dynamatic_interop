//===- circt-lec.cpp - The circt-lec driver ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-lec' tool, which interfaces with a logical
/// engine to allow its user to check whether two input circuit descriptions
/// are equivalent, and when not provides a counterexample as for why.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"


#include <iostream>
#include "dynamatic/InitAllDialects.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"


namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic;


//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string> firstModuleName(
    "c1", cl::Required,
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> secondModuleName(
    "c2", cl::Required,
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));


//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid
// conflict.
static FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src,
                                          StringAttr name) {

  SymbolTable destTable(dest), srcTable(src);
  StringAttr newName = {};
  for (auto &op : src.getOps()) {
    if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
      auto oldSymbol = symbol.getNameAttr();
      // auto result = renameToUnique(&srcTable, op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()));
      // if (failed(result))
      //   return src->emitError() << "failed to rename symbol " << oldSymbol;

      if (oldSymbol == name) {
        assert(!newName && "symbol must be unique");
        // newName = *result;
      }
    }
  }

  // if (!newName)
  //   return src->emitError()
  //          << "module " << name << " was not found in the second module";

  dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                         src.getBody()->getOperations());
  return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
static FailureOr<OwningOpRef<ModuleOp>>
parseAndMergeModules(MLIRContext &context) {

  llvm::StringRef filename("../tools/lec/rewrites/a_lhs.mlir");
  llvm::StringRef filename2("../tools/lec/rewrites/a_rhs.mlir");
  OwningOpRef<ModuleOp> lhs_module = parseSourceFile<ModuleOp>(filename, &context);

  OpBuilder builder(&context);

  // Iterate over the module's operations.
  for (handshake::FuncOp funcOp : lhs_module->getOps<handshake::FuncOp>()) {
    for (handshake::EndOp op : funcOp.getOps<handshake::EndOp>()) {
      builder.setInsertionPoint(op);
      Operation::operand_range operands = op->getOperands();
      for(int i = 0; i < operands.size(); i++) {
        handshake::BufferOp newBufferOp = builder.create<handshake::BufferOp>(op.getLoc(), operands[i]);
        Value bufferRes = newBufferOp.getResult();
        op.setOperand(i, bufferRes);
      }


      // std::cout << op.getName().getStringRef().str() << std::endl;
    }
  }
  std::cout << "Hello there " << __FILE__ << ":"<< __LINE__ << std::endl;

  OpPrintingFlags pf;
  lhs_module->print(llvm::outs(), pf);
  

  if (!lhs_module) {
    return failure();
  }

  auto moduleOpt = parseSourceFile<ModuleOp>(filename2, &context);
  if (!moduleOpt)
    return failure();
  auto result = mergeModules(lhs_module.get(), moduleOpt.get(),
                              StringAttr::get(&context, secondModuleName));
  if (failed(result)) {
    return failure();
  }
  return lhs_module;
}

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeLEC(MLIRContext &context) {
  
  auto parsedModule = parseAndMergeModules(context);
  if (failed(parsedModule))
    return failure();

  std::cout << "Hello there" << __LINE__ << std::endl;

  OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  OpPrintingFlags printingFlags;
  module->print(outputFile.value()->os(), printingFlags);
  outputFile.value()->keep();
  return success();
}

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {

  std::cout << "Hello there " << __FILE__ << ":"<< __LINE__ << std::endl;
  llvm::InitLLVM y(argc, argv);
  std::cout << "Hello there" << __LINE__ << std::endl;

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  std::cout << "Hello there" << __LINE__ << std::endl;

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  // registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
  //                 mlir::arith::ArithDialect, mlir::BuiltinDialect>();
  // mlir::func::registerInlinerExtension(registry);
  // mlir::registerBuiltinDialectTranslation(registry);
  // mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  std::cout << "Hello there" << __LINE__ << std::endl;
  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);
  std::cout << "Hello there" << __LINE__ << std::endl;

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeLEC(context)));
}