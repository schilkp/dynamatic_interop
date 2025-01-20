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

// TODO remove this
OpPrintingFlags printingFlags;


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

// TODO rewrite this
static LogicalResult prefixOperation(Operation &op, std::string prefix) {
  NamedAttribute attr = op.getAttrDictionary().getNamed("handshake.name").value(); // TODO check if it exists, also WTF is this?
  Attribute value = attr.getValue();
  auto name = value.dyn_cast<StringAttr>();
  if(!name) return failure();
  std::string old_name = name.getValue().str();
  std::string new_name = prefix;
  new_name.append(old_name);
  StringAttr newNameAttr = StringAttr::get(op.getContext(), new_name);
  op.setAttr("handshake.name", newNameAttr);
  return success();

}

// Parse one or two MLIR modules and merge it into a single module.
static FailureOr<OwningOpRef<ModuleOp>>
parseAndMergeModules(MLIRContext &context) {

  llvm::StringRef filename("../tools/elastic-miter/rewrites/a_lhs.mlir");
  llvm::StringRef filename2("../tools/elastic-miter/rewrites/a_rhs.mlir");
  OwningOpRef<ModuleOp> lhs_module = parseSourceFile<ModuleOp>(filename, &context);
  OwningOpRef<ModuleOp> rhs_module = parseSourceFile<ModuleOp>(filename2, &context);

  // The module can only have one function so we just take the first element TODO add a check for this
  handshake::FuncOp lhs_funcOp = *lhs_module->getOps<handshake::FuncOp>().begin();
  handshake::FuncOp rhs_funcOp = *rhs_module->getOps<handshake::FuncOp>().begin();


  OpBuilder builder(&context);

  auto stringAttr_D = builder.getStringAttr("D");
  auto stringAttr_C = builder.getStringAttr("C");
  auto stringAttr_T = builder.getStringAttr("EQ_T");
  auto stringAttr_F = builder.getStringAttr("EQ_F");


  ArrayRef<Attribute> inputAttr_arrayRef({stringAttr_D, stringAttr_C});
  ArrayRef<Attribute> resAttr_arrayRef({stringAttr_T, stringAttr_F});

  auto arg_named_attr = builder.getNamedAttr("argNames", builder.getArrayAttr(inputAttr_arrayRef));
  auto res_named_attr = builder.getNamedAttr("resNames", builder.getArrayAttr(resAttr_arrayRef));

  ArrayRef<NamedAttribute> arrayRef2({arg_named_attr, res_named_attr});

  ModuleOp new_module = ModuleOp::create(builder.getUnknownLoc());

  int output_size = lhs_funcOp.getResultTypes().size();
  std::cout << "We havbe " << output_size << " outputs" << std::endl;
  handshake::ChannelType i1_ChannelType = handshake::ChannelType::get(builder.getI1Type());
  llvm::SmallVector<mlir::Type> outputTypes(output_size, i1_ChannelType);  // replace all output with i1 which represent the result of the comparison
  mlir::FunctionType funcType = builder.getFunctionType(lhs_funcOp.getArgumentTypes(), outputTypes);

  // Create the function
  handshake::FuncOp new_funcOp = builder.create<handshake::FuncOp>(builder.getUnknownLoc(), "elastic_miter", funcType, arrayRef2);

  // Create a block and put it in the funcOp, this is borrowed from func::funcOp.addEntryBlock()
  Block *entry = new Block();
  new_funcOp.push_back(entry);

  builder.setInsertionPointToStart(entry);

  // FIXME: Allow for passing in locations for these arguments instead of using
  // the operations location.
  ArrayRef<Type> inputTypes = new_funcOp.getArgumentTypes();
  std::cout << __FILE__ << ":" << __LINE__ << "  " << inputTypes.size() << std::endl;
  SmallVector<Location> locations(inputTypes.size(),
                                  new_funcOp.getOperation()->getLoc());
  entry->addArguments(inputTypes, locations);

  // Add the function to the module
  new_module.push_back(new_funcOp);

  // for (const auto &attr : lhs_funcOp->getAttrs()) {
  //     llvm::outs() << "Attribute name: " << attr.getName()
  //                   << ", Attribute value: " << attr.getValue() << "\n";
  // }

  new_module->print(llvm::outs(), printingFlags);


  /* TODO:
  3: Add auxillary operations
  4: Connect those operations up
  */


  // Rename the operations in the existing lhs module
  for (Operation &op : lhs_funcOp.getOps()) {
    prefixOperation(op, "lhs_");
  }
  // Rename the operations in the existing rhs module
  for (Operation &op : rhs_funcOp.getOps()) {
    prefixOperation(op, "rhs_");
  }


  for (unsigned i = 0; i < lhs_funcOp.getNumArguments(); ++i) {
    mlir::BlockArgument blockArg = lhs_funcOp.getArgument(i);

    // llvm::outs() << "  Block Argument " << i << "\n";
    // llvm::outs() << "    Index: " << blockArg.getArgNumber() << "\n";
    // llvm::outs() << "    Type: " << blockArg.getType() << "\n";
  }

  builder.setInsertionPointToStart(entry);

  handshake::ForkOp forkOp = nullptr;
  for (unsigned i = 0; i < lhs_funcOp.getNumArguments(); ++i) {
    BlockArgument blockArg  = lhs_funcOp.getArgument(i);
    BlockArgument blockArg2 = rhs_funcOp.getArgument(i);
    BlockArgument blockArg3 = new_funcOp.getArgument(i);


    forkOp = builder.create<handshake::ForkOp>(new_funcOp.getLoc(), blockArg3.getType(), blockArg3);


    // Use the newly created fork's output instead of the origial argument in the lhs_funcOp's operations
    for (Operation *op : llvm::make_early_inc_range(blockArg.getUsers())) {
      op->replaceUsesOfWith(blockArg, forkOp.getResults()[0]);
    }
    // Use the newly created fork's output instead of the origial argument in the rhs_funcOp's operations
    for (Operation *op : llvm::make_early_inc_range(blockArg2.getUsers())) {
      op->replaceUsesOfWith(blockArg2, forkOp.getResults()[0]);
    }
  }

  // Get lhs and rhs EndOp TODO without loop
  handshake::EndOp lhs_endOp;
  for (handshake::EndOp endOp : llvm::make_early_inc_range(lhs_funcOp.getOps<handshake::EndOp>())) {
    lhs_endOp = endOp;
  }
  handshake::EndOp rhs_endOp;
  for (handshake::EndOp endOp : llvm::make_early_inc_range(rhs_funcOp.getOps<handshake::EndOp>())) {
    rhs_endOp = endOp;
  }


  // Create comparison logic
  llvm::SmallVector<mlir::Value> eq_results;
  for (unsigned i = 0; i < lhs_endOp.getOperands().size(); ++i) {
    Value lhs_result = lhs_endOp.getOperand(i);
    Value rhs_result = rhs_endOp.getOperand(i);

    handshake::CmpIOp compOp = builder.create<handshake::CmpIOp>(builder.getUnknownLoc(), handshake::CmpIPredicate::eq, lhs_result, rhs_result);
    eq_results.push_back(compOp.getResult());
  }

  handshake::EndOp newEndOp = builder.create<handshake::EndOp>(builder.getUnknownLoc(), eq_results);

  // Delete old end operation, we can only have one end operation in a function
  rhs_endOp.erase();
  lhs_endOp.erase();


  // Move operations from lhs to new
  Operation *previousOp = forkOp;
  for(Operation &op : llvm::make_early_inc_range(lhs_funcOp.getOps())) {
    op.moveAfter(previousOp);
    previousOp = &op;
  }

  // Move operations from lhs to new
  for(Operation &op : llvm::make_early_inc_range(rhs_funcOp.getOps())) {
    op.moveAfter(previousOp);
    previousOp = &op;
  }

  new_module->print(llvm::outs(), printingFlags);


  if (!lhs_module) {
    return failure();
  }

  if (!rhs_module)
    return failure();
  // auto result = mergeModules(lhs_module.get(), rhs_module.get(),
  //                             StringAttr::get(&context, secondModuleName));
  // if (failed(result)) {
  //   return failure();
  // }
  return lhs_module;
}

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeLEC(MLIRContext &context) {

  auto parsedModule = parseAndMergeModules(context);
  if (failed(parsedModule))
    return failure();

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

  // OpPrintingFlags printingFlags;
  // module->print(outputFile.value()->os(), printingFlags);
  outputFile.value()->keep();
  return success();
}

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Register the supported dynamatic dialects and create a context to work with.
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeLEC(context)));
}