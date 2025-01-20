//===- elastic-miter.cpp - The elastic-miter driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the elastic-miter tool, it creates an elastic miter
// circuit, which can later be used to formally verify equivalence of two
// handshake circuits.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include "dynamatic/InitAllDialects.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"


namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic;

// TODO remove this
OpPrintingFlags printingFlags;

// Command line TODO actually use arguments
static cl::OptionCategory mainCategory("elastic-miter Options");


// TODO documentation
static LogicalResult prefixOperation(Operation &op, std::string prefix) {
  auto nameAttr = op.getAttrOfType<StringAttr>("handshake.name");
  if (!nameAttr)
      return failure();

  std::string new_name = prefix + nameAttr.getValue().str();

  op.setAttr("handshake.name", StringAttr::get(op.getContext(), new_name));

  return success();
}

// create the elatic miter module, given two circuits
static FailureOr<OwningOpRef<ModuleOp>>
createElasticMiter(MLIRContext &context) {

  // TODO use arguments
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

  OwningOpRef<ModuleOp> miterModule = ModuleOp::create(builder.getUnknownLoc());

  int output_size = lhs_funcOp.getResultTypes().size();

  handshake::ChannelType i1_ChannelType = handshake::ChannelType::get(builder.getI1Type());
  llvm::SmallVector<mlir::Type> outputTypes(output_size, i1_ChannelType);  // replace all output with i1 which represent the result of the comparison
  mlir::FunctionType funcType = builder.getFunctionType(lhs_funcOp.getArgumentTypes(), outputTypes);

  // Create the miter function
  handshake::FuncOp new_funcOp = builder.create<handshake::FuncOp>(builder.getUnknownLoc(), "elastic_miter", funcType, arrayRef2);

  // Create a block and put it in the funcOp, this is borrowed from func::funcOp.addEntryBlock()
  Block *entry = new Block();
  new_funcOp.push_back(entry);

  builder.setInsertionPointToStart(entry);

  // FIXME: Allow for passing in locations for these arguments instead of using
  // the operations location.
  ArrayRef<Type> inputTypes = new_funcOp.getArgumentTypes();
  SmallVector<Location> locations(inputTypes.size(),
                                  new_funcOp.getOperation()->getLoc());
  entry->addArguments(inputTypes, locations);


  // Add the function to the module
  miterModule->push_back(new_funcOp);


  /* TODO:
  3: Add auxillary operations
  4: Connect those operations up
  */


  // Rename the operations in the existing lhs module TODO check for success
  for (Operation &op : lhs_funcOp.getOps()) {
    prefixOperation(op, "lhs_");
  }
  // Rename the operations in the existing rhs module TODO check for success
  for (Operation &op : rhs_funcOp.getOps()) {
    prefixOperation(op, "rhs_");
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

  miterModule->print(llvm::outs(), printingFlags);


  if (!miterModule) {
    return failure();
  }
  return miterModule;
}


int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  exit(failed(createElasticMiter(context)));
}