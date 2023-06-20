//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

ChannelConstraints
buffer::BufferPlacementStrategy::getChannelConstraints(channel *ch) {
  ChannelConstraints constraints;
  // set the channel constraints according to the global constraints
  constraints.minSlots = this->minSlots;
  constraints.maxSlots = this->maxSlots;
  constraints.transparentAllowed = true;
  constraints.nonTransparentAllowed = true;
  constraints.bufferizable = true;
  return constraints;
}

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx,
                                   BufferPlacementStrategy &strategy,
                                   bool firstMG, std::string stdLevelInfo) {

  std::vector<Operation *> visitedOpList;
  std::vector<unit *> unitList;

  // DFS build the dataflowCircuit from the handshake level
  for (auto &op : funcOp.getOps())
    if (isEntryOp(&op, visitedOpList))
      dfsHandshakeGraph(&op, unitList, visitedOpList);

  // create CFDFC circuits
  std::vector<dataFlowCircuit *> dataFlowCircuitList;
  // speficy by a flag, whether read the bb file from std level
  if (stdLevelInfo != "") {
    std::map<archBB *, int> archs;
    std::map<int, int> bbs;

    readSimulateFile(stdLevelInfo, archs, bbs);
    int execNum = buffer::extractCFDFCircuit(archs, bbs);
    while (execNum > 0) {
      // write the execution frequency to the dataflowCircuit
      auto circuit = createCFDFCircuit(unitList, archs, bbs);
      circuit->targetCP = 4.0;
      circuit->execN = execNum;
      dataFlowCircuitList.push_back(circuit);
      if (firstMG)
        break;
      execNum = buffer::extractCFDFCircuit(archs, bbs);
    }
  }

  for (auto dfc : dataFlowCircuitList) {
    // store the output results
    std::map<std::string, GRBVar> resVars;
    dfc->createMILPModel(strategy, resVars);
  }

  // Insert buffers in the remaining dataflowCircuit

  return success();
}

namespace {
/// Simple driver for prepare for legacy pass.
class customBufferPlaceStrategy : public BufferPlacementStrategy {
public:
  ChannelConstraints getChannelConstraints(channel *ch) override {
    ChannelConstraints constraints;
    // set the channel constraints according to the global constraints
    constraints.minSlots = this->minSlots;
    constraints.maxSlots = this->maxSlots;
    constraints.transparentAllowed = true;
    constraints.nonTransparentAllowed = true;
    constraints.bufferizable = true;

    if (isa<handshake::MergeOp, handshake::MuxOp>(ch->unitSrc->op)) {
      constraints.minSlots = 1;
      constraints.transparentAllowed = true;
      constraints.nonTransparentAllowed = true;
    }

    if (isa<handshake::ControlMergeOp>(ch->unitSrc->op) ||
        isa<handshake::ControlMergeOp>(ch->unitDst->op) ) {
      constraints.bufferizable = false;
    }
    return constraints;
  }
};

struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  PlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    customBufferPlaceStrategy strategy;
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), strategy, firstMG,
                               stdLevelInfo)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG,
                                           std::string stdLevelInfo) {
  return std::make_unique<PlaceBuffersPass>(firstMG, stdLevelInfo);
}