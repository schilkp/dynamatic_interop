//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the fast token delivery methodology
// https://ieeexplore.ieee.org/abstract/document/10035134, together with the
// straight LSQ allocation https://dl.acm.org/doi/abs/10.1145/3543622.3573050.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<ControlDependenceAnalysis>(),
        getAnalysis<gsa::GSAAnalysis>(), getAnalysis<NameAnalysis>(), converter,
        ctx);

    patterns.add<ConvertCalls,
                 ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
                 ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
                 OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
                 OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
                 OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
                 OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
                 OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
                 OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
                 OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
                 OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
                 OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
                 OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
                 OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
                 OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
                 OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
                 OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
                 OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
                 OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
                 OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
                 OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
                 OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
                 OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
                 OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
                 OneToOneConversion<arith::TruncFOp, handshake::TruncFOp>,
                 OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>,
                 OneToOneConversion<arith::SIToFPOp, handshake::SIToFPOp>,
                 OneToOneConversion<arith::FPToSIOp, handshake::FPToSIOp>,
                 OneToOneConversion<arith::ExtFOp, handshake::ExtFOp>,
                 OneToOneConversion<math::AbsFOp, handshake::AbsFOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

void ftd::FtdLowerFuncToHandshake::analyzeLoop(handshake::FuncOp funcOp) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  std::ofstream ofs;

  ofs.open("ftdscripting/loopinfo.txt", std::ofstream::out);
  std::string loopDescription;
  llvm::raw_string_ostream loopDescriptionStream(loopDescription);

  auto muxes = funcOp.getBody().getOps<handshake::MuxOp>();
  for (auto phi : muxes) {
    if (!loopInfo.getLoopFor(phi->getBlock()))
      continue;
    ofs << namer.getName(phi).str();
    if (llvm::isa<handshake::MergeOp>(phi->getOperand(0).getDefiningOp()))
      ofs << " (MU)\n";
    else
      ofs << " (GAMMA)\n";
    loopInfo.getLoopFor(phi->getBlock())
        ->print(loopDescriptionStream, false, false, 0);
    ofs << loopDescription << "\n";
    loopDescription = "";
  }

  ofs.close();
}

// --- Helper functions ---

/// When init merges for MU functions are instantiated, the function does not
/// have a start signal yet. Once that the start signal is created, it needs
/// to be connected to all the init merges.
static void connectInitMerges(ConversionPatternRewriter &rewriter,
                              handshake::FuncOp funcOp) {
  auto startValue = (Value)funcOp.getArguments().back();
  auto cstType = rewriter.getIntegerType(1);
  auto cstAttr = IntegerAttr::get(cstType, 0);
  for (auto initMerge : funcOp.getBody().getOps<handshake::MergeOp>()) {
    rewriter.setInsertionPointToStart(initMerge->getBlock());
    auto constOp = rewriter.create<handshake::ConstantOp>(initMerge->getLoc(),
                                                          cstAttr, startValue);
    constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
    initMerge->setOperand(0, constOp.getResult());
  }
}

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, crate a group graph.
static void constructGroupsGraph(SmallVector<Operation *> &operations,
                                 SmallVector<ProdConsMemDep> &allMemDeps,
                                 DenseSet<Group *> &groups) {

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : operations) {
    if (isHandhsakeLSQOperation(op)) {
      Block *b = op->getBlock();
      auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
      if (it == groups.end()) {
        Group *g = new Group(b);
        groups.insert(g);
      }
    }
  }

  // If there exist a relationship O_i -> O_j, with O_i being in basic BB_i
  // (thus group i) and O_j being in BB_j (thus in group j), add G_i to the
  // predecessors of G_j, G_j to the successors of G_i
  for (ProdConsMemDep memDep : allMemDeps) {
    // Find the group related to the producer
    Group *producerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.prodBb;
        });

    // Find the group related to the consumer
    Group *consumerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.consBb;
        });

    // create edges to link the groups
    producerGroup->succs.insert(consumerGroup);
    consumerGroup->preds.insert(producerGroup);
  }

  // TODO
  // for (Group *g : groups) {
  //   if (!g->preds.size()) {
  //     g->preds.insert(g);
  //     g->succs.insert(g);
  //   }
  // }
}

/// Minimizes the connections between groups based on dominance info. Let's
/// consider the graph
///
/// B -> C -> D
/// |         ^
/// |---------|
///
/// having B, C and D as groups, B being predecessor of both C and D, C of D.
/// Since C has to wait for B to be done, and D has to wait for C to be done,
/// there is no point in D waiting for C to be done. For this reason, the
/// graph can be simplified, saving and edge:
///
/// B -> C -> D
static void minimizeGroupsConnections(DenseSet<Group *> &groupsGraph,
                                      const BlockIndexing &bi) {

  // Get the dominance info for the region
  DominanceInfo domInfo;

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the
  // whole group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<Group *> predsToRemove;
    for (auto &bp : group->preds) {

      for (auto &sp : group->preds) {

        // if we are considering the same elements, ignore them
        if (sp->bb == bp->bb || bi.greaterIndex(sp->bb, bp->bb))
          continue;

        // Add the small predecessors to the list of elements to remove in
        // case the big predecessor has the small one among its
        // predecessors, and the big precessor's BB properly dominates the
        // BB of the group currently under analysis
        if ((bp->preds.find(sp) != bp->preds.end()) &&
            domInfo.properlyDominates(bp->bb, group->bb)) {
          predsToRemove.insert(sp);
        }
      }
    }

    for (auto *pred : predsToRemove) {
      group->preds.erase(pred);
      pred->succs.erase(group);
    }
  }
}

/// Allocate some joins in front of each lazy fork, so that the number of
/// inputs for each of them is exactly one. The current inputs of the lazy
/// forks become inputs for the joins.
static LogicalResult joinInsertion(OpBuilder &builder,
                                   DenseSet<Group *> &groups,
                                   DenseMap<Block *, Operation *> &forksGraph) {
  // For each group
  for (Group *group : groups) {
    // Get the corresponding fork and operands
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    // If the number of inputs is higher than one
    if (operands.size() > 1) {

      // Join all the inputs, and set the output of this new element as input
      // of the lazy fork
      builder.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          builder.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::addSupp(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    ControlDependenceAnalysis::BlockControlDepsMap &cdaDeps,
    const BlockIndexing &bi) const {

  // A set of relationships between producer and consumer needs to be covered.
  // To do that, we consider each possible operation in the circuit as
  // producer. However, some operations are added throughout the execution of
  // the function, and those are possibly to be analyzed as well. This vector
  // maintains the list of operations to be analyzed.
  // [TBD] We could consider slightly changing the analysis to avoid needing
  // to rerun the analysis on the operations inserted throughout the
  // execution, but this is future work...
  std::vector<Operation *> producersToCover;

  // Add all the operations in the IR to the above vector
  for (Block &producerBlock : funcOp.getBlocks()) {
    for (Operation &producerOp : producerBlock.getOperations())
      producersToCover.push_back(&producerOp);
  }

  // Loop through the vector until all the elements have been analyzed
  unsigned producerIndex = 0;
  while (producerIndex < producersToCover.size()) {
    Operation *producerOp = producersToCover.at(producerIndex++);
    if (failed(addSuppToProducer(rewriter, funcOp, producerOp, bi,
                                 producersToCover, cdaDeps)))
      return failure();
  }

  return success();
}

LogicalResult
ftd::FtdLowerFuncToHandshake::addRegen(ConversionPatternRewriter &rewriter,
                                       handshake::FuncOp &funcOp) const {

  // For each producer/consumer relationship
  for (Operation &consumerOp : funcOp.getOps()) {
    if (failed(addRegenToConsumer(rewriter, funcOp, &consumerOp)))
      return failure();
  }

  // Once that all the multiplexers have been added, it is necessary to modify
  // the type of the result, for it to be a channel type (that could not be
  // done before)
  for (Operation *mux : funcOp.getOps<handshake::MuxOp>())
    mux->getResult(0).setType(channelifyType(mux->getResult(0).getType()));

  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::convertUndefinedValues(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp) const {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each undefined value
  auto undefinedValues = funcOp.getBody().getOps<LLVM::UndefOp>();

  for (auto undefOp : llvm::make_early_inc_range(undefinedValues)) {
    // Create an attribute of the appropriate type for the constant
    auto resType = undefOp.getRes().getType();
    TypedAttr cstAttr;
    if (isa<IndexType>(resType)) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    } else if (isa<IntegerType>(resType)) {
      cstAttr = rewriter.getIntegerAttr(resType, 0);
    } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
      cstAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    }

    // Create a constant with a default value and replace the undefined value
    rewriter.setInsertionPoint(undefOp);
    auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(),
                                                        cstAttr, startValue);
    cstOp->setDialectAttrs(undefOp->getAttrDictionary());
    undefOp.getResult().replaceAllUsesWith(cstOp.getResult());
    namer.replaceOp(cstOp, cstOp);
    rewriter.replaceOp(undefOp, cstOp.getResult());
  }

  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::convertConstants(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp) const {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each constant
  auto constants = funcOp.getBody().getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : llvm::make_early_inc_range(constants)) {

    rewriter.setInsertionPoint(cstOp);

    // Convert the constant to the handshake equivalent, using the start value
    // as control signal
    TypedAttr cstAttr = cstOp.getValue();

    if (isa<IndexType>(cstAttr.getType())) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = IntegerAttr::get(
          intType, cast<IntegerAttr>(cstAttr).getValue().trunc(32));
    }

    auto newCstOp = rewriter.create<handshake::ConstantOp>(cstOp.getLoc(),
                                                           cstAttr, startValue);

    newCstOp->setDialectAttrs(cstOp->getDialectAttrs());

    // Replace the constant and the usage of its result
    namer.replaceOp(cstOp, newCstOp);
    cstOp.getResult().replaceAllUsesWith(newCstOp.getResult());
    rewriter.replaceOp(cstOp, newCstOp->getResults());
  }
  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::ftdVerifyAndCreateMemInterfaces(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
    MemInterfacesInfo &memInfo, const BlockIndexing &bi) const {

  /// Given an LSQ, extract the list of operations which require that same LSQ
  auto getLSQOperations =
      [&](const llvm::MapVector<unsigned, SmallVector<Operation *>> &lsqPorts)
      -> SmallVector<Operation *> {
    // Result vector holding the result
    SmallVector<Operation *> combinedOperations;

    // Iterate over the MapVector and add all Operation* to the
    // combinedOperations vector
    for (const auto &entry : lsqPorts) {
      const SmallVector<Operation *> &operations = entry.second;
      combinedOperations.insert(combinedOperations.end(), operations.begin(),
                                operations.end());
    }
    return combinedOperations;
  };

  if (memInfo.empty())
    return success();

  // Get the CFG loop information
  mlir::DominanceInfo domInfo;

  // Create a mapping between each block and all the other blocks it
  // properly dominates so that we can quickly determine whether LSQ groups
  // make sense
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : funcOp) {
    // Start with an empty set of dominated blocks for each potential
    // dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : funcOp) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Find the control value indicating the last control flow decision in the
  // function; it will be fed to memory interfaces to indicate that no more
  // group allocations will be coming
  Value ctrlEnd;
  auto returns = funcOp.getOps<func::ReturnOp>();
  assert(!returns.empty() && "no returns in function");
  if (std::distance(returns.begin(), returns.end()) == 1) {
    ctrlEnd = getBlockControl((*returns.begin())->getBlock());
  } else {
    // Merge the control signals of all blocks with a return to create a
    // control representing the final control flow decision
    SmallVector<Value> controls;
    func::ReturnOp lastRetOp;
    for (func::ReturnOp retOp : returns) {
      lastRetOp = retOp;
      controls.push_back(getBlockControl(retOp->getBlock()));
    }
    rewriter.setInsertionPointToStart(lastRetOp->getBlock());
    auto mergeOp =
        rewriter.create<handshake::MergeOp>(lastRetOp.getLoc(), controls);
    ctrlEnd = mergeOp.getResult();

    // The merge goes into an extra "end block" after all others, this will
    // be where the function end terminator will be located as well
    mergeOp->setAttr(BB_ATTR_NAME,
                     rewriter.getUI32IntegerAttr(funcOp.getBlocks().size()));
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [blockIdx, block] : llvm::enumerate(funcOp))
    ctrlVals.insert({blockIdx, getBlockControl(&block)});

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
    SmallPtrSet<Block *, 4> controlBlocks;

    FtdMemoryInterfaceBuilder memBuilder(funcOp, memref, memAccesses.memStart,
                                         ctrlEnd, ctrlVals);

    // Add MC ports to the interface builder
    for (auto &[_, mcBlockOps] : memAccesses.mcPorts) {
      for (Operation *mcOp : mcBlockOps)
        memBuilder.addMCPort(mcOp);
    }

    // Determine LSQ group validity and add ports the interface builder
    // at the same time
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      assert(!groupOps.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<Operation *>> opsPerBlock;
      for (Operation *op : groupOps)
        opsPerBlock[op->getBlock()].push_back(op);

      // Check whether there is a clear "linear dominance" relationship
      // between all blocks, and derive a port ordering for the group from
      // it
      SmallVector<Block *> order;
      if (failed(computeLinearDominance(dominations, opsPerBlock, order)))
        return failure();

      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(order.front()); !newCtrl)
        return groupOps.front()->emitError()
               << "Inconsistent LSQ group for memory interface the "
                  "operation "
                  "references. No two groups can have the same control "
                  "signal.";

      // Add all group ports in the correct order to the builder. Within
      // each block operations are naturally in program order since we
      // always use ordered maps and iterated over the operations in program
      // order to begin with
      for (Block *block : order) {
        for (Operation *lsqOp : opsPerBlock[block])
          memBuilder.addLSQPort(group, lsqOp);
      }
    }

    // Build the memory interfaces.
    // If the memory accesses require an LSQ, then the Fast Load-Store queue
    // allocation method from FPGA'23 is used. In particular, first the
    // groups allocation is performed together with the creation of the fork
    // graph. Afterwards, the FTD methodology is used to interconnect the
    // elements correctly.
    if (memAccesses.lsqPorts.size() > 0) {

      mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

      // Get all the operations associated to an LSQ
      SmallVector<Operation *> allOperations =
          getLSQOperations(memAccesses.lsqPorts);

      // Get all the dependencies among the BBs of the related operations.
      // Two memory operations are dependant if:
      // 1. They are in different BBs;
      // 2. One of them is a write operations;
      // 3. They are not mutually exclusive.
      SmallVector<ProdConsMemDep> allMemDeps;
      identifyMemoryDependencies(allOperations, allMemDeps, loopInfo, bi);

      for (auto &dep : allMemDeps)
        dep.printDependency();

      // Get the initial start signal, which is the last argument of the
      // function
      auto startValue = (Value)funcOp.getArguments().back();

      // Stores the Groups graph required for the allocation network
      // analysis
      DenseSet<Group *> groupsGraph;
      constructGroupsGraph(allOperations, allMemDeps, groupsGraph);
      minimizeGroupsConnections(groupsGraph, bi);

      for (auto &g : groupsGraph)
        g->printDependenices();

      // Build the memory interfaces
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;

      // As we instantiate the interfaces for the LSQ for each memory
      // operation, we need to add some forks in order for the control input
      // to be propagated. In particular, we want to keep track of the control
      // value associated to each basic block in the region
      DenseMap<Block *, Operation *> forksGraph;

      // Instantiate the interfaces and a lazy fork for each group
      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groupsGraph, forksGraph, startValue)))
        return failure();

      for (Group *consumerGroup : groupsGraph) {
        SmallVector<Value> differentInputs;
        Operation *consumerLF = forksGraph[consumerGroup->bb];
        for (Group *producerGroup : consumerGroup->preds) {
          Operation *producerLF = forksGraph[producerGroup->bb];
          SmallVector<Value> forkValuesToConnect = {startValue,
                                                    producerLF->getResult(0)};
          auto phiNetworkOrFailure = createPhiNetwork(
              funcOp.getRegion(), rewriter, forkValuesToConnect);
          if (failed(phiNetworkOrFailure))
            return failure();

          auto &phiNetwork = *phiNetworkOrFailure;
          differentInputs.push_back(phiNetwork[consumerGroup->bb]);
        }

        if (differentInputs.size() == 0)
          differentInputs.push_back(startValue);

        consumerLF->setOperands(differentInputs);
      }

      if (failed(joinInsertion(rewriter, groupsGraph, forksGraph)))
        return failure();

    } else {
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
        return failure();
    }
  }

  return success();
}

void ftd::FtdLowerFuncToHandshake::identifyMemoryDependencies(
    const SmallVector<Operation *> &operations,
    SmallVector<ProdConsMemDep> &allMemDeps, const mlir::CFGLoopInfo &li,
    const BlockIndexing &bi) const {

  // Returns true if there exist a path between `op1` and `op2`
  auto isThereAPath = [&](Operation *op1, Operation *op2) -> bool {
    return !findAllPaths(op1->getBlock(), op2->getBlock(), bi).empty();
  };

  // Returns true if two operations are both load
  auto areBothLoad = [](Operation *op1, Operation *op2) {
    return (isa<handshake::LSQLoadOp>(op1) && isa<handshake::LSQLoadOp>(op2));
  };

  // Returns true if two operations belong to the same block
  auto isSameBlock = [](Operation *op1, Operation *op2) {
    return (op1->getBlock() == op2->getBlock());
  };

  // Given all the operations which are assigned to an LSQ, loop over them
  // and skip those which are not memory operations
  for (Operation *i : operations) {

    if (!isHandhsakeLSQOperation(i))
      continue;

    // Loop over all the other operations in the LSQ. There is no dependency
    // in the following cases:
    // 1. One of them is not a memory operation;
    // 2. The two operation are in the same group, thus they are in the same
    // BB;
    // 3. They are both load operations;
    // 4. The operations are mutually exclusive (i.e. there is no path which
    // goes from i to j and vice-versa);
    for (Operation *j : operations) {

      if (!isHandhsakeLSQOperation(j) || isSameBlock(i, j) ||
          areBothLoad(i, j) || (!isThereAPath(i, j) && !isThereAPath(j, i)))
        continue;

      // Get the two blocks
      Block *bbI = i->getBlock();
      Block *bbJ = j->getBlock();

      // If the relationship was already present, then skip the pairs of
      // blocks
      auto *it = llvm::find_if(allMemDeps, [bbI, bbJ](ProdConsMemDep p) {
        return p.prodBb == bbJ && p.consBb == bbI;
      });

      if (it != allMemDeps.end())
        continue;

      // Insert a dependency only if index _j is smaller than index _i: in
      // this case i is the producer, j is the consumer. If this doesn't
      // hold, the dependency will be added when the two blocks are analyzed
      // in the opposite direction
      if (bi.lessIndex(bbJ, bbI)) {

        // and add it to the list of dependencies
        ProdConsMemDep oneMemDep(bbJ, bbI, false);
        allMemDeps.push_back(oneMemDep);

        // If the two blocks are in the same loop, then bbI is also a
        // consumer, while bbJ is a producer. This relationship is backward.
        if (isSameLoopBlocks(bbI, bbJ, li)) {
          ProdConsMemDep opp(bbI, bbJ, true);
          allMemDeps.push_back(opp);
        }
      }
    }
  }
}

LogicalResult ftd::FtdLowerFuncToHandshake::addExplicitPhi(
    func::FuncOp funcOp, ConversionPatternRewriter &rewriter) const {

  using namespace experimental::gsa;

  // The function instantiates the GAMMA and MU gates as provided by the GSA
  // analysis pass. A GAMMA function is translated into a multiplxer driven by
  // single control signal and fed by two operands; a MU function is
  // translated into a multiplxer driven by an init (it is currently
  // implemented as a Merge fed by a constant triggered from Start once and
  // from the loop condition thereafter). The input of one of these functions
  // might be another GSA function, and it's possible that the function was
  // not instantiated yet. For this reason, we keep track of the missing
  // operands, and reconnect them later on.
  //
  // Also, a GAMMA function might have an empty data input: GAMMA(c, EMPTY,
  // V). In this case, the function is translated into a branch.
  //
  // To simplify the way GSA functions are handled, each of them has an unique
  // index.

  struct MissingGsa {
    // Index of the GSA function to modify
    unsigned phiIndex;
    // Index of the GSA function providing the result
    unsigned edgeIndex;
    // Index of the operand to modify
    unsigned operandInput;

    MissingGsa(unsigned pi, unsigned ei, unsigned oi)
        : phiIndex(pi), edgeIndex(ei), operandInput(oi) {}
  };

  if (funcOp.getBlocks().size() == 1)
    return success();

  // List of missing GSA functions
  SmallVector<MissingGsa> missingGsaList;
  // List of gammas with only one input
  DenseSet<Operation *> oneInputGammaList;
  // Maps the index of each GSA function to each real operation
  DenseMap<unsigned, Operation *> gsaList;

  // For each block excluding the first one, which has no gsa
  for (Block &block : llvm::drop_begin(funcOp)) {

    // For each GSA function
    ArrayRef<Gate *> phis = gsaAnalysis.getGates(&block);
    for (Gate *phi : phis) {

      // Skip if it's a phi
      if (phi->gsaGateFunction == PhiGate)
        continue;

      Location loc = block.front().getLoc();
      rewriter.setInsertionPointToStart(&block);
      SmallVector<Value> operands;

      // Maintain the index of the current operand
      unsigned operandIndex = 0;
      // Checks whether one index is empty
      int nullOperand = -1;

      // For each of its operand
      for (auto *operand : phi->operands) {
        // If the input is another GSA function, then a dummy value is used as
        // operand and the operations will be reconnected later on.
        // If the input is empty, we keep track of its index.
        // In the other cases, we already have the operand of the function.
        if (operand->isTypeGate()) {
          Gate *g = std::get<Gate *>(operand->input);
          operands.emplace_back(g->result);
          missingGsaList.emplace_back(
              MissingGsa(phi->index, g->index, operandIndex));
        } else if (operand->isTypeEmpty()) {
          nullOperand = operandIndex;
          operands.emplace_back(nullptr);
        } else {
          auto val = std::get<Value>(operand->input);
          val.setType(channelifyType(val.getType()));
          operands.emplace_back(val);
        }
        operandIndex++;
      }

      // The condition value is provided by the `condition` field of the phi
      rewriter.setInsertionPointAfterValue(phi->result);
      Value conditionValue =
          phi->conditionBlock->getTerminator()->getOperand(0);

      // If the function is MU, then we create a merge
      // and use its result as condition
      if (phi->gsaGateFunction == MuGate) {
        Region &region = funcOp.getBody();
        mlir::DominanceInfo domInfo;
        mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

        // The inputs of the merge are the condition value and a `false`
        // constant driven by the start value of the function. This will
        // created later on, so we use a dummy value.
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(conditionValue);
        mergeOperands.push_back(conditionValue);

        auto initMergeOp =
            rewriter.create<handshake::MergeOp>(loc, mergeOperands);

        initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

        // Replace the new condition value
        conditionValue = initMergeOp->getResult(0);
        conditionValue.setType(channelifyType(conditionValue.getType()));
      }

      // When a single input gamma is encountered, a mux is inserted as a
      // placeholder to perform the gamma/mu allocation flow. In the end,
      // these muxes are erased from the IR
      if (nullOperand >= 0) {
        operands[0] = operands[1 - nullOperand];
        operands[1] = operands[1 - nullOperand];
      }

      // Create the multiplexer
      auto mux = rewriter.create<handshake::MuxOp>(loc, phi->result.getType(),
                                                   conditionValue, operands);

      // The one input gamma is marked at an operation to skip in the IR and
      // later removed
      if (nullOperand >= 0) {
        oneInputGammaList.insert(mux);
        mux->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
      }

      if (phi->isRoot)
        rewriter.replaceAllUsesWith(phi->result, mux.getResult());

      gsaList.insert({phi->index, mux});
      mux->setAttr(FTD_EXPLICIT_PHI, rewriter.getUnitAttr());
    }
  }

  // For each of the GSA missing inputs, perform a replacement
  for (auto &missingMerge : missingGsaList) {

    auto *operandMerge = gsaList[missingMerge.phiIndex];
    auto *resultMerge = gsaList[missingMerge.edgeIndex];

    operandMerge->setOperand(missingMerge.operandInput + 1,
                             resultMerge->getResult(0));

    // In case of a one-input gamma, the other input must be replaced as well,
    // to avoid errors when the block arguments are erased later on
    if (oneInputGammaList.contains(operandMerge))
      operandMerge->setOperand(2 - missingMerge.operandInput,
                               resultMerge->getResult(0));
  }

  // Get rid of the multiplexers adopted as place-holders of one input gamma
  for (auto &op : llvm::make_early_inc_range(oneInputGammaList)) {
    int operandToUse = llvm::isa_and_nonnull<handshake::MuxOp>(
                           op->getOperand(1).getDefiningOp())
                           ? 1
                           : 2;
    op->getResult(0).replaceAllUsesWith(op->getOperand(operandToUse));
    rewriter.eraseOp(op);
  }

  // Remove all the block arguments for all the non starting blocks
  for (Block &block : llvm::drop_begin(funcOp))
    block.eraseArguments(0, block.getArguments().size());

  // Each terminator must be replaced so that it does not provide any block
  // arguments (possibly only the final control argument)
  for (Block &block : funcOp) {
    Operation *terminator = block.getTerminator();
    if (terminator) {
      rewriter.setInsertionPointAfter(terminator);
      if (isa<cf::CondBranchOp>(terminator)) {
        auto condBranch = dyn_cast<cf::CondBranchOp>(terminator);
        SmallVector<Value> trueOperands;
        SmallVector<Value> falseOperands;
        auto newCondBranch = rewriter.create<cf::CondBranchOp>(
            condBranch->getLoc(), condBranch.getCondition(),
            condBranch.getTrueDest(), trueOperands, condBranch.getFalseDest(),
            falseOperands);
        rewriter.replaceOp(condBranch, newCondBranch);
      } else if (isa<cf::BranchOp>(terminator)) {
        auto branch = dyn_cast<cf::BranchOp>(terminator);
        SmallVector<Value> operands;
        auto newBranch = rewriter.create<cf::BranchOp>(
            branch->getLoc(), branch.getDest(), operands);
        rewriter.replaceOp(branch, newBranch);
      }
    }
  }

  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get the map of control dependencies for the blocks in the function to
  // lower
  auto cdaDeps = cdAnalaysis.getAllBlockDeps();

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // Add the muxes as obtained by the GSA analysis pass
  if (failed(addExplicitPhi(lowerFuncOp, rewriter)))
    return failure();

  // Save pointers to old block
  SmallVector<Block *> blocksCfFunction;
  for (auto &bb : lowerFuncOp.getBlocks())
    blocksCfFunction.push_back(&bb);

  // First lower the parent function itself, without modifying its body
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // Save pointers from new blocks
  SmallVector<Block *> blocksHandshakeFunc;
  for (auto &bb : funcOp.getBlocks())
    blocksHandshakeFunc.push_back(&bb);

  // Remap control dependency analysis
  for (unsigned i = 0; i < blocksCfFunction.size(); i++) {
    auto deps = cdaDeps[blocksCfFunction[i]];

    for (unsigned i = 0; i < blocksCfFunction.size(); i++) {
      if (deps.allControlDeps.contains(blocksCfFunction[i]))
        deps.allControlDeps.insert(blocksHandshakeFunc[i]);
      if (deps.forwardControlDeps.contains(blocksCfFunction[i]))
        deps.forwardControlDeps.insert(blocksHandshakeFunc[i]);
    }

    cdaDeps.erase(blocksCfFunction[i]);
    cdaDeps.insert({blocksHandshakeFunc[i], deps});
  }

  // Get the block indexing
  BlockIndexing bi(funcOp.getRegion());

  // When GSA-MU functions are translated into multiplexers, an `init merge`
  // is created to feed them. This merge requires the start value of the
  // function as one of its data inputs. However, the start value was not
  // present yet when `addExplicitPhi` is called, thus we need to reconnect
  // it.
  connectInitMerges(rewriter, funcOp);

  // Stores mapping from each value that passes through a merge-like
  // operation to the data result of that merge operation
  ArgReplacements argReplacements;

  // Currently, the following 2 functions do nothing but construct the network
  // of CMerges in complete isolation from the rest of the components
  // implementing the operations
  // In particular, the addMergeOps relies on adding Merges for every block
  // argument but because we removed all "real" arguments, we are only left
  // with the Start value as an argument for every block
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports
  // get tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);

  // Create the memory interface according to the algorithm from FPGA'23. This
  // functions introduce new data dependencies that are then passed to FTD for
  // correctly delivering data between them like any real data dependencies
  if (failed(ftdVerifyAndCreateMemInterfaces(funcOp, rewriter, memInfo, bi)))
    return failure();

  // Convert the constants and undefined values from the `arith` dialect to
  // the `handshake` dialect, while also using the start value as their
  // control value
  if (failed(convertConstants(rewriter, funcOp)) ||
      failed(convertUndefinedValues(rewriter, funcOp)))
    return failure();

  if (funcOp.getBlocks().size() != 1) {
    // Add muxes for regeneration of values in loop
    if (failed(addRegen(rewriter, funcOp)))
      return failure();

    analyzeLoop(funcOp);

    // Add suppression blocks between each pair of producer and consumer
    if (failed(addSupp(rewriter, funcOp, cdaDeps, bi)))
      return failure();
  }

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  auto blockConnections = getCFGEdges(funcOp.getRegion(), namer);
  auto resAttr = CFGEdge::serializeEdges(blockConnections);
  funcOp->setAttr(CFG_EDGES, rewriter.getStringAttr(resAttr));

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass> ftd::createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}
