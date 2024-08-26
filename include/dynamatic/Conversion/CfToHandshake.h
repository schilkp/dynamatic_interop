//===- CfToHandhsake.h - Convert func/cf to handhsake dialect ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-cf-to-handshake conversion pass along with a
// helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
#define DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Shannon.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <set>

namespace dynamatic {

/// This class is strongly inspired by CIRCT's own `HandshakeLowering` class. It
/// provides all the conversion steps necessary to concert a func-level function
/// into a matching handshake-level function.
class HandshakeLowering {
public:
  /// Groups memory operations by interface and group for a given memory region.
  struct MemAccesses {
    /// Memory operations for a simple memory controller, grouped by
    /// originating basic block.
    llvm::MapVector<Block *, SmallVector<Operation *>> mcPorts;
    /// Memory operations for an LSQ, grouped by belonging LSQ group.
    llvm::MapVector<unsigned, SmallVector<Operation *>> lsqPorts;
  };

  /// Groups information to "rewire the IR" around a particular merge-like
  /// operation.
  struct MergeOpInfo {
    /// The merge-like operation under consideration.
    handshake::MergeLikeOpInterface mergeLikeOp;
    /// The original block argument that the merge-like operation "replaces".
    BlockArgument blockArg;
    /// All data operands to the merge-like operation that need to be resolved
    /// during branch insertion.
    SmallVector<Backedge> dataEdges;
    /// An optional index operand that needs to be resolved for mux-like
    /// operations.
    std::optional<Backedge> indexEdge{};
  };

  /// Groups information to rewire the IR around merge-like operations by owning
  /// basic block (which must still exist).
  using BlockOps = DenseMap<Block *, std::vector<MergeOpInfo>>;

  /// Stores a mapping between memory regions (identified by the function
  /// argument they correspond to) and the set of memory operations referencing
  /// them.
  using MemInterfacesInfo = llvm::MapVector<Value, MemAccesses>;

  /// Constructor simply takes the region being lowered and a reference to the
  /// top-level name analysis.
  // explicit HandshakeLowering(Region &region, NameAnalysis &nameAnalysis)
  //     : region(region), nameAnalysis(nameAnalysis) {}

  explicit HandshakeLowering(Region &region, int funcOpIdx,
                             NameAnalysis &nameAnalysis,
                             ControlDependenceAnalysis &cdgAnalysis,
                             mlir::DominanceInfo &domInfo)
      : region(region), funcOpIdx(funcOpIdx), nameAnalysis(nameAnalysis),
        cdgAnalysis(cdgAnalysis), li(domInfo.getDomTree(&region)) {}

  /// Creates the control-only network by adding a control-only argument to the
  /// region's entry block and forwarding it through all basic blocks.
  LogicalResult createControlNetwork(ConversionPatternRewriter &rewriter);

  /// Adds merge-like operations after all block arguments within the region,
  /// then removes all block arguments and corresponding branch operands. This
  /// always succeeds.
  LogicalResult addMergeOps(ConversionPatternRewriter &rewriter);

  /// Adds handshake-level branch-like operations before all cf-level
  /// branch-like terminators within the region. This needs to happen after
  /// merge-insertion because it also replaces data operands of merge-like
  /// operations with the result value(s) of inserted branch-like operations.
  /// This always succeeds.
  LogicalResult addBranchOps(ConversionPatternRewriter &rewriter);

  /// Identifies all memory interfaces and their associated operations in the
  /// function, replaces all load/store-like operations by their handshake
  /// counterparts, and fills `memInfo` with information about which operations
  /// use which interface.
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  /// Verifies that LSQ groups derived from input IR annotations make sense
  /// (check for linear dominance property within each group and cross-group
  /// control signal compatibility). Then, instantiates all memory interfaces
  /// and connects them to their respective load/store operations. For each
  /// memory region:
  /// - A single `handshake::MemoryControllerOp` will be instantiated if all of
  /// its accesses indicate that they should connect to an MC.
  /// - A single `handshake::LSQOp` will be instantiated if none of
  /// its accesses indicate that they should connect to an LSQ.
  /// - Both a `handhsake::MemoryControllerOp` and `handhsake::LSQOp` will be
  /// instantiated if some but not all of its accesses indicate that they should
  /// connect to an LSQ.
  LogicalResult
  verifyAndCreateMemInterfaces(ConversionPatternRewriter &rewriter,
                               MemInterfacesInfo &memInfo);

  /// Converts each `func::CallOp` operation to an equivalent
  /// `handshake::InstanceOp` operation. This always succeeds.
  LogicalResult convertCalls(ConversionPatternRewriter &rewriter);

  /// Connect constants to the rest of the circuit. Constants are triggered by a
  /// source if their successor is not a branch/return or memory operation.
  /// Otherwise they are triggered by the control-only network.
  LogicalResult connectConstants(ConversionPatternRewriter &rewriter);

  /// Replaces undefined operations (mlir::LLVM::UndefOp) with a default "0"
  /// constant triggered by the enclosing block's control merge.
  LogicalResult replaceUndefinedValues(ConversionPatternRewriter &rewriter);

  /// Sets an integer "bb" attribute on each operation to identify the basic
  /// block from which the operation originates in the std-level IR.
  LogicalResult idBasicBlocks(ConversionPatternRewriter &rewriter);

  /// Creates the region's return network by sequentially moving all blocks'
  /// operations to the entry block, replacing func::ReturnOp's with
  /// handshloopsake::ReturnOp's, deleting all block terminators and non-entry
  /// blocks, merging the results of all return statements, and creating the
  /// region's end operation./// This class is strongly inspired by CIRCT's own
  /// `HandshakeLowering` class. It
  /// provides all the conversion steps necessary to concert a func-level
  /// function into a matching handshake-level function.
  LogicalResult createReturnNetwork(ConversionPatternRewriter &rewriter);

  /// Returns the entry control value for operations contained within this
  /// block.
  Value getBlockEntryControl(Block *block) const {
    auto it = blockControls.find(block);
    assert(it != blockControls.end() &&
           "No block entry control value registerred for this block!");
    return it->second;
  }

  /// Set the control value of a basic block.
  void setBlockEntryControl(Block *block, Value v) {
    blockControls[block] = v;
  };

  /// Returns a reference to the region being lowered.
  Region &getRegion() { return region; }

  //----------Construction of Allocation Network----------

  /// Interfaces dataflow circuits with LSQs

  /// Identifies all the memory dependencies between the predecessors of an LSQ.
  /// This
  /// is the first step towards making memory deps explicit
  void identifyMemDeps(std::vector<Operation *> &operations,
                       std::vector<ProdConsMemDep> &allMemDeps);

  /// Builds a dependence graph betweeen the groups
  void constructGroupsGraph(std::vector<Operation *> &operations,
                            std::vector<ProdConsMemDep> &allMemDeps,
                            std::set<Group *, GroupsComparator> &groups);

  /// Minimizes the connections between groups based on dominance info
  void minimizeGroupsConnections(std::set<Group *, GroupsComparator> &groups);

  /// Add MERGEs in the case where the consumer might consume but the producer
  /// not necessarily poduce (the counsumer is being fed by another producer)
  LogicalResult addMergeNonLoop(OpBuilder &builder,
                                std::vector<ProdConsMemDep> &allMemDeps,
                                std::set<Group *, GroupsComparator> &groups,
                                DenseMap<Block *, Operation *> &forksGraph);

  /// Add MERGEs in the case where the producer BB is after the consumer BB (the
  /// producer and the consumer are in a loop)
  LogicalResult addMergeLoop(OpBuilder &builder,
                             std::set<Group *, GroupsComparator> &groups,
                             DenseMap<Block *, Operation *> &forksGraph);

  /// Join all the operands of the LazyForks
  LogicalResult joinInsertion(OpBuilder &builder,
                              std::set<Group *, GroupsComparator> &groups,
                              DenseMap<Block *, Operation *> &forksGraph);

  /// If a Fork operation has more than 2 operands, then it creates a join for
  /// the operands. The result of the JOIN becomes the operand of the ForkOp
  void insertJoins(std::set<Operation *> forks);

  ///----------Fast Token Delivery Algorithm----------

  /// Adds MERGES in fast token delivery algorithm
  LogicalResult addPhi(ConversionPatternRewriter &rewriter);

  /// Removes redundant MERGEs that were created by addPhi
  LogicalResult removeRedundantPhis(ConversionPatternRewriter &rewriter);

  /// Adds BRANCHES in fast token delivery algorithm
  LogicalResult addSupp(ConversionPatternRewriter &rewriter);

  /// Adds BRANCHES in fast token delivery algorithm for the case where the
  /// producer is a BRANCH added by addSupp
  LogicalResult addSuppBranches(ConversionPatternRewriter &rewriter,
                                std::set<Operation *> &oldBranches,
                                size_t &count);

  /// Adds BRANCHES in fast token delivery algorithm for the case where the
  /// producer is START by doing back-propagation
  LogicalResult addSuppForStart(ConversionPatternRewriter &rewriter);

  /// Inserts a BRANCH for in the loop with condition depending on the exit
  /// blocks Loop convention followed is: True Side --> loop exit False Side
  /// --> iterate
  Value insertBranchToLoop(ConversionPatternRewriter &rewriter,
                           mlir::CFGLoop *loop, Operation *consumer,
                           Value connection, bool moreProdThanCons,
                           bool selfRegeneration);

  /// Adds BRANCHes in the case where the producer is in more loops than the
  /// consumer (to solve token count mismatch problem)
  void manageMoreProdThanCons(ConversionPatternRewriter &rewriter,
                              Block *producerBlock, Operation *consumer,
                              Value connection);

  /// Adds BRANCHes in the case where an operation is feeding itself, i.e the
  /// producer is the same as the consumer(to solve token count mismatch
  /// problrm)
  void manageSelfRegeneration(ConversionPatternRewriter &rewriter,
                              Operation *consumer, Value connection);

  /// Adds BRANCHes in the case where the consumer is in more loops than the
  /// producer
  void manageNonLoop(ConversionPatternRewriter &rewriter, Block *producerBlock,
                     Operation *consumer, Value connection);

  void manageDifferentRegeneration(ConversionPatternRewriter &rewriter,
                                   Operation *consumer, Value connection);

  ///----------Fast Token Delivery Cleanup----------

  /// Converts te MERGEs created in fast token delivery to MUXes
  LogicalResult convertMergesToMuxes(ConversionPatternRewriter &rewriter);

  Value addInit(ConversionPatternRewriter &rewriter,
                SmallVector<Operation *> &initMerges, Operation *oldMerge);

  /// Replaces the input of constants with the START value
  LogicalResult triggerConstantsFromStart(ConversionPatternRewriter &rewriter);

  /// Adds BRANCHes in the case where the consumer is a mux and the cdg is empty
  LogicalResult addSuppGSA(ConversionPatternRewriter &rewriter);

protected:
  /// The region being lowered.
  Region &region;

  // The index of the funcOp containing the region being lowered in the list of
  // funcOps in the ModuleOp
  int funcOpIdx;

  /// Start point of the control-only network
  BlockArgument startCtrl;

  /// Inserts a merge-like operation in the IR for the block argument and
  /// returns information necessary to rewire the IR around the new operation
  /// once all merges have been inserted. A control-merge is inserted for
  /// control-only (data-less) arguments. For other types of arguments, a
  /// non-deterministic merge is inserted for blocks with 0 or a single
  /// predecessor while a mux is inserted for blocks with multiple predecessors.
  MergeOpInfo insertMerge(BlockArgument blockArg, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

  bool sameLoop(Block *source, Block *dest);

  /// Checks if all the blocks in the path are in the control dependency
  bool checkControlDep(const SmallVector<Block *, 4> &controlDeps,
                       const std::vector<Block *> &path);

private:
  /// Associates basic blocks of the region being lowered to their respective
  /// control value.
  DenseMap<Block *, Value> blockControls;
  /// Name analysis to name new memory operations as they are created and keep
  /// reference accesses in memory dependencies consistent.
  NameAnalysis &nameAnalysis;
  // Control dependence analysis to identify the conditions of production and
  // consumption of operations to implement fast token delivery
  ControlDependenceAnalysis &cdgAnalysis;

  /// Stores the loop info of the control flow graph
  mlir::CFGLoopInfo li;

  // Enumerates all paths from the start Block to the end Block in the CFG and
  // returns a minimizedSOP while respecting the control dependencies
  experimental::boolean::BoolExpression *
  enumeratePaths(Block *start, Block *end,
                 const SmallVector<Block *, 4> &controlDeps);

  // Gets the innermost loop containing bpth block1 nd block 2
  mlir::CFGLoop *getInnermostCommonLoop(Block *block1, Block *block2);

  // Gets all the loops that the consumer is in but not te producer, in-order of
  // outermost to innermost loop
  SmallVector<mlir::CFGLoop *> getLoopsConsNotInProd(Block *cons, Block *prod);

  bool isaMergeLoop(Operation *merge);

  // Checks if an operation is a Branch in a loop exit block
  bool isBranchInLoopExit(Operation *op);

  // Gets the loop exit condition of the bck.
  // If the loop exit is on the false side of the block, then the condition
  // is negated
  experimental::boolean::BoolExpression *
  getBlockLoopExitCondition(Block *loopExit, mlir::CFGLoop *loop);

  //----------BooleanExpression to Circuit----------

  /// Associates the condition of the block in string format to its
  /// corresponding control value. The control alue is given by the condition of
  /// the terminator of the block
  std::map<std::string, Value> conditionToValue;

  /// Fills conditionToValue map by getting the condition and the value of each
  /// block
  void fillConditionToValueMapping();

  // Converts a DATA (mux or boolean expression) in boolean logic library to
  // actual circuitry
  Value dataToCircuit(ConversionPatternRewriter &rewriter,
                      experimental::boolean::Data *data, Block *block);

  // Converts a mux in boolean logic library to actual circuitry
  Value muxToCircuit(ConversionPatternRewriter &rewriter,
                     experimental::boolean::MUX *mux, Block *block);

  // Converts a boolean expession to actual circuitry
  Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                experimental::boolean::BoolExpression *expr,
                                Block *block);

  // Converts a boolean variable to actual circuitry
  Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                              experimental::boolean::BoolExpression *expr,
                              Block *block);

  // Fixes a MERGE operation to follow the loop convention
  /// True Side --> loop exit
  /// False Side --> iterate
  void fixConvention(Operation *merge, mlir::CFGLoop *loop);

  /// contains all operations created by fast token delivery algorithm
  std::vector<Operation *> allocationNetwork;

  /// contains all Merges created by addPhi
  SmallVector<Operation *> phiMerges;

  /// contains all Branches created by manageMoreProdThanCons or
  /// manageDifferentRegeneration
  std::vector<Operation *> suppBranches;

  /// contains all Branches created by manageSelfRegeneration
  SmallVector<Operation *> selfGenBranches;

  /// contains all merges added in the straight LSQ
  SmallVector<Operation *> memDepLoopMerges;

  /// contains all MUXes created by Shannon
  SmallVector<Operation *> shannonMUXes;

  /// contains all constants created by addINIT or for Shannon’s
  SmallVector<Operation *> networkConstants;
};

/// Pointer to function lowering a region using a conversion pattern rewriter.
using RegionLoweringFunc =
    llvm::function_ref<LogicalResult(Region &, ConversionPatternRewriter &)>;

/// Partially lowers a region using a provided lowering function.
LogicalResult partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                                   Region &region);

/// Runs a partial lowering method on an instance of the class the method
/// belongs to. We need two variadic template parameters because arguments
/// provided to this function may be slightly differeprod_cons_mem_depnt but
/// convertible to the arguments expected by the partial lowering method.
/// Success status is forwarded from the partial lowering method.
template <typename T, typename... TArgs1, typename... TArgs2>
static LogicalResult runPartialLowering(
    T &instance,
    LogicalResult (T::*memberFunc)(ConversionPatternRewriter &, TArgs2...),
    TArgs1 &...args) {
  return partiallyLowerRegion(
      [&](Region &, ConversionPatternRewriter &rewriter) -> LogicalResult {
        return (instance.*memberFunc)(rewriter, args...);
      },
      instance.getRegion());
}

#define GEN_PASS_DECL_CFTOHANDSHAKE
#define GEN_PASS_DEF_CFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createCfToHandshake();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
