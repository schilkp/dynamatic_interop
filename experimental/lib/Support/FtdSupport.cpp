//===- FtdSupport.cpp - FTD conversion support -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FtdSupport.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;

DenseMap<unsigned, ftd::CFGEdge> ftd::getCFGEdges(Region &funcRegion,
                                                  NameAnalysis &namer) {

  DenseMap<unsigned, ftd::CFGEdge> edgeMap;

  // Get the ID of a block through the ID annotated in the terminator operation
  auto getIDBlock = [&](Block *bb) -> unsigned {
    auto idOptional = getLogicBB(bb->getTerminator());
    if (!idOptional.has_value())
      bb->getTerminator()->emitError() << "Operation has no BB annotated\n";
    return idOptional.value();
  };

  // For each block in the IR
  for (auto &block : funcRegion.getBlocks()) {

    // Get the terminator and its block ID
    auto *terminator = block.getTerminator();
    unsigned blockID = getIDBlock(&block);

    // If the terminator is a branch, then the edge is unconditional. If the
    // terminator is `cond_br`, then the branch is conditional.
    if (auto branchOp = dyn_cast<cf::BranchOp>(terminator); branchOp) {
      edgeMap.insert(
          {blockID, ftd::CFGEdge(getIDBlock(branchOp->getSuccessor(0)))});
    } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(terminator);
               condBranchOp) {
      // Get the name of the operation which defines the condition used for the
      // branch
      std::string conditionName =
          namer.getName(condBranchOp.getOperand(0).getDefiningOp()).str();

      // Get IDs of both true and false destinations
      unsigned trueDestID = getIDBlock(condBranchOp.getTrueDest());
      unsigned falseDestID = getIDBlock(condBranchOp.getFalseDest());
      edgeMap.insert(
          {blockID, ftd::CFGEdge(trueDestID, falseDestID, conditionName)});
    } else if (!llvm::isa_and_nonnull<func::ReturnOp>(terminator)) {
      terminator->emitError()
          << "Not a cf terminator for BB" << blockID << "\n";
    }
  }

  for (auto &[start, edge] : edgeMap) {
    llvm::dbgs() << start << " -> ";
    edge.print();
  }

  return edgeMap;
}

LogicalResult ftd::restoreCfStructure(
    handshake::FuncOp &funcOp, const DenseMap<unsigned, ftd::CFGEdge> &edges,
    ConversionPatternRewriter &rewriter, NameAnalysis &namer) {

  // Maintains the ID of the current block under analysis
  unsigned currentBlockID = 0;

  // Maintains the current block under analysis: all the operations in block 0
  // are maintained in the same location, while the others operations are
  // inserted in the new blocks
  Block *currentBlock = &funcOp.getBlocks().front();

  // Keep a mapping between each index and each block. Since the block 0 is kept
  // as it is, it can be inserted already in the map.
  DenseMap<unsigned, Block *> indexToBlock;
  indexToBlock.insert({0, currentBlock});

  // Temporary store all the operations in the original function
  SmallVector<Operation *> originalOps;
  for (auto &op : funcOp.getOps())
    originalOps.push_back(&op);

  // For each operation
  for (auto *op : originalOps) {

    // Get block ID of the current operation. If it is not annotated (end
    // operation/LSQ/memory operation) then use the current ID
    unsigned opBlock = getLogicBB(op).value_or(currentBlockID);

    // Do not modify the block structure if we are in block 0
    if (opBlock == 0)
      continue;

    // If a new ID is found with respect to the old one, then create a new block
    // in the function
    if (opBlock != currentBlockID) {
      currentBlock = funcOp.addBlock();
      currentBlockID++;
      indexToBlock.insert({currentBlockID, currentBlock});
    }

    // Move the current operation at the end of the new block we are currently
    // using
    op->moveBefore(currentBlock, currentBlock->end());
  }

  // Once we are done creating the blocks, we need to insert the terminators to
  // obtain a proper block structure, using the edge information provided as
  // input

  // For each block
  for (auto [blockID, bb] : llvm::enumerate(funcOp.getBlocks())) {
    rewriter.setInsertionPointToEnd(&bb);

    if (!edges.contains(blockID))
      return failure();

    auto edge = edges.lookup(blockID);

    // Either create a conditional or unconditional branch depending on the type
    // of edge we have
    if (edge.isConditional()) {
      rewriter.create<cf::CondBranchOp>(
          bb.getTerminator()->getLoc(),
          namer.getOp(edge.getCondition())->getResult(0),
          indexToBlock[edge.getTrueSuccessor()],
          indexToBlock[edge.getFalseSuccessor()]);
    } else {
      unsigned successor = edge.getSuccessor();
      rewriter.create<cf::BranchOp>(bb.getTerminator()->getLoc(),
                                    indexToBlock[successor]);
    }
  }
  return success();
}

LogicalResult ftd::flattenFunction(handshake::FuncOp &funcOp,
                                   ConversionPatternRewriter &rewriter) {

  // remove all the `cf.br` and `cf.cond_br` terminators
  for (Block &block : funcOp) {
    Operation *termOp = &block.back();
    if (llvm::isa_and_nonnull<cf::CondBranchOp, cf::BranchOp>(termOp))
      rewriter.eraseOp(termOp);
  }

  // Inline all non-entry blocks into the entry block, erasing them as we go
  Operation *lastOp = &funcOp.front().back();
  for (Block &block : llvm::make_early_inc_range(funcOp)) {
    if (block.isEntryBlock())
      continue;
    rewriter.inlineBlockBefore(&block, lastOp);
  }

  return success();
}

bool ftd::isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || isSameLoop(loop1->getParentLoop(), loop2) ||
          isSameLoop(loop1, loop2->getParentLoop()) ||
          isSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

bool ftd::isSameLoopBlocks(Block *source, Block *dest,
                           const mlir::CFGLoopInfo &li) {
  return isSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

bool ftd::isHandhsakeLSQOperation(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

void ftd::eliminateCommonBlocks(DenseSet<Block *> &s1, DenseSet<Block *> &s2) {

  std::vector<Block *> intersection;
  for (auto &e1 : s1) {
    if (s2.contains(e1))
      intersection.push_back(e1);
  }

  for (auto &bb : intersection) {
    s1.erase(bb);
    s2.erase(bb);
  }
}

bool ftd::isBranchLoopExit(Operation *op, CFGLoopInfo &li) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    if (CFGLoop *loop = li.getLoopFor(op->getBlock()); loop) {
      llvm::SmallVector<Block *> exitBlocks;
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS
static void dfsAllPaths(Block *start, Block *end, std::vector<Block *> &path,
                        std::unordered_set<Block *> &visited,
                        std::vector<std::vector<Block *>> &allPaths,
                        Block *blockToTraverse,
                        const std::vector<Block *> &blocksToAvoid,
                        const ftd::BlockIndexing &bi,
                        bool blockToTraverseFound) {

  // The current block is part of the current path
  path.push_back(start);
  // The current block has been visited
  visited.insert(start);

  bool blockFound = (!blockToTraverse || start == blockToTraverse);

  // If we are at the end of the path, then add it to the list of paths
  if (start == end && (blockFound || blockToTraverseFound)) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (Block *successor : start->getSuccessors()) {

      // Do not run DFS if the successor is in the list of blocks to traverse
      bool incorrectPath = false;
      for (auto *toAvoid : blocksToAvoid) {
        if (toAvoid == successor && bi.greaterIndex(toAvoid, blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      if (visited.find(successor) == visited.end()) {
        dfsAllPaths(successor, end, path, visited, allPaths, blockToTraverse,
                    blocksToAvoid, bi, blockFound || blockToTraverseFound);
      }
    }
  }

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

std::vector<std::vector<Block *>>
ftd::findAllPaths(Block *start, Block *end, const BlockIndexing &bi,
                  Block *blockToTraverse, ArrayRef<Block *> blocksToAvoid) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, blockToTraverse,
              blocksToAvoid, bi, false);
  return allPaths;
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool ftd::isaMergeLoop(Operation *merge, CFGLoopInfo &li) {

  if (merge->getNumOperands() == 1)
    return false;

  Block *bb1 = merge->getOperand(0).getParentBlock();
  if (merge->getOperand(0).getDefiningOp()) {
    auto *op1 = merge->getOperand(0).getDefiningOp();
    while (llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(op1) &&
           op1->getBlock() == merge->getBlock()) {
      auto op = dyn_cast<handshake::ConditionalBranchOp>(op1);
      if (op.getOperand(1).getDefiningOp()) {
        op1 = op.getOperand(1).getDefiningOp();
        bb1 = op1->getBlock();
      } else {
        break;
      }
    }
  }

  Block *bb2 = merge->getOperand(1).getParentBlock();
  if (merge->getOperand(1).getDefiningOp()) {
    auto *op2 = merge->getOperand(1).getDefiningOp();
    while (llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(op2) &&
           op2->getBlock() == merge->getBlock()) {
      auto op = dyn_cast<handshake::ConditionalBranchOp>(op2);
      if (op.getOperand(1).getDefiningOp()) {
        op2 = op.getOperand(1).getDefiningOp();
        bb2 = op2->getBlock();
      } else {
        break;
      }
    }
  }

  return li.getLoopFor(bb1) != li.getLoopFor(bb2);
}

boolean::BoolExpression *
ftd::getPathExpression(ArrayRef<Block *> path,
                       DenseSet<unsigned> &blockIndexSet,
                       const BlockIndexing &bi, const DenseSet<Block *> &deps,
                       const bool ignoreDeps) {

  // Start with a boolean expression of one
  boolean::BoolExpression *exp = boolean::BoolExpression::boolOne();

  // Cover each pair of adjacent blocks
  unsigned pathSize = path.size();
  for (unsigned i = 0; i < pathSize - 1; i++) {
    Block *firstBlock = path[i];
    Block *secondBlock = path[i + 1];

    // Skip pair if the first block has only one successor, thus no conditional
    // branch
    if (firstBlock->getSuccessors().size() == 1)
      continue;

    if (ignoreDeps || deps.contains(firstBlock)) {

      // Get last operation of the block, also called `terminator`
      Operation *terminatorOp = firstBlock->getTerminator();

      if (isa<cf::CondBranchOp>(terminatorOp)) {
        unsigned blockIndex = bi.getIndexFromBlock(firstBlock);
        std::string blockCondition = bi.getBlockCondition(firstBlock);

        // Get a boolean condition out of the block condition
        boolean::BoolExpression *pathCondition =
            boolean::BoolExpression::parseSop(blockCondition);

        // Possibly add the condition to the list of cofactors
        if (!blockIndexSet.contains(blockIndex))
          blockIndexSet.insert(blockIndex);

        // Negate the condition if `secondBlock` is reached when the condition
        // is false
        auto condOp = dyn_cast<cf::CondBranchOp>(terminatorOp);
        if (condOp.getFalseDest() == secondBlock)
          pathCondition->boolNegate();

        // And the condition with the rest path
        exp = boolean::BoolExpression::boolAnd(exp, pathCondition);
      }
    }
  }

  // Minimize the condition and return
  return exp;
}

BoolExpression *ftd::enumeratePaths(Block *start, Block *end,
                                    const BlockIndexing &bi,
                                    const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end, bi);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    DenseSet<unsigned> tempCofactorSet;
    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm =
        getPathExpression(path, tempCofactorSet, bi, controlDeps, false);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

Type ftd::channelifyType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<IndexType, IntegerType, FloatType>(
          [](auto type) { return handshake::ChannelType::get(type); })
      .Case<MemRefType>([](MemRefType memrefType) {
        if (!isa<IndexType>(memrefType.getElementType()))
          return memrefType;
        OpBuilder builder(memrefType.getContext());
        IntegerType elemType = builder.getIntegerType(32);
        return MemRefType::get(memrefType.getShape(), elemType);
      })
      .Case<handshake::ChannelType, handshake::ControlType>(
          [](auto type) { return type; })

      .Default([](auto type) { return nullptr; });
}

BoolExpression *ftd::getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                               CFGLoopInfo &li,
                                               const BlockIndexing &bi) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(bi.getBlockCondition(loopExit));
  auto *terminatorOperation = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(terminatorOperation) &&
         "Terminator condition of a loop exit must be a conditional branch.");
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);

  // If the destination of the false outcome is not the block, then the
  // condition must be negated
  if (li.getLoopFor(condBranch.getFalseDest()) != loop)
    blockCond->boolNegate();

  return blockCond;
}

SmallVector<Type> ftd::getBranchResultTypes(Type inputType) {
  SmallVector<Type> handshakeResultTypes;
  handshakeResultTypes.push_back(channelifyType(inputType));
  handshakeResultTypes.push_back(channelifyType(inputType));
  return handshakeResultTypes;
}

Block *ftd::getImmediateDominator(Region &region, Block *bb) {
  // Avoid a situation with no blocks in the region
  if (region.getBlocks().empty())
    return nullptr;
  // The first block in the CFG has both non predecessors and no dominators
  if (bb->hasNoPredecessors())
    return nullptr;
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&region);
  return domTree.getNode(bb)->getIDom()->getBlock();
}

DenseMap<Block *, DenseSet<Block *>> ftd::getDominanceFrontier(Region &region) {

  // This algorithm comes from the following paper:
  // Cooper, Keith D., Timothy J. Harvey and Ken Kennedy. “AS imple, Fast
  // Dominance Algorithm.” (1999).

  DenseMap<Block *, DenseSet<Block *>> result;

  // Create an empty set of reach available block
  for (Block &bb : region.getBlocks())
    result.insert({&bb, DenseSet<Block *>()});

  for (Block &bb : region.getBlocks()) {

    // Get the predecessors of the block
    auto predecessors = bb.getPredecessors();

    // Count the number of predecessors
    int numberOfPredecessors = 0;
    for (auto *pred : predecessors)
      if (pred)
        numberOfPredecessors++;

    // Skip if the node has none or only one predecessors
    if (numberOfPredecessors < 2)
      continue;

    // Run the algorihm as explained in the paper
    for (auto *pred : predecessors) {
      Block *runner = pred;
      // Runer performs a bottom up traversal of the dominator tree
      while (runner != getImmediateDominator(region, &bb)) {
        result[runner].insert(&bb);
        runner = getImmediateDominator(region, runner);
      }
    }
  }

  return result;
}

/// Run the cryton algorithm to determine, give a set of values, in which blocks
/// should we add a merge in order for those values to be merged
static DenseSet<Block *>
runCrytonAlgorithm(Region &funcRegion, DenseMap<Block *, Value> &inputBlocks) {
  // Get dominance frontier
  auto dominanceFrontier = ftd::getDominanceFrontier(funcRegion);

  // Temporary data structures to run the Cryton algorithm for phi positioning
  DenseMap<Block *, bool> work;
  DenseMap<Block *, bool> hasAlready;
  SmallVector<Block *> w;

  DenseSet<Block *> result;

  // Initialize data structures to run the Cryton algorithm
  for (auto &bb : funcRegion.getBlocks()) {
    work.insert({&bb, false});
    hasAlready.insert({&bb, false});
  }

  for (auto &[bb, val] : inputBlocks)
    w.push_back(bb), work[bb] = true;

  // Until there are no more elements in `w`
  while (w.size() != 0) {

    // Pop the top of `w`
    auto *x = w.back();
    w.pop_back();

    // Get the dominance frontier of `w`
    auto xFrontier = dominanceFrontier[x];

    // For each of its elements
    for (auto &y : xFrontier) {

      // Add the block in the dominance frontier to the list of blocks which
      // require a new phi. If it was not analyzed yet, also add it to `w`
      if (!hasAlready[y]) {
        result.insert(y);
        hasAlready[y] = true;
        if (!work[y])
          work[y] = true, w.push_back(y);
      }
    }
  }

  return result;
}

FailureOr<DenseMap<Block *, Value>>
ftd::createPhiNetwork(Region &funcRegion, ConversionPatternRewriter &rewriter,
                      SmallVector<Value> &vals) {

  if (vals.empty()) {
    llvm::errs() << "Input of \"createPhiNetwork\" is empty";
    return failure();
  }

  // Type of the inputs
  Type valueType = vals[0].getType();
  // All the input values associated to one block
  DenseMap<Block *, SmallVector<Value>> valuesPerBlock;
  // Associate for each block the value that is dominated by all the others in
  // the same block
  DenseMap<Block *, Value> inputBlocks;
  // Backedge builder to insert new merges
  BackedgeBuilder edgeBuilder(rewriter, funcRegion.getLoc());
  // Backedge corresponding to each phi
  DenseMap<Block *, Backedge> resultPerPhi;
  // Operands of each merge
  DenseMap<Block *, SmallVector<Value>> operandsPerPhi;
  // Which value should be the input of each input value
  DenseMap<Block *, Value> inputPerBlock;

  // Check that all the values have the same type, then collet them according to
  // their input blocks
  for (auto &val : vals) {
    if (val.getType() != valueType) {
      llvm::errs() << "All values must have the same type\n";
      return failure();
    }
    auto *bb = val.getParentBlock();
    valuesPerBlock[bb].push_back(val);
  }

  // Sort the vectors of values in each block according to their dominance and
  // get only the last input value for each block. This is necessary in case in
  // the input sets there is more than one value per blocks
  for (auto &[bb, vals] : valuesPerBlock) {
    mlir::DominanceInfo domInfo;
    std::sort(vals.begin(), vals.end(), [&](Value a, Value b) -> bool {
      if (!a.getDefiningOp())
        return true;
      if (!b.getDefiningOp())
        return false;
      return domInfo.dominates(a.getDefiningOp(), b.getDefiningOp());
    });
    inputBlocks.insert({bb, vals[vals.size() - 1]});
  }

  // In which block a new phi is necessary
  DenseSet<Block *> blocksToAddPhi =
      runCrytonAlgorithm(funcRegion, inputBlocks);

  // A backedge is created for each block in `blocksToAddPhi`, and it will
  // contain the value used as placeholder for the phi
  for (auto &bb : blocksToAddPhi) {
    Backedge mergeResult = edgeBuilder.get(valueType, bb->front().getLoc());
    operandsPerPhi.insert({bb, SmallVector<Value>()});
    resultPerPhi.insert({bb, mergeResult});
  }

  // For each phi, we need one input for every predecessor of the block
  for (auto &bb : blocksToAddPhi) {

    // Avoid to cover a predecessor twice
    llvm::DenseSet<Block *> coveredPred;
    auto predecessors = bb->getPredecessors();

    for (Block *pred : predecessors) {
      if (coveredPred.contains(pred))
        continue;
      coveredPred.insert(pred);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      Block *predecessorOrDominator = nullptr;
      Value valueToUse = nullptr;

      do {
        predecessorOrDominator =
            !predecessorOrDominator
                ? pred
                : getImmediateDominator(funcRegion, predecessorOrDominator);

        if (inputBlocks.contains(predecessorOrDominator))
          valueToUse = inputBlocks[predecessorOrDominator];
        else if (resultPerPhi.contains(predecessorOrDominator))
          valueToUse = resultPerPhi.find(predecessorOrDominator)->getSecond();

      } while (!valueToUse);

      operandsPerPhi[bb].push_back(valueToUse);
    }
  }

  // Create the merge and then replace the values
  DenseMap<Block *, handshake::MergeOp> newMergePerPhi;

  for (auto *bb : blocksToAddPhi) {
    rewriter.setInsertionPointToStart(bb);
    auto mergeOp = rewriter.create<handshake::MergeOp>(bb->front().getLoc(),
                                                       operandsPerPhi[bb]);
    mergeOp->setAttr(ftd::NEW_PHI, rewriter.getUnitAttr());
    newMergePerPhi.insert({bb, mergeOp});
  }

  for (auto *bb : blocksToAddPhi)
    resultPerPhi.find(bb)->getSecond().setValue(newMergePerPhi[bb].getResult());

  // For each block, find the incoming value of the network
  for (Block &bb : funcRegion.getBlocks()) {

    Value foundValue = nullptr;
    Block *blockOrDominator = &bb;

    if (blocksToAddPhi.contains(&bb)) {
      inputPerBlock[&bb] = newMergePerPhi[&bb].getResult();
      continue;
    }

    do {
      if (!blockOrDominator->hasNoPredecessors())
        blockOrDominator = getImmediateDominator(funcRegion, blockOrDominator);

      if (inputBlocks.contains(blockOrDominator)) {
        foundValue = inputBlocks[blockOrDominator];
      } else if (blocksToAddPhi.contains(blockOrDominator)) {
        foundValue = newMergePerPhi[blockOrDominator].getResult();
      }

    } while (!foundValue);

    inputPerBlock[&bb] = foundValue;
  }

  return inputPerBlock;
}

SmallVector<CFGLoop *> ftd::getLoopsConsNotInProd(Block *cons, Block *prod,
                                                  mlir::CFGLoopInfo &li) {
  SmallVector<CFGLoop *> result;

  // Get all the loops in which the consumer is but the producer is
  // not, starting from the innermost
  for (CFGLoop *loop = li.getLoopFor(cons); loop;
       loop = loop->getParentLoop()) {
    if (!loop->contains(prod))
      result.push_back(loop);
  }

  // Reverse to the get the loops from outermost to innermost
  std::reverse(result.begin(), result.end());
  return result;
};

LogicalResult ftd::addRegenToConsumer(ConversionPatternRewriter &rewriter,
                                      handshake::FuncOp &funcOp,
                                      Operation *consumerOp) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  auto startValue = (Value)funcOp.getArguments().back();

  // Skip if the consumer was added by this function, if it is an init merge, if
  // it comes from the explicit phi process or if it is an operation to skip
  if (consumerOp->hasAttr(FTD_REGEN) || consumerOp->hasAttr(FTD_EXPLICIT_PHI) ||
      consumerOp->hasAttr(FTD_INIT_MERGE) ||
      consumerOp->hasAttr(FTD_OP_TO_SKIP))
    return success();

  // Skip if the consumer has to do with memory operations or with che C-network
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp))
    return success();

  // Consider all the operands of the consumer
  for (Value operand : consumerOp->getOperands()) {

    mlir::Operation *producerOp = operand.getDefiningOp();

    // Skip if the producer was added by this function or if it is an op to skip
    if (producerOp &&
        (producerOp->hasAttr(FTD_REGEN) || producerOp->hasAttr(FTD_OP_TO_SKIP)))
      continue;

    // Skip if the producer has to do with memory operations
    if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(producerOp) ||
        llvm::isa_and_nonnull<MemRefType>(operand.getType()))
      continue;

    // Last regenerated value
    Value regeneratedValue = operand;

    // Get all the loops for which we need to regenerate the
    // corresponding value
    SmallVector<CFGLoop *> loops = getLoopsConsNotInProd(
        consumerOp->getBlock(), operand.getParentBlock(), loopInfo);

    auto cstType = rewriter.getIntegerType(1);
    auto cstAttr = IntegerAttr::get(cstType, 0);
    unsigned numberOfLoops = loops.size();

    // For each of the loop, from the outermost to the innermost
    for (unsigned i = 0; i < numberOfLoops; i++) {

      // If we are in the innermost loop (thus the iterator is at its end)
      // and the consumer is a loop merge, stop
      if (i == numberOfLoops - 1 && consumerOp->hasAttr(NEW_PHI))
        break;

      // Add the merge to the network, by substituting the operand with
      // the output of the merge, and forwarding the output of the merge
      // to its inputs.
      //
      rewriter.setInsertionPointToStart(loops[i]->getHeader());

      // The type of the input must be channelified
      regeneratedValue.setType(channelifyType(regeneratedValue.getType()));

      // Create an INIT merge to provide the select of the multiplexer
      auto constOp = rewriter.create<handshake::ConstantOp>(
          consumerOp->getLoc(), cstAttr, startValue);
      constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
      Value conditionValue =
          loops[i]->getExitingBlock()->getTerminator()->getOperand(0);

      SmallVector<Value> mergeOperands;
      mergeOperands.push_back(constOp.getResult());
      mergeOperands.push_back(conditionValue);
      auto initMergeOp = rewriter.create<handshake::MergeOp>(
          consumerOp->getLoc(), mergeOperands);
      initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

      // Create the multiplexer
      auto selectSignal = initMergeOp->getResult(0);
      selectSignal.setType(channelifyType(selectSignal.getType()));

      SmallVector<Value> muxOperands;
      muxOperands.push_back(regeneratedValue);
      muxOperands.push_back(regeneratedValue);

      auto muxOp = rewriter.create<handshake::MuxOp>(regeneratedValue.getLoc(),
                                                     regeneratedValue.getType(),
                                                     selectSignal, muxOperands);

      // The new producer operand is the output of the multiplxer
      regeneratedValue = muxOp.getResult();

      // Set the output of the mux as its input as well
      muxOp->setOperand(2, muxOp->getResult(0));
      muxOp->setAttr(FTD_REGEN, rewriter.getUnitAttr());
    }

    consumerOp->replaceUsesOfWith(operand, regeneratedValue);
  }

  return success();
}

std::string dynamatic::experimental::ftd::BlockIndexing::getBlockCondition(
    Block *block) const {
  return "c" + std::to_string(getIndexFromBlock(block));
}

dynamatic::experimental::ftd::BlockIndexing::BlockIndexing(Region &region) {
  mlir::DominanceInfo domInfo;

  // Create a vector with all the blocks
  SmallVector<Block *> allBlocks;
  for (Block &bb : region.getBlocks())
    allBlocks.push_back(&bb);

  // Sort the vector according to the dominance information
  std::sort(allBlocks.begin(), allBlocks.end(),
            [&](Block *a, Block *b) { return domInfo.dominates(a, b); });

  // Associate a smalled index in the map to the blocks at higer levels of the
  // dominance tree
  unsigned bbIndex = 0;
  for (Block *bb : allBlocks) {
    indexToBlock.insert({bbIndex, bb});
    blockToIndex.insert({bb, bbIndex});
    bbIndex++;
  }
}

Block *dynamatic::experimental::ftd::BlockIndexing::getBlockFromIndex(
    unsigned index) const {
  auto it = indexToBlock.find(index);
  return (it == indexToBlock.end()) ? nullptr : it->getSecond();
}

Block *dynamatic::experimental::ftd::BlockIndexing::getBlockFromCondition(
    const std::string &condition) const {
  std::string conditionNumber = condition;
  conditionNumber.erase(0, 1);
  unsigned index = std::stoi(conditionNumber);
  return this->getBlockFromIndex(index);
}

unsigned dynamatic::experimental::ftd::BlockIndexing::getIndexFromBlock(
    Block *bb) const {
  auto it = blockToIndex.find(bb);
  return (it == blockToIndex.end()) ? -1 : it->getSecond();
}

bool dynamatic::experimental::ftd::BlockIndexing::greaterIndex(
    Block *bb1, Block *bb2) const {
  return getIndexFromBlock(bb1) > getIndexFromBlock(bb2);
}

bool dynamatic::experimental::ftd::BlockIndexing::lessIndex(Block *bb1,
                                                            Block *bb2) const {
  return getIndexFromBlock(bb1) < getIndexFromBlock(bb2);
}

/// Get a value out of the input boolean expression
static Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                                   experimental::boolean::BoolExpression *expr,
                                   Block *block, const ftd::BlockIndexing &bi) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  auto condition =
      bi.getBlockFromCondition(singleCond->id)->getTerminator()->getOperand(0);
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        ftd::channelifyType(condition.getType()), condition);
    notOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());
    return notOp->getResult(0);
  }
  condition.setType(ftd::channelifyType(condition.getType()));
  return condition;
}

/// Get a circuit out a boolean expression, depending on the different kinds
/// of expressions you might have
static Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                     BoolExpression *expr, Block *block,
                                     const ftd::BlockIndexing &bi) {

  // Variable case
  if (expr->type == ExpressionType::Variable)
    return boolVariableToCircuit(rewriter, expr, block, bi);

  // Constant case (either 0 or 1)
  rewriter.setInsertionPointToStart(block);
  auto sourceOp = rewriter.create<handshake::SourceOp>(
      block->getOperations().front().getLoc());
  Value cnstTrigger = sourceOp.getResult();

  auto intType = rewriter.getIntegerType(1);
  auto cstAttr = rewriter.getIntegerAttr(
      intType, (expr->type == ExpressionType::One ? 1 : 0));

  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->getOperations().front().getLoc(), cstAttr, cnstTrigger);

  constOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return constOp.getResult();
}

/// Convert a `BDD` object as obtained from the bdd expansion to a
/// circuit
static Value bddToCircuit(ConversionPatternRewriter &rewriter, BDD *bdd,
                          Block *block, const ftd::BlockIndexing &bi) {
  if (!bdd->inputs.has_value())
    return boolExpressionToCircuit(rewriter, bdd->boolVariable, block, bi);

  rewriter.setInsertionPointToStart(block);

  // Get the two operands by recursively calling `bddToCircuit` (it possibly
  // creates other muxes in a hierarchical way)
  SmallVector<Value> muxOperands;
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().first, block, bi));
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().second, block, bi));
  Value muxCond =
      boolExpressionToCircuit(rewriter, bdd->boolVariable, block, bi);

  // Create the multiplxer and add it to the rest of the circuit
  auto muxOp = rewriter.create<handshake::MuxOp>(
      block->getOperations().front().getLoc(), muxCond, muxOperands);
  muxOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return muxOp.getResult();
}

/// Insert a branch to the correct position, taking into account whether it
/// should work to suppress the over-production of tokens or self-regeneration
static Value addSuppressionInLoop(ConversionPatternRewriter &rewriter,
                                  CFGLoop *loop, Operation *consumer,
                                  Value connection, ftd::BranchToLoopType btlt,
                                  CFGLoopInfo &li,
                                  std::vector<Operation *> &producersToCover,
                                  const ftd::BlockIndexing &bi) {

  handshake::ConditionalBranchOp branchOp;

  // Case in which there is only one termination block
  if (Block *loopExit = loop->getExitingBlock(); loopExit) {

    // Do not add the branch in case of a while loop with backward edge
    if (btlt == ftd::BackwardRelationship &&
        bi.greaterIndex(connection.getParentBlock(), loopExit))
      return connection;

    // Get the termination operation, which is supposed to be conditional
    // branch.
    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    // A conditional branch is now to be added next to the loop terminator, so
    // that the token can be suppressed
    auto *exitCondition =
        ftd::getBlockLoopExitCondition(loopExit, loop, li, bi);
    auto conditionValue =
        boolVariableToCircuit(rewriter, exitCondition, loopExit, bi);

    rewriter.setInsertionPointToStart(loopExit);

    // Since only one output is used, the other one will be connected to sink
    // in the materialization pass, as we expect from a suppress branch
    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().back().getLoc(),
        ftd::getBranchResultTypes(connection.getType()), conditionValue,
        connection);

  } else {

    std::vector<std::string> cofactorList;
    SmallVector<Block *> exitBlocks;
    loop->getExitingBlocks(exitBlocks);
    loopExit = exitBlocks.front();

    BoolExpression *fLoopExit = BoolExpression::boolZero();

    // Get the list of all the cofactors related to possible exit conditions
    for (Block *exitBlock : exitBlocks) {
      BoolExpression *blockCond =
          ftd::getBlockLoopExitCondition(exitBlock, loop, li, bi);
      fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
      cofactorList.push_back(bi.getBlockCondition(exitBlock));
    }

    // Sort the cofactors alphabetically
    std::sort(cofactorList.begin(), cofactorList.end());

    // Apply a BDD expansion to the loop exit expression and the list of
    // cofactors
    BDD *bdd = buildBDD(fLoopExit, cofactorList);

    // Convert the boolean expression obtained through bdd to a circuit
    Value branchCond = bddToCircuit(rewriter, bdd, loopExit, bi);

    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    rewriter.setInsertionPointToStart(loopExit);

    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().front().getLoc(),
        ftd::getBranchResultTypes(connection.getType()), branchCond,
        connection);
  }

  // If we are handling a case with more producers than consumers, the new
  // branch must undergo the `addSupp` function so we add it to our structure
  // to be able to loop over it
  if (btlt == ftd::MoreProducerThanConsumers) {
    branchOp->setAttr(ftd::FTD_SUPP_BRANCH, rewriter.getUnitAttr());
    producersToCover.push_back(branchOp);
  }

  Value newConnection = btlt == ftd::MoreProducerThanConsumers
                            ? branchOp.getTrueResult()
                            : branchOp.getFalseResult();

  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
static LogicalResult insertDirectSuppression(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    Operation *consumer, Value connection, const ftd::BlockIndexing &bi,
    ControlDependenceAnalysis::BlockControlDepsMap &cdAnalysis) {

  Block *entryBlock = &funcOp.getBody().front();
  Block *producerBlock = connection.getParentBlock();
  Block *consumerBlock = consumer->getBlock();

  // Get the control dependencies from the producer
  DenseSet<Block *> prodControlDeps =
      cdAnalysis[producerBlock].forwardControlDeps;

  // Get the control dependencies from the consumer
  DenseSet<Block *> consControlDeps =
      cdAnalysis[consumer->getBlock()].forwardControlDeps;

  // Get rid of common entries in the two sets
  ftd::eliminateCommonBlocks(prodControlDeps, consControlDeps);

  // Compute the activation function of producer and consumer
  BoolExpression *fProd =
      ftd::enumeratePaths(entryBlock, producerBlock, bi, prodControlDeps);
  BoolExpression *fCons =
      ftd::enumeratePaths(entryBlock, consumerBlock, bi, consControlDeps);

  // The condition related to the select signal of the consumer mux must be
  // added if the following conditions hold: The consumer is a mux; The
  // mux was a GAMMA from GSA analysis; The input of the mux (i.e., coming
  // from the producer) is a data input.
  if (ftd::isMergeOrMux(consumer) && consumer->hasAttr(ftd::FTD_EXPLICIT_PHI) &&
      consumer->getOperand(0) != connection &&
      consumer->getOperand(0).getParentBlock() != consumer->getBlock() &&
      consumer->getBlock() != producerBlock) {

    auto selectOperand = consumer->getOperand(0);
    BoolExpression *selectOperandCondition = BoolExpression::parseSop(
        bi.getBlockCondition(selectOperand.getDefiningOp()->getBlock()));

    // The condition must be taken into account for `fCons` only if the
    // producer is not control dependent from the block which produces the
    // condition of the mux
    if (!prodControlDeps.contains(selectOperand.getParentBlock())) {
      if (consumer->getOperand(1) == connection)
        fCons = BoolExpression::boolAnd(fCons,
                                        selectOperandCondition->boolNegate());
      else
        fCons = BoolExpression::boolAnd(fCons, selectOperandCondition);
    }
  }

  /// f_supp = f_prod and not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fSup, cofactorList);
    Value branchCond = bddToCircuit(rewriter, bdd, consumer->getBlock(), bi);

    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), ftd::getBranchResultTypes(connection.getType()),
        branchCond, connection);
    consumer->replaceUsesOfWith(connection, branchOp.getFalseResult());
  }

  return success();
}

LogicalResult
ftd::addSuppToProducer(ConversionPatternRewriter &rewriter,
                       handshake::FuncOp &funcOp, Operation *producerOp,
                       const ftd::BlockIndexing &bi,
                       std::vector<Operation *> &producersToCover,
                       ControlDependenceAnalysis::BlockControlDepsMap &cda) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  Block *producerBlock = producerOp->getBlock();

  // Skip the prod-cons if the producer is part of the operations related to
  // the BDD expansion or INIT merges
  if (producerOp->hasAttr(ftd::FTD_OP_TO_SKIP) ||
      producerOp->hasAttr(ftd::FTD_INIT_MERGE))
    return success();

  // Consider all the consumers of each value of the producer
  for (Value result : producerOp->getResults()) {

    std::vector<Operation *> users(result.getUsers().begin(),
                                   result.getUsers().end());
    users.erase(unique(users.begin(), users.end()), users.end());

    for (Operation *consumerOp : users) {
      Block *consumerBlock = consumerOp->getBlock();

      // If the consumer and the producer are in the same block without the
      // consumer being a multiplxer skip because no delivery is needed
      if (consumerBlock == producerBlock && !isMergeOrMux(consumerOp))
        continue;

      // Skip the prod-cons if the consumer is part of the operations
      // related to the BDD expansion or INIT merges
      if (consumerOp->hasAttr(ftd::FTD_OP_TO_SKIP) ||
          consumerOp->hasAttr(ftd::FTD_INIT_MERGE))
        continue;

      // TODO: Group the conditions of memory and the conditions of Branches
      // in 1 function?
      // Skip if either the producer of the consumer are
      // related to memory operations, or if the consumer is a conditional
      // branch
      if (llvm::isa_and_nonnull<handshake::MemoryControllerOp>(consumerOp) ||
          llvm::isa_and_nonnull<handshake::MemoryControllerOp>(producerOp) ||
          llvm::isa_and_nonnull<handshake::LSQOp>(producerOp) ||
          llvm::isa_and_nonnull<handshake::LSQOp>(consumerOp) ||
          llvm::isa_and_nonnull<handshake::ControlMergeOp>(producerOp) ||
          llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
          llvm::isa<handshake::ConditionalBranchOp>(consumerOp) ||
          llvm::isa<cf::CondBranchOp>(consumerOp) ||
          llvm::isa<cf::BranchOp>(consumerOp) ||
          (llvm::isa<memref::LoadOp>(consumerOp) &&
           !llvm::isa<handshake::LSQLoadOp>(consumerOp)) ||
          (llvm::isa<memref::StoreOp>(consumerOp) &&
           !llvm::isa<handshake::LSQStoreOp>(consumerOp)) ||
          (llvm::isa<memref::LoadOp>(consumerOp) &&
           !llvm::isa<handshake::MCLoadOp>(consumerOp)) ||
          (llvm::isa<memref::StoreOp>(consumerOp) &&
           !llvm::isa<handshake::MCStoreOp>(consumerOp)) ||
          llvm::isa<mlir::MemRefType>(result.getType()))
        continue;

      // The next step is to identify the relationship between the producer
      // and consumer in hand: Are they in the same loop or at different
      // loop levels? Are they connected through a bwd edge?

      // Set true if the producer is in a loop which does not contains
      // the consumer
      bool producingGtUsing =
          loopInfo.getLoopFor(producerBlock) &&
          !loopInfo.getLoopFor(producerBlock)->contains(consumerBlock);

      auto *consumerLoop = loopInfo.getLoopFor(consumerBlock);

      // Set to true if the consumer uses its own result
      bool selfRegeneration =
          llvm::any_of(consumerOp->getResults(),
                       [&result](const Value &v) { return v == result; });

      // We need to suppress all the tokens produced within a loop and
      // used outside each time the loop is not terminated. This should be
      // done for as many loops there are
      if (producingGtUsing && !ftd::isBranchLoopExit(producerOp, loopInfo)) {
        Value con = result;
        for (CFGLoop *loop = loopInfo.getLoopFor(producerBlock); loop;
             loop = loop->getParentLoop()) {

          // For each loop containing the producer but not the consumer, add
          // the branch
          if (!loop->contains(consumerBlock))
            con = addSuppressionInLoop(rewriter, loop, consumerOp, con,
                                       ftd::MoreProducerThanConsumers, loopInfo,
                                       producersToCover, bi);
        }
      }

      // We need to suppress a token if the consumer is the producer itself
      // within a loop
      else if (selfRegeneration && consumerLoop &&
               !producerOp->hasAttr(ftd::FTD_SUPP_BRANCH)) {
        addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                             ftd::SelfRegeneration, loopInfo, producersToCover,
                             bi);
      }

      // We need to suppress a token if the consumer comes before the
      // producer (backward edge)
      else if ((bi.greaterIndex(producerBlock, consumerBlock) ||
                (isMergeOrMux(consumerOp) && producerBlock == consumerBlock &&
                 ftd::isaMergeLoop(consumerOp, loopInfo))) &&
               consumerLoop) {
        addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                             ftd::BackwardRelationship, loopInfo,
                             producersToCover, bi);
      }

      // If no loop is involved, then there is a direct relationship between
      // consumer and producer
      else if (failed(insertDirectSuppression(rewriter, funcOp, consumerOp,
                                              result, bi, cda)))
        return failure();
    }
  }

  // Once that we have considered all the consumers of the results of a
  // producer, we consider the operands of the producer. Some of these
  // operands might be the arguments of the functions, and these might need
  // to be suppressed as well.

  // Do not take into account conditional branch
  if (llvm::isa<handshake::ConditionalBranchOp>(producerOp))
    return success();

  // For all the operands of the operation, take into account only the
  // start value if exists
  for (Value operand : producerOp->getOperands()) {
    // The arguments of a function do not have a defining operation
    if (operand.getDefiningOp())
      continue;

    // Skip if we are in block 0 and no multiplexer is involved
    if (operand.getParentBlock() == producerBlock && !isMergeOrMux(producerOp))
      continue;

    // Handle the suppression
    if (failed(insertDirectSuppression(rewriter, funcOp, producerOp, operand,
                                       bi, cda)))
      return failure();
  }

  return success();
}

bool ftd::isMergeOrMux(Operation *op) {
  return llvm::isa_and_nonnull<handshake::MergeOp>(op) ||
         llvm::isa_and_nonnull<handshake::MuxOp>(op);
}

bool ftd::CFGEdge::isConditional() const { return conditionName.has_value(); }

bool ftd::CFGEdge::isUnconditional() const {
  return !conditionName.has_value();
}

unsigned ftd::CFGEdge::getSuccessor() const {
  return isUnconditional() ? std::get<unsigned>(successors) : -1;
}
unsigned ftd::CFGEdge::getTrueSuccessor() const {
  return isConditional()
             ? std::get<std::pair<unsigned, unsigned>>(successors).first
             : -1;
}
unsigned ftd::CFGEdge::getFalseSuccessor() const {
  return isConditional()
             ? std::get<std::pair<unsigned, unsigned>>(successors).second
             : -1;
}
std::string ftd::CFGEdge::getCondition() const {
  return isConditional() ? conditionName.value() : "";
}
void ftd::CFGEdge::print() const {
  if (isConditional()) {
    llvm::dbgs() << "{ " << getTrueSuccessor() << " " << getFalseSuccessor()
                 << " " << conditionName.value() << " }\n";
  } else {
    llvm::dbgs() << "{ " << getSuccessor() << " }\n";
  }
}

std::string
ftd::CFGEdge::serializeEdges(const DenseMap<unsigned, ftd::CFGEdge> &edgeMap) {
  return "";
}

DenseMap<unsigned, ftd::CFGEdge>
ftd::CFGEdge::unserializeEdges(const std::string &edges) {
  return DenseMap<unsigned, ftd::CFGEdge>();
}
