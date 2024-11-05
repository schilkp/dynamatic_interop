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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic::experimental::boolean;

namespace dynamatic {
namespace experimental {
namespace ftd {

int getBlockIndex(Block *bb) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  bb->printAsOperand(os1);
  std::string block1id = os1.str();
  return std::stoi(block1id.substr(3));
}

bool lessThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) < getBlockIndex(block2);
}

bool greaterThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) > getBlockIndex(block2);
}

bool isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || isSameLoop(loop1->getParentLoop(), loop2) ||
          isSameLoop(loop1, loop2->getParentLoop()) ||
          isSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li) {
  return isSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

std::string getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(ftd::getBlockIndex(block));
  return blockCondition;
}

bool isHandhsakeLSQOperation(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

void eliminateCommonBlocks(DenseSet<Block *> &s1, DenseSet<Block *> &s2) {

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

/// Helper recursive function to get the innermost common loop
static CFGLoop *checkInnermostCommonLoop(CFGLoop *loop1, CFGLoop *loop2) {

  // None of them is a loop
  if (!loop1 || !loop2)
    return nullptr;

  // Same loop
  if (loop1 == loop2)
    return loop1;

  // Check whether the parent loop of `loop1` is `loop2`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop1->getParentLoop(), loop2); pl)
    return pl;

  // Check whether the parent loop of `loop2` is `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(), loop1); pl)
    return pl;

  // Check whether the parent loop of `loop1` is identical to the parent loop
  // of `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(),
                                             loop1->getParentLoop());
      pl)
    return pl;

  // If no common loop is found, return nullptr
  return nullptr;
}

CFGLoop *getInnermostCommonLoop(Block *block1, Block *block2,
                                mlir::CFGLoopInfo &li) {
  return checkInnermostCommonLoop(li.getLoopFor(block1), li.getLoopFor(block2));
}

bool isBranchLoopExit(Operation *op, CFGLoopInfo &li) {
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
        if (toAvoid == successor &&
            getBlockIndex(toAvoid) > getBlockIndex(blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      if (visited.find(successor) == visited.end())
        dfsAllPaths(successor, end, path, visited, allPaths, blockToTraverse,
                    blocksToAvoid, blockFound || blockToTraverseFound);
    }
  }

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

/// Recursive function which allows to obtain all the paths from operation
/// `start` to operation `end` using a DFS
static void dfsAllPaths(Operation *current, Operation *end,
                        std::unordered_set<Operation *> &visited,
                        std::vector<Operation *> &path,
                        std::vector<std::vector<Operation *>> &allPaths) {
  visited.insert(current);
  path.push_back(current);

  if (current == end) {
    // If the current operation is the end, add the path to allPaths
    allPaths.push_back(path);
  } else {
    // Otherwise, explore the successors
    for (auto result : current->getResults()) {
      for (auto *successor : result.getUsers()) {
        if (visited.find(successor) == visited.end()) {
          dfsAllPaths(successor, end, visited, path, allPaths);
        }
      }
    }
  }

  // Backtrack
  path.pop_back();
  visited.erase(current);
}

std::vector<std::vector<Operation *>> findAllPaths(Operation *start,
                                                   Operation *end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::unordered_set<Operation *> visited;
  std::vector<Operation *> path;
  dfsAllPaths(start, end, visited, path, allPaths);
  return allPaths;
}

std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, Block *blockToTraverse,
             const std::vector<Block *> &blocksToAvoid) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, blockToTraverse,
              blocksToAvoid, false);
  return allPaths;
}

/// Helper recursive function for getPostDominantSuccessor
static Block *getPostDominantSuccessor(Block *prod, Block *cons,
                                       std::unordered_set<Block *> &visited,
                                       PostDominanceInfo &postDomInfo) {

  // If the producer is not valid, return, otherwise insert it among the
  // visited ones.
  if (!prod)
    return nullptr;

  visited.insert(prod);

  // For each successor of the producer
  for (Block *successor : prod->getSuccessors()) {

    // Check if the successor post-dominates cons
    if (successor != cons && postDomInfo.postDominates(successor, cons))
      return successor;

    // If not visited, recursively search successors of the current successor
    if (visited.find(successor) == visited.end()) {
      Block *result =
          getPostDominantSuccessor(successor, cons, visited, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

Block *getPostDominantSuccessor(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  PostDominanceInfo postDomInfo;
  return getPostDominantSuccessor(prod, cons, visited, postDomInfo);
}

/// Helper recursive function for getPredecessorDominatingAndPostDominating
static Block *getPredecessorDominatingAndPostDominating(
    Block *producer, Block *consumer, std::unordered_set<Block *> &visited,
    DominanceInfo &domInfo, PostDominanceInfo &postDomInfo) {

  // If the consumer is not valid, return, otherwise insert it in the visited
  // ones
  if (!consumer)
    return nullptr;
  visited.insert(consumer);

  // For each predecessor of the consumer
  for (Block *predecessor : consumer->getPredecessors()) {

    // If the current predecessor is not the producer itself, and this block
    // both dominates the consumer and post-dominates the producer, return it
    if (predecessor != producer &&
        postDomInfo.postDominates(predecessor, producer) &&
        domInfo.dominates(predecessor, consumer))
      return predecessor;

    // If not visited, recursively search predecessors of the current
    // predecessor
    if (visited.find(predecessor) == visited.end()) {
      Block *result = getPredecessorDominatingAndPostDominating(
          producer, predecessor, visited, domInfo, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

Block *getPredecessorDominatingAndPostDominating(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  return getPredecessorDominatingAndPostDominating(prod, cons, visited, domInfo,
                                                   postDomInfo);
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool isaMergeLoop(Operation *merge, CFGLoopInfo &li) {

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
getPathExpression(const std::vector<Block *> &path,
                  std::vector<std::string> &cofactorList,
                  const DenseSet<Block *> &deps, const bool ignoreDeps) {

  // Start with a boolean expression of one
  boolean::BoolExpression *exp = boolean::BoolExpression::boolOne();

  // Cover each pair of adjacent blocks
  for (int i = 0; i < (int)path.size() - 1; i++) {
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
        auto blockCondition = getBlockCondition(firstBlock);

        // Get a boolean condition out of the block condition
        boolean::BoolExpression *pathCondition =
            boolean::BoolExpression::parseSop(blockCondition);

        // Possibly add the condition to the list of cofactors
        if (std::find(cofactorList.begin(), cofactorList.end(),
                      blockCondition) == cofactorList.end())
          cofactorList.push_back(blockCondition);

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

BoolExpression *enumeratePaths(Block *start, Block *end,
                               const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    std::vector<std::string> tempCofactorList;
    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm =
        getPathExpression(path, tempCofactorList, controlDeps, false);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

Type channelifyType(Type type) {
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

BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                          CFGLoopInfo &li) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(getBlockCondition(loopExit));
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

SmallVector<Type> getBranchResultTypes(Type inputType) {
  SmallVector<Type> handshakeResultTypes;
  handshakeResultTypes.push_back(channelifyType(inputType));
  handshakeResultTypes.push_back(channelifyType(inputType));
  return handshakeResultTypes;
}

Block *getImmediateDominator(Region &region, Block *bb) {
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

DenseMap<Block *, DenseSet<Block *>> getDominanceFrontier(Region &region) {

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

#define PRINT_DEBUG
#ifdef PRINT_DEBUG
  for (auto &entry : result) {
    llvm::dbgs() << "[DOM FRONT] Domination frontier of ";
    entry.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ": ";
    llvm::dbgs() << "{";
    for (auto &dom : entry.second) {
      dom->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " ";
    }
    llvm::dbgs() << "}\n";
  }
#endif

  return result;
}

FailureOr<DenseMap<Block *, Value>>
insertPhi(Region &funcRegion, ConversionPatternRewriter &rewriter,
          SmallVector<Value> &vals) {

  auto dominanceFrontier = getDominanceFrontier(funcRegion);

  // The number of values to be considered cannot be empty
  if (vals.empty())
    return funcRegion.getParentOp()->emitError()
           << "The number values provided in `insertPhi` "
              "must be larger than 2\n";

  // All the values provided must have the same type.
  // As an additional constraint, all the values should be in a different basic
  // block
  DenseSet<Block *> foundBlocks;
  for (auto &val : vals) {
    if (val.getType() != vals[0].getType())
      return funcRegion.getParentOp()->emitError()
             << "The values provided to `addPhi` do not all have the same type";
    if (foundBlocks.contains(val.getParentBlock()))
      return funcRegion.getParentOp()->emitError()
             << "Some of the values provided to `addPhi` "
                "belong to the same basic block";
    foundBlocks.insert(val.getParentBlock());
  }

  llvm::dbgs() << "[NEW PHI] Producers in: {";
  for (auto &val : vals) {
    val.getParentBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " ";
  }
  llvm::dbgs() << "}\n";

  // Temporary data structures to run the Cryton algorithm for phi positioning
  DenseMap<Block *, bool> work;
  DenseMap<Block *, bool> hasAlready;
  SmallVector<Block *> w;

  // Initialize data structures
  for (auto &bb : funcRegion.getBlocks()) {
    work.insert({&bb, false});
    hasAlready.insert({&bb, false});
  }

  for (auto val : vals) {
    w.push_back(val.getParentBlock());
    work[val.getParentBlock()] = true;
  }

  // This vector ends up containig the blocks in which a new argument is to be
  // added
  DenseSet<Block *> blocksToAddPhi;

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
        blocksToAddPhi.insert(y);
        hasAlready[y] = true;
        if (!work[y])
          work[y] = true, w.push_back(y);
      }
    }
  }

  // Get a location to insert the new phis
  auto loc = UnknownLoc::get(vals[0].getContext());
  for (auto &bb : blocksToAddPhi)
    bb->addArgument(vals[0].getType(), loc);

  llvm::dbgs() << "[NEW PHI] Insertion in { ";
  for (auto &bb : blocksToAddPhi) {
    bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " ";
  }
  llvm::dbgs() << "}\n";

  if (blocksToAddPhi.empty())
    return success();

  // Since a new block argument was added for each block in `blocksToAddPhi`,
  // new values must be provided to them together with the branches (either
  // conditional or non-conditional). The values might come from one of the
  // input operations or from another of the added block arguments. In order to
  // find the correct value, we first anlayzed the predecessor of each node: if
  // it has a redefinition of the value, then we use it, otherwise we move the
  // anlaysis to its immediate dominator. Since a definition of the value must
  // always exist, BB0 must define the value as well.

  for (auto &bb : blocksToAddPhi) {

    // For each bb in `blocksToAddPhi`, we need to modify the terminator of each
    // of its predecessors
    auto predecessors = bb->getPredecessors();

    for (auto *pred : predecessors) {

      auto *terminator = pred->getTerminator();
      rewriter.setInsertionPointAfter(terminator);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      auto *predecessorOrDominator = pred;

      Value valueToUse = nullptr;

      while (valueToUse == nullptr) {

        // For each of the values provided as input
        for (auto &val : vals) {

          // If the block of the current `predecessorOrDominator` contains a
          // definition of the value, then we use it in the terminator
          if (val.getParentBlock() == predecessorOrDominator) {
            valueToUse = val;
            break;
          }
        }

        if (valueToUse == nullptr) {
          // Go through the blocks having a new arugment for the value
          for (auto &phibb : blocksToAddPhi) {
            if (predecessorOrDominator == phibb) {
              valueToUse = phibb->getArgument(phibb->getNumArguments() - 1);
              break;
            }
          }
        }

        if (valueToUse) {

          // Case in which the terminator is a branch
          if (llvm::isa_and_nonnull<cf::BranchOp>(terminator)) {
            auto branch = cast<cf::BranchOp>(terminator);
            SmallVector<Value> operands = branch.getDestOperands();
            operands.push_back(valueToUse);
            auto newBranch = rewriter.create<cf::BranchOp>(
                branch->getLoc(), branch.getDest(), operands);
            rewriter.replaceOp(branch, newBranch);
          }

          // Case in which the terminator is a conditional branch
          if (llvm::isa_and_nonnull<cf::CondBranchOp>(terminator)) {
            auto branch = cast<cf::CondBranchOp>(terminator);
            SmallVector<Value> trueOperands = branch.getTrueDestOperands();
            SmallVector<Value> falseOperands = branch.getFalseDestOperands();

            if (branch.getTrueDest() == bb)
              trueOperands.push_back(valueToUse);
            else
              falseOperands.push_back(valueToUse);

            auto newBranch = rewriter.create<cf::CondBranchOp>(
                branch->getLoc(), branch.getCondition(), branch.getTrueDest(),
                trueOperands, branch.getFalseDest(), falseOperands);
            rewriter.replaceOp(branch, newBranch);
            break;
          }
        }

        // Terminate if the value was found
        if (valueToUse != nullptr ||
            predecessorOrDominator->hasNoPredecessors())
          break;

        // Move to the immediate dominator
        predecessorOrDominator =
            getImmediateDominator(funcRegion, predecessorOrDominator);
      }

      if (!valueToUse)
        return funcRegion.getParentOp()->emitError()
               << "A branch could not be modified, because no definition of "
                  "the value was found\n";
    }
  }

  DenseMap<Block *, Value> result;

  for (auto &bb : funcRegion.getBlocks()) {

    if (blocksToAddPhi.contains(&bb)) {
      result.insert({&bb, bb.getArgument(bb.getNumArguments() - 1)});
      continue;
    }

    auto predecessors = bb.getPredecessors();

    for (auto *pred : predecessors) {

      auto *terminator = pred->getTerminator();
      rewriter.setInsertionPointAfter(terminator);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      auto *predecessorOrDominator = pred;

      Value valueToUse = nullptr;

      while (valueToUse == nullptr) {

        // For each of the values provided as input
        for (auto &val : vals) {

          // If the block of the current `predecessorOrDominator` contains a
          // definition of the value, then we use it in the terminator
          if (val.getParentBlock() == predecessorOrDominator) {
            valueToUse = val;
            break;
          }
        }

        if (valueToUse == nullptr) {
          // Go through the blocks having a new arugment for the value
          for (auto &phibb : blocksToAddPhi) {
            if (predecessorOrDominator == phibb) {
              valueToUse = phibb->getArgument(phibb->getNumArguments() - 1);
              break;
            }
          }
        }

        if (valueToUse)
          result.insert({&bb, valueToUse});

        // Terminate if the value was found
        if (valueToUse != nullptr ||
            predecessorOrDominator->hasNoPredecessors())
          break;

        // Move to the immediate dominator
        predecessorOrDominator =
            getImmediateDominator(funcRegion, predecessorOrDominator);
      }

      if (!valueToUse)
        return funcRegion.getParentOp()->emitError()
               << "Cannot find defintion of a value for a block";
    }
  }

#define PRINT_DEBUG
#ifdef PRINT_DEBUG
  for (auto &entry : result) {
    llvm::dbgs() << "[PHI RESULT] In ";
    entry.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " value is ";
    entry.second.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
#endif

  return result;
}

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic
