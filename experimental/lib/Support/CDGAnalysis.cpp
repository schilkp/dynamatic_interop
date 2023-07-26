//===- CDGAnalysis.cpp - Exp. support for CDG analysis -----*- C++ -*-===//
//
// This file contains the function for CDG analysis.
//
//===-----------------------------------------------------------------===//

#include <set>
#include <stack>
#include <vector>

#include "mlir/Analysis/CFGLoopInfo.h"

#include "experimental/Support/CDGAnalysis.h"

using namespace dynamatic::experimental;

namespace {

/// Helper struct for Control Dependence Graph (CDG) analysis that
/// represents a directed edge (A, B) in the Control Flow Graph (CFG).
struct CFGEdge {
  /// Node A of the CFG edge (A, B).
  DominanceInfoNode *a;
  /// Node B of the CFG edge (A, B).
  DominanceInfoNode *b;

  CFGEdge(DominanceInfoNode *a, DominanceInfoNode *b) : a(a), b(b) {}

  /// Finds the Least Common Ancestor (LCA) in the Post-Dominator Tree
  /// for nodes A and B of a CFG edge (A, B).
  DominanceInfoNode *findLCAInPostDomTree() {
    std::set<DominanceInfoNode *> ancestorsA;

    // Traverse upwards from node A and store its ancestors in the set.
    unsigned level = a->getLevel();
    for (DominanceInfoNode *curr = a; level > 0;
         --level, curr = curr->getIDom())
      ancestorsA.insert(curr);

    // Traverse upwards from node B until a common ancestor is found.
    level = b->getLevel();
    for (DominanceInfoNode *curr = b; level > 0;
         --level, curr = curr->getIDom())
      if (ancestorsA.find(curr) != ancestorsA.end())
        // Lowest common ancestor
        return curr;

    return nullptr;
  }
};

/// Traversal of the Control Flow Graph (CFG) that creates a set of graph
/// edges (A, B) where A is NOT post-dominated by B.
void cfgTraversal(Block &rootBlock, std::vector<CFGEdge> &edges,
                  PostDominanceInfo &postDomInfo,
                  llvm::DominatorTreeBase<Block, true> &postDomTree) {
  std::set<Block *> visitedSet;
  std::stack<Block *> blockStack;

  blockStack.push(&rootBlock);

  while (!blockStack.empty()) {
    Block *currBlock = blockStack.top();
    blockStack.pop();

    // Mark current node as visited.
    visitedSet.insert(currBlock);

    // end of visit

    for (Block *successor : currBlock->getSuccessors()) {
      // Check if curr is not post-dominated by his successor.
      if (postDomInfo.properlyPostDominates(successor, currBlock)) {
        // Every node must be marked as visited for the CDG construction.
        visitedSet.insert(successor);
        continue;
      }

      // Form the edge struct and add it to the set.
      DominanceInfoNode *succPostDomNode = postDomTree.getNode(successor);
      DominanceInfoNode *currPostDomNode = postDomTree.getNode(currBlock);

      CFGEdge edge(currPostDomNode, succPostDomNode);
      edges.push_back(edge);

      // Check if successor is already visited.
      if (visitedSet.find(successor) != visitedSet.end())
        continue;

      blockStack.push(successor);
    }
  } // end while
}

DenseMap<Block *, BlockLoopInfo> findLoopDetails(func::FuncOp &funcOp) {
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree =
      domInfo.getDomTree(&funcOp.getRegion());
  CFGLoopInfo li(domTree);

  // Finding all loops.
  std::vector<CFGLoop *> loops;
  Region &funcReg = funcOp.getRegion();
  for (auto &block : funcReg.getBlocks()) {
    CFGLoop *loop = li.getLoopFor(&block);

    while (loop) {
      auto pos = std::find(loops.begin(), loops.end(), loop);
      if (pos == loops.end()) {
        llvm::outs() << "Found a loop!\n";
        loops.push_back(loop);
      }
      loop = loop->getParentLoop();
    }
  }

  // Iterating over BB of each loop, and attaching loop info.
  DenseMap<Block *, BlockLoopInfo> blockToLoopInfoMap;
  for (auto &block : funcReg.getBlocks()) {
    BlockLoopInfo bli;
    blockToLoopInfoMap.insert(std::make_pair(&block, bli));
  }

  for (auto &loop : loops) {
    Block *loopHeader = loop->getHeader();
    blockToLoopInfoMap[loopHeader].isHeader = true;
    blockToLoopInfoMap[loopHeader].loop = loop;

    llvm::outs() << "Loop header: ";
    loopHeader->printAsOperand(llvm::outs());
    llvm::outs() << "\n";

    llvm::SmallVector<Block *> exitBlocks;
    loop->getExitingBlocks(exitBlocks);
    for (auto &block : exitBlocks) {
      blockToLoopInfoMap[block].isExit = true;
      blockToLoopInfoMap[block].loop = loop;

      llvm::outs() << "Loop exit: ";
      block->printAsOperand(llvm::outs());
      llvm::outs() << "\n";
    }

    // A latch block is a block that contains a branch back to the header.
    llvm::SmallVector<Block *> latchBlocks;
    loop->getLoopLatches(latchBlocks);
    for (auto &block : latchBlocks) {
      blockToLoopInfoMap[block].isLatch = true;
      blockToLoopInfoMap[block].loop = loop;

      llvm::outs() << "Loop latch: ";
      block->printAsOperand(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  return blockToLoopInfoMap;
}

} // namespace

DenseMap<Block *, BlockNeighbors>
dynamatic::experimental::cdgAnalysis(func::FuncOp &funcOp, MLIRContext &ctx) {
  DenseMap<Block *, BlockNeighbors> cdg;

  Region &funcReg = funcOp.getRegion();

  for (Block &block : funcReg.getBlocks()) {
    BlockNeighbors bn;
    cdg.insert(std::make_pair(&block, bn));
  }

  // Handling single block regions, cannot get DomTree for single block regions.
  if (funcReg.hasOneBlock())
    return cdg;

  Block &rootBlockCFG = funcReg.getBlocks().front();
  // Get the data structure containg information about post-dominance.
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Find set of CFG edges (A,B) so A is NOT post-dominated by B.
  std::vector<CFGEdge> edges;
  cfgTraversal(rootBlockCFG, edges, postDomInfo, postDomTree);

  for (CFGEdge &edge : edges) {
    // Find the least common ancesstor (LCA) in post-dominator tree of A and B
    // for each CFG edge (A,B).
    DominanceInfoNode *lca = edge.findLCAInPostDomTree();

    Block *controlBlock = edge.a->getBlock();

    // LCA = parent(A) or LCA = A.
    // All nodes on the path (LCA, B] are control dependent on A.
    for (DominanceInfoNode *curr = edge.b; curr != lca;
         curr = curr->getIDom()) {
      Block *dependentBlock = curr->getBlock();

      cdg[controlBlock].successors.push_back(dependentBlock);
      cdg[dependentBlock].predecessors.push_back(controlBlock);
    }
  }

  findLoopDetails(funcOp);

  return cdg;
} // CDGAnalysis end
