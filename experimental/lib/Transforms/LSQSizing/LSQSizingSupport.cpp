#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"


#include <unordered_set>
#include <stack>
#include <set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;

int extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB) {
  double latency = 0;

  if(!failed(timingDB.getLatency(op, SignalType::DATA, latency)))
    return latency;

  if(op->getName().getStringRef() == "handshake.buffer") { // TODO use some build in class method instead of name?
    auto params = op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!params) {
      llvm::dbgs() << "BufferOp" << op->getAttrOfType<StringAttr>("handshake.name").str() << " does not have parameters\n";
    }

    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (!optTiming) {
      llvm::dbgs() << "BufferOp" << op->getAttrOfType<StringAttr>("handshake.name").str() << " does not have timing\n";
    }

    if (auto timing =dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      handshake::TimingInfo info = timing.getInfo();
      latency = info.getLatency(SignalType::DATA).value_or(0);
    }   
  }
  else if(op->getName().getStringRef() == "handshake.muli") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 4\n";
    latency = 4;
  }
  else if(op->getName().getStringRef() == "handshake.mulf") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 6\n";
    latency = 6;
  }
  else if(op->getName().getStringRef() == "handshake.addf") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 10\n";
    latency = 10;
  } 
  else if(op->getName().getStringRef() == "handshake.subf") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 10\n";
    latency = 10;
  } 
  else if(op->getName().getStringRef() == "handshake.divui") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 36\n";
    latency = 36;
  }
  else if(op->getName().getStringRef() == "handshake.divsi") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 36\n";
    latency = 36;
  } 
  else if(op->getName().getStringRef() == "handshake.divf") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 30\n";
    latency = 30;
  } 
  else if(op->getName().getStringRef() == "handshake.cmpf") { // TODO Remove Hardcoding once this is fixed in the handshake dialect
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency, using hardcoded latency of 2\n";
    latency = 2;
  }
  else {
    llvm::dbgs() << "Operation " << op->getName().getStringRef() << " does not have latency\n";
    latency = 0;
  }
  
  return latency;
}

AdjListGraph::AdjListGraph(handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs, TimingDatabase timingDB, unsigned II) {

  for (Operation &op : funcOp.getOps()) {
    // Get operation's basic block
    unsigned srcBB;
    if (auto optBB = getLogicBB(&op); !optBB.has_value())
      continue;
    else
      srcBB = *optBB;

    // The basic block the operation belongs to must be selected
    if (!cfdfcBBs.contains(srcBB))
      continue;

    // Add the unit and valid outgoing channels to the CFDFC
    addNode(&op, extractNodeLatency(&op, timingDB));

    for (OpResult res : op.getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      // Get the value's unique user and its basic block
      Operation *user = *res.getUsers().begin();
      unsigned dstBB;
      if (std::optional<unsigned> optBB = getLogicBB(user); !optBB.has_value())
        continue;
      else
        dstBB = *optBB;

      if (srcBB != dstBB) {
        // The channel is in the CFDFC if it belongs belong to a selected arch
        // between two basic blocks
        for (size_t i = 0; i < cfdfcBBs.size(); ++i) {
          unsigned nextBB = i == cfdfcBBs.size() - 1 ? 0 : i + 1;
          if (srcBB == cfdfcBBs[i] && dstBB == cfdfcBBs[nextBB]) {
            addChannelEdges(res);
            if (buffer::CFDFC::isCFDFCBackedge(res))
              addChannelBackedges(res, (II * -1));
            break;
          }
        }
      } else if (cfdfcBBs.size() == 1) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the CFDFC is just a block looping to itself
            addChannelEdges(res);
            if (buffer::CFDFC::isCFDFCBackedge(res))
              addChannelBackedges(res, (II * -1));
      } else if (!isBackedge(res)) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the channel is not a backedge
        addChannelEdges(res);
      }
    }
  }

  
}



void AdjListGraph::addNode(mlir::Operation *op, int latency) {
    nodes.insert({op->getAttrOfType<StringAttr>("handshake.name").str(), AdjListNode{latency, op, {}, {}}});
}

void AdjListGraph::addEdge(mlir::Operation * src, mlir::Operation * dest) {
    nodes.at(src->getAttrOfType<StringAttr>("handshake.name").str()).edges.push_back(dest->getAttrOfType<StringAttr>("handshake.name").str()); // Add edge from node u to node v
}

void AdjListGraph::addBackedge(mlir::Operation * src, mlir::Operation * dest) {
    nodes.at(src->getAttrOfType<StringAttr>("handshake.name").str()).backedges.push_back(dest->getAttrOfType<StringAttr>("handshake.name").str()); // Add edge from node u to node v
}

void AdjListGraph::addChannelEdges(mlir::Value res) {
    mlir::Operation *srcOp = res.getDefiningOp();
    for(Operation *destOp: res.getUsers()) {
      addEdge(srcOp, destOp);
    }
}

void AdjListGraph::addChannelBackedges(mlir::Value res, int latency) {
    mlir::Operation *srcOp = res.getDefiningOp();
    for(Operation *destOp: res.getUsers()) {
      insertArtificialNodeOnBackedge(srcOp, destOp, latency);
    }
}


void AdjListGraph::printGraph() {
    for (const auto& pair : nodes) {
        std::string opName = pair.first;
        const AdjListNode& node = pair.second;
        llvm::dbgs()  << opName << " (lat: " << node.latency << "): ";
        for (std::string edge : node.edges) {
            llvm::dbgs() << edge << ", ";
        }
        if(node.backedges.size() > 0) {
          llvm::dbgs() << " || ";
          for(std::string backedge : node.backedges) {
              llvm::dbgs() << backedge << ", ";
          }
        }
        llvm::dbgs() << "\n";
    }
}

void AdjListGraph::printPath(std::vector<std::string> path) {
    for (std::string node : path) {
        llvm::dbgs() << node << "(" << nodes.at(node).latency << ") - ";
    }
    llvm::dbgs() << "\n";
}

void AdjListGraph::insertArtificialNodeOnBackedge(mlir::Operation* src, mlir::Operation* dest, int latency) {
  // create new node name from src and dest name
  std::string srcName = src->getAttrOfType<StringAttr>("handshake.name").str();
  std::string destName = dest->getAttrOfType<StringAttr>("handshake.name").str();
  std::string newNodeName = "backedge_" + srcName + "_" + destName;

  //remove regular edge from src to dest
  nodes.at(srcName).edges.remove(destName);

  // create node and add edge from src to new node and new node to dest
  nodes.insert({newNodeName, AdjListNode{latency, nullptr, {}, {destName}}});
  nodes.at(srcName).backedges.push_back(newNodeName);
}


void AdjListGraph::dfs(std::string& currentNode, std::string& end, std::vector<std::string>& currentPath, std::set<std::string>& visited, std::vector<std::vector<std::string>>& paths, bool ignoreBackedges) {
    // If the current node is the target, add the current path to paths and return.
    if (currentNode == end) {
        paths.push_back(currentPath);
        return;
    }

    // Iterate over all adjacent nodes
    for (std::string& neighbor : nodes.at(currentNode).edges) {
        // If the neighbor has not been visited, visit it
        if (visited.find(neighbor) == visited.end()) {
            visited.insert(neighbor); // Mark as visited
            currentPath.push_back(neighbor); // Add to the current pat
            // Recursively visit the neighbor
            dfs(neighbor, end, currentPath, visited, paths, ignoreBackedges);
            // Backtrack: remove the neighbor from the current path and visited set for other paths
            currentPath.pop_back();
            visited.erase(neighbor);
        }
    }

    if(!ignoreBackedges) {
        for (std::string& neighbor : nodes.at(currentNode).backedges) {
        // If the neighbor has not been visited, visit it
        if (visited.find(neighbor) == visited.end()) {
            visited.insert(neighbor); // Mark as visited
            currentPath.push_back(neighbor); // Add to the current pat
            // Recursively visit the neighbor
            dfs(neighbor, end, currentPath, visited, paths, ignoreBackedges);
            // Backtrack: remove the neighbor from the current path and visited set for other paths
            currentPath.pop_back();
            visited.erase(neighbor);
        }
    }
    }
}

std::vector<std::vector<std::string>> AdjListGraph::findPaths(std::string start, std::string end, bool ignoreBackedge) {
    std::vector<std::vector<std::string>> paths;
    std::vector<std::string> currentPath{start};
    std::set<std::string> visited{start};
    dfs(start, end, currentPath, visited, paths, ignoreBackedge);
    return paths;
}


std::vector<std::vector<std::string>> AdjListGraph::findPaths(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge) {
  assert(startOp && endOp && "Start and end operations must not be null");
  llvm::dbgs() << "Finding paths from " << startOp->getAttrOfType<StringAttr>("handshake.name").str() << " to " << endOp->getAttrOfType<StringAttr>("handshake.name").str() << "\n";
  return findPaths(startOp->getAttrOfType<StringAttr>("handshake.name").str(), endOp->getAttrOfType<StringAttr>("handshake.name").str(), ignoreBackedge);
}


std::vector<std::string> AdjListGraph::findLongestNonCyclicPath(mlir::Operation *startOp) {
  std::string start = startOp->getAttrOfType<StringAttr>("handshake.name").str();
  std::vector<std::string> path;
  std::stack<std::pair<std::vector<std::string>, std::set<std::string>>> pathStack;
  int maxLatency = 0;
  // Initialize the stack with the path containing the source node and its visited set
  pathStack.push({{start}, {start}});
  while (!pathStack.empty()) {
    // Get the current path and visited set from the stack
    auto [currentPath, visited] = pathStack.top();
    pathStack.pop();
    // Get the last node in the current path
    std::string currentNode = currentPath.back();
    // If the current latency is higher than the max latency, update the max latency and path
    if (getPathLatency(currentPath) >= maxLatency) {
      maxLatency = getPathLatency(currentPath);
      path = currentPath;
    }
    // Get all adjacent nodes of the current node
    for (const std::string& neighbor : nodes.at(currentNode).edges) {
      // If the neighbor has not been visited in the current path, extend the path
      if (visited.find(neighbor) == visited.end()) {
          std::vector<std::string> newPath = currentPath;
          newPath.push_back(neighbor);
          std::set<std::string> newVisited = visited;
          newVisited.insert(neighbor);
          // Push the new path and updated visited set onto the stack
          pathStack.push({newPath, newVisited});
      }
    }
  }
  return path;
}


int AdjListGraph::getPathLatency(std::vector<std::string> path) {
  int latency = 0;
  for(auto &node: path) {
    latency += nodes.at(node).latency;
  }
  return latency;
}

std::vector<mlir::Operation*> AdjListGraph::getOperationsWithOpName(std::string opName) {
  std::vector<mlir::Operation*> ops;
  for(auto &node: nodes) {
    if(node.second.op && std::string(node.second.op->getName().getStringRef()) == opName)
    {
      ops.push_back(node.second.op);
    }
  }
  return ops;
}


int AdjListGraph::findMaxPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge) {
  assert(startOp && endOp && "Start and end operations must not be null");
  std::vector<std::vector<std::string>> paths = findPaths(startOp, endOp, ignoreBackedge);
  int maxLatency = 0;
  std::vector<std::string> maxPath;
  for(auto &path: paths)
  {
    int latency = getPathLatency(path);
    if(maxLatency < latency) {
      maxLatency = latency;
      maxPath = path;
    }
  }
  
  llvm::dbgs() << "latency: " << maxLatency << " path: ";
  printPath(maxPath);

  return maxLatency;
}

int AdjListGraph::findMinPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge) {
  assert(startOp && endOp && "Start and end operations must not be null");
  std::vector<std::vector<std::string>> paths = findPaths(startOp, endOp, ignoreBackedge);
  int minLatency = INT_MAX;
  for(auto &path: paths)
  {
    minLatency = std::min(minLatency, getPathLatency(path));
  }
  return minLatency;
}


std::vector<mlir::Operation*> AdjListGraph::getConnectedOps(mlir::Operation *op) {
  std::vector<mlir::Operation*> connectedOps;
  std::string opName = op->getAttrOfType<StringAttr>("handshake.name").str();

  for(auto &node: nodes.at(opName).edges) {
    connectedOps.push_back(nodes.at(node).op);
  }

  for(auto &aritificalNode: nodes.at(opName).backedges) {
    for(auto &node: nodes.at(aritificalNode).backedges) {
      connectedOps.push_back(nodes.at(node).op);
    }
  }

  return connectedOps;
}

