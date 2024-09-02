//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the supporting datastructures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingSupport.h"

using namespace mlir;
using namespace dynamatic;

void SwitchingInfo::insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel) {
  std::pair<unsigned, unsigned> BBPair = {srcBB, dstBB};

  // Check the existence of the backedge pair
  if (backEdgeToCFDFCMap.find(BBPair) != backEdgeToCFDFCMap.end()) {
    backEdgeToCFDFCMap[BBPair].push_back(static_cast<unsigned>(std::stoul(mgLabel.str())));
  } else {
    std::vector<unsigned> tmpVector{static_cast<unsigned>(std::stoul(mgLabel.str()))};
    backEdgeToCFDFCMap[std::make_pair(srcBB, dstBB)] = tmpVector;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjNode
//
//===----------------------------------------------------------------------===//
AdjNode::AdjNode(mlir::Operation* selOp, 
          const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
          const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency) {
  // Initialize all variables
  this->op = selOp;
  this->pres = predecessors;
  this->sucs = successors;
  this->sucsDataWidthMap = sucDataWidthMap;
  this->nodeLatency = latency;

  // Initialize toggle count map
  for (auto& [key, value] : sucsDataWidthMap) {
    std::map<unsigned, unsigned> tmpMap;

    for (int i = 0; i < value; i++) {
      tmpMap[i] = 0;
    }

    perChannelToggle[key] = tmpMap;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjGraph
//
//===----------------------------------------------------------------------===//
// Initialize the whole adjacency graph for the selected segment
AdjGraph::AdjGraph(const buffer::CFDFC& cfdfc, const TimingDatabase& timingDB, unsigned II) {
  // TODO: Need to check whether invalid backedges are removed from the CFDFC or not

  // Traverse all units in the mg
  for (auto& unit: cfdfc.units) {
    unsigned latency = 0;
    std::string unitType = unit->getName().getStringRef().str();
    std::string unitName = unit->getAttrOfType<StringAttr>("handshake.name").str();
    unsigned numSlots = 0;

    // Get the unit latency
    if (unitType == "handshake.buffer") {
      // Buffer node encountered, need to retrieve the buffer attributes
      auto hwAttr = unit->getAttrOfType<DictionaryAttr>("hw.parameters");

      for (auto& namedAttr : hwAttr.getValue()) {
        mlir::Attribute value = namedAttr.getValue();
        mlir::StringAttr name = namedAttr.getName().cast<mlir::StringAttr>();

        if (name.str() == "TIMING") {
          // Timing information
          auto timingInfo = value.dyn_cast<TimingAttr>().getInfo();

          if (timingInfo.getLatency(SignalType::DATA)) {
            latency = timingInfo.getLatency(SignalType::DATA).value();
          } else {
            latency = 0;
          }
        } else {
          // Get the number of slots
          if (auto actualValue = value.dyn_cast<mlir::IntegerAttr>()) {
            numSlots = static_cast<unsigned>(actualValue.getValue().getZExtValue());
          }
        }

      }
      

    } else {
      // Check the existence of the key
      if (OP_DELAY_MAP.find(unitType) != OP_DELAY_MAP.end()) {
        latency = OP_DELAY_MAP.at(unitType);
      } else {
        llvm::errs() << "[ERROR] Unit Type not found in OP_DELAY_MAP: " << unitType << "\n";
      }
    }



  }
}


//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap) {
  for (const auto& selPair : selMap) {
    const std::pair<unsigned, unsigned>& key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", " << key.second <<") : [";

    for (const auto& selMG: mgList) {
      llvm::dbgs() << selMG << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap) {
  for (const auto& selPair: selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel <<" : [";

    for (const auto& selBB: BBList) {
      llvm::dbgs() << selBB << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

std::string removeDigits(const std::string& inStr) {
  std::regex digitsRegex("\\d");

  std::string outStr = std::regex_replace(inStr, digitsRegex, "");

  return outStr;
}

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  std::regex re(delimiter);
  std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
  std::sregex_token_iterator end;

  for (; it != end; ++it) {
    tokens.push_back(it->str());
  }

  return tokens;
}

std::string strip(const std::string &inputStr, const std::string &toRemove) {
  // Trim leading and trailing whitespace
  size_t start = inputStr.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
      return "";  // Return an empty string if there are only whitespaces
  }
  size_t end = inputStr.find_last_not_of(" \t\n\r\f\v");
  std::string stripped = inputStr.substr(start, end - start + 1);

  if (toRemove == "") {
    return stripped;
  }

  // Remove all occurrences of the specified substring 'toRemove'
  size_t pos = stripped.find(toRemove);
  while (pos != std::string::npos) {
      stripped.erase(pos, toRemove.length());
      pos = stripped.find(toRemove, pos);
  }

  return stripped;
}
