//===- GsaAnalysis.h - Gated Single Assignment analyis utilities
//----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assingment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/GsaAnalysis.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;

// TODO: 
// (1) figure out something for when a Phi is fed from a global argument: Use block == &region.front()
// (2) Create the getter function!!
// (3) In the conversion pass, check how to kill SSA maximization

SmallVector<SSAPhi *, 4> GsaAnalysis::getSsaPhis(int funcOpIdx) {
  SmallVector<SSAPhi *, 4> ssa_phis;

  for(auto& ssa_phi : all_ssa_phis[funcOpIdx])
    ssa_phis.emplace_back(ssa_phi);

  return ssa_phis;
}


void fillSsaPhiOps(SSAPhi* phi, Block* arg_block, int arg_idx) {
// Loop over the predecessor blocks of the owner_block to identify the producer operations
  for (Block *pred_block : arg_block->getPredecessors()) {
    // For each block, identify its terminator branching instruction
    auto branch_op =
        dyn_cast<BranchOpInterface>(pred_block->getTerminator());
    assert(branch_op && "Expected terminator operation in a predecessor "
                        "block feeding a block argument!");

    for (auto [idx, succ_branch_block] :
    llvm::enumerate(branch_op->getSuccessors())) {
      if(succ_branch_block == arg_block) {
        Operation *producer_operation = branch_op.getSuccessorOperands(idx)[arg_idx].getDefiningOp();

        // If there is no operation in this block, then it must be in the block's arguments....
        if(producer_operation == nullptr) {
          // llvm::errs() << "The desired operation for phi number " << arg_idx << " in ";
          // arg_block->printAsOperand(llvm::errs());
          // llvm::errs() << " is the block argument of ";
          // pred_block->printAsOperand(llvm::errs());
          // llvm::errs() << "\n\n";

          // identify the arg_idx of this value in the arguments of the pred_block
          int pred_arg_idx  = 0;
          for(BlockArgument arg : pred_block->getArguments()) {
            if(arg == branch_op.getSuccessorOperands(idx)[arg_idx])
              break;

            pred_arg_idx++;
          }

          phi->arg_idx_producer_operations.emplace_back(pred_arg_idx);
          phi->is_phi_producer_operations.emplace_back(true);
        } else {
          phi->is_phi_producer_operations.emplace_back(false);
          phi->arg_idx_producer_operations.emplace_back(-1);
        }

        phi->producer_operations.emplace_back(producer_operation);
        phi->producer_blocks.emplace_back(pred_block);
      }
    }

  }

}

// Adds a new entry to the private field all_ssa_phis that is composed of SSAPhi
// objects. Each SSAPhi contains the owner Block along with a vector of the
// producer operations
void GsaAnalysis::identifySsaPhis(func::FuncOp &funcOp) {
  Region &funcReg = funcOp.getRegion();

  SmallVector<SSAPhi *, 4> ssa_phis;

  // Loop over the blocks
  for (Block &block : funcReg.getBlocks()) {
    // Loop over the block's arguments
    int arg_idx = 0;
    for (BlockArgument arg : block.getArguments()) {
      Block *owner_block = arg.getOwner();

      // Create a new SSAPhi object
      SSAPhi* phi = new SSAPhi;
      fillSsaPhiOps(phi, owner_block, arg_idx); 

      // Add Phi only for Blocks that have multiple predecessors to avoid counting the initial block that directly take the function arguments
      if(!owner_block->getPredecessors().empty()) {
        // the block that the argument is inside
        phi->owner_block = owner_block; 
        phi->arg_idx = arg_idx;
        ssa_phis.emplace_back(phi);
      } else 
        delete phi;

      arg_idx++;
    }
  }

  all_ssa_phis.emplace_back(ssa_phis);
  fillProducerSsaPhis();
}

void GsaAnalysis::fillProducerSsaPhis() {
  for(auto& ssa_phi : all_ssa_phis[all_ssa_phis.size() - 1]) {
    for(size_t i = 0; i < ssa_phi->producer_operations.size(); i++) {
      if(ssa_phi->is_phi_producer_operations[i]){
        // search for the ssa_phi that has owner block equivalent to producer_blocks[i] and arg_idx equivalent to arg_idx_producer_operations[i] 
        Block* desired_block = ssa_phi->producer_blocks[i];
        int desired_arg_idx = ssa_phi->arg_idx_producer_operations[i];
        for(auto& another_ssa_phi : all_ssa_phis[all_ssa_phis.size() - 1]) {
          if(another_ssa_phi == ssa_phi)
            continue;

          if(another_ssa_phi->owner_block == desired_block && another_ssa_phi->arg_idx == desired_arg_idx) {
            ssa_phi->producer_ssa_phis.push_back(another_ssa_phi);
          }
        }
      }
    }

  }
}


// takes a function ID and a Block* and searches for this Block's name in the
// all_deps of this function and overwrites its pointer value
void GsaAnalysis::adjustBlockPtr(int funcOp_idx, Block *new_block) {
  // use the name to search for this block in all ssa phis and update its
  // ptr
  for (auto &one_ssa_phi : all_ssa_phis[funcOp_idx]) {
    Block *old_block = one_ssa_phi->owner_block;
    compareNamesAndModifyBlockPtr(new_block, old_block);
  }

  //  // use the name to search for this block in all gsa gates and update its
  // // ptr
  // for (auto &one_ssa_phi : all_gsa_gates[funcOp_idx]) {
  //   Block *old_block = one_ssa_phi.first->owner_block;
  //   compareNamesAndModifyBlockPtr(new_block, old_block);
  // }
}

void GsaAnalysis::compareNamesAndModifyBlockPtr(
    Block *new_block, Block *old_block) {
  // get the name of the new_block
  std::string name;
  llvm::raw_string_ostream os(name);
  new_block->printAsOperand(os);
  std::string new_block_name = os.str();

  // check if the new_block_name is the same as the name of the block in the old_block
  old_block->printAsOperand(os);
  std::string old_block_name = os.str();
  if (old_block_name == new_block_name)
    old_block = new_block;
}


bool isCyclicPhis(SSAPhi* one_phi, SSAPhi* another_phi) {
  bool flag_1 = false;
  int phi_pred_count = 0;
  int pred_count = 0;
  // loop over the inputs of the another_phi
  for(auto& phi_pred_flag : another_phi->is_phi_producer_operations) {
     if(phi_pred_flag) {
      if(another_phi->producer_ssa_phis[phi_pred_count] == one_phi) {
        flag_1 = true;
         break;
      }
      phi_pred_count++;
    }
    pred_count++;
  }

  bool flag_2 = false;
  phi_pred_count = 0;
  pred_count = 0;
  for(auto& phi_pred_flag : one_phi->is_phi_producer_operations) {
     if(phi_pred_flag) {
      if(one_phi->producer_ssa_phis[phi_pred_count] == another_phi) {
        flag_2 = true;
         break;
      }
      phi_pred_count++;
    }
    pred_count++;
  }

  return (flag_1 && flag_2);
}

void printOnePhi(SSAPhi* one_phi) {
  int phi_pred_count = 0;
  int pred_count = 0;
  // loop over the inputs of the one_phi
  for(auto& phi_pred_flag : one_phi->is_phi_producer_operations) {
    if(phi_pred_flag) {
      // this input is a Phi
      if(!isCyclicPhis(one_phi, one_phi->producer_ssa_phis[phi_pred_count]))
        printOnePhi(one_phi->producer_ssa_phis[phi_pred_count]);
      else {
        llvm::errs() << " cyclic phi in ";
        one_phi->producer_ssa_phis[phi_pred_count]->owner_block->printAsOperand(llvm::errs());
        llvm::errs() << "  ";
      }
      phi_pred_count++;
    } else {
      // this input is an operation
      llvm::errs() << one_phi->producer_operations[pred_count]->getName();
      llvm::errs() << " in ";
      one_phi->producer_blocks[pred_count]->printAsOperand(llvm::errs());
      llvm::errs() << "  ";
    }
    pred_count++;
  }
}

void GsaAnalysis::printSsaPhis(int funcOp_idx) {
  llvm::errs() << "\n*********************************\n\n";

  // The goal is to identify the number of preds (which should be 2),
    // Then identify for each pred if it is an operation or another Phi (LATER ON, ADD ANOTHER CASE WHICH IS IF IT IS A GLOBAL ARGUMENT) 

  for(auto& one_phi : all_ssa_phis[funcOp_idx]) {
    // Print the inputs of each Phi
    llvm::errs() << "\n\t";
    printOnePhi(one_phi);
  }
  
  llvm::errs() << "\n*********************************\n\n";
}



// llvm::errs() << "Printing some facts:\n";
//   for(auto& one_phi : all_ssa_phis[funcOp_idx]) {
//     llvm::errs() << " Phi in ";
//     one_phi->owner_block->printAsOperand(llvm::errs());
//     llvm::errs() << " at arg_idx: ";
//     llvm::errs() << one_phi->arg_idx << "\n";
//     if(one_phi->producer_operations.size() == one_phi->is_phi_producer_operations.size() && one_phi->producer_operations.size() == one_phi->producer_blocks.size() && one_phi->producer_operations.size() == one_phi->arg_idx_producer_operations.size())
//       llvm::errs() << " has correct sizes \n";
//     else
//     llvm::errs() << " has INcorrect sizes \n";

//     int count = 0;
//     for(auto& flag : one_phi->is_phi_producer_operations) {
//       if(flag)
//         count++;
//     }

//     if(count == one_phi->producer_ssa_phis.size())
//       llvm::errs() << "\tCORRECT\n";
//     else
//       llvm::errs() << "\tINCORRECT\n";
//   }