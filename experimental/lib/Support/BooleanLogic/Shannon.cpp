//===- Shannon.cpp - Shannon Decomposition for Bool Expressions -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the functions for performing Shannon decomposition on
// boolean logic expressions. It includes methods for replacing variables with
// values, performing positive and negative Shannon expansions, and applying
// Shannon decomposition.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/BooleanLogic/Shannon.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <string>
#include <vector>

using namespace dynamatic::experimental::boolean;
using namespace llvm;

void MultiplexerIn::print() {
  if (boolexpression.has_value())
    boolexpression.value()->print();

  if (mux)
    mux->print();
}

void Multiplexer::print() {
  llvm::errs() << "in0\n";
  in0->print();
  llvm::errs() << "in1\n";
  in1->print();
  llvm::errs() << "cond\n";
  cond->print();
}

void dynamatic::experimental::boolean::replaceVarWithValue(
    BoolExpression *exp, const std::string &var, ExpressionType t) {
  if (exp->type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(exp);
    if (singleCond->id == var) {
      if (singleCond->isNegated) {
        if (t == ExpressionType::One)
          exp->type = ExpressionType::Zero;
        if (t == ExpressionType::Zero)
          exp->type = ExpressionType::One;
      } else {
        exp->type = t;
      }
    }

  } else if (exp->type == ExpressionType::And ||
             exp->type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(exp);
    if (op->left)
      replaceVarWithValue(op->left, var, t);

    if (op->right)
      replaceVarWithValue(op->right, var, t);
  }
}

/// Find the postivie cofactor by simply replacing the variable with 1 then
/// minimizing the expression
BoolExpression *dynamatic::experimental::boolean::shannonExpansionPositive(
    BoolExpression *exp, const std::string &var) {
  replaceVarWithValue(exp, var, ExpressionType::One);
  return exp->boolMinimize();
}

/// Find the negative cofactor by simply replacing the variable with 0 then
/// minimizing the expression
BoolExpression *dynamatic::experimental::boolean::shannonExpansionNegative(
    BoolExpression *exp, const std::string &var) {
  replaceVarWithValue(exp, var, ExpressionType::Zero);
  return exp->boolMinimize();
}

MultiplexerIn *dynamatic::experimental::boolean::applyShannon(
    BoolExpression *exp, const std::vector<std::string> &cofactorList) {
  if (exp->type == ExpressionType::Variable ||
      exp->type == ExpressionType::Zero || exp->type == ExpressionType::One)
    return new MultiplexerIn(exp);

  const std::string &var = cofactorList[0];

  BoolExpression *expCopyForNeg = exp->deepCopy();
  BoolExpression *neg = shannonExpansionNegative(expCopyForNeg, var);

  BoolExpression *expCopyForPos = exp->deepCopy();
  BoolExpression *pos = shannonExpansionPositive(expCopyForPos, var);

  std::vector<std::string> subList(std::next(cofactorList.begin()),
                                   cofactorList.end());

  MultiplexerIn *in0 = applyShannon(neg, subList);
  MultiplexerIn *in1 = applyShannon(pos, subList);
  MultiplexerIn *select =
      new MultiplexerIn(new SingleCond(ExpressionType::Variable, var));

  Multiplexer *mux = new Multiplexer(in0, in1, select);

  return new MultiplexerIn(mux);
}
