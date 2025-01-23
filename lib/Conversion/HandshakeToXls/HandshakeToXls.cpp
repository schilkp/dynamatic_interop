//===- HandshakeToXls.cpp - Convert Handshake to Xls ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Converts Handshake constructs into an equivalent xls proc graph.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/HandshakeToXls.h"

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cctype>
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//
namespace {

unsigned int innerTypeWidth(Type type) {

  assert(isa<handshake::ControlType>(type) ||
         isa<handshake::ChannelType>(type));

  unsigned int width = 0;
  if (auto ch = dyn_cast<handshake::ChannelType>(type)) {
    width = ch.getDataBitWidth();
  }
  return width;
}

Type convertInnerType(OpBuilder builder, Type type) {
  return builder.getIntegerType(innerTypeWidth(type));
}

// Wraps a type in a SchanType.
// NOTE: Taken from Xls.
xls::SchanType makeSchanType(Type type, bool isInput) {
  return xls::SchanType::get(type.getContext(), type, isInput);
}

// Creates a skeleton for an SprocOp. The sproc has input and output channels
// corresponding to inputs and results. inputs/results should NOT be channel
// types, they should be the underlying types.
// NOTE: Taken from Xls.
xls::SprocOp createSprocSkeleton(ImplicitLocOpBuilder &builder,
                                 TypeRange inputs, TypeRange results,
                                 std::string_view name) {

  OpBuilder::InsertionGuard guard(builder);

  auto sproc = builder.create<xls::SprocOp>(name,
                                            /*is_top=*/false,
                                            /*boundary_channel_names=*/
                                            nullptr,
                                            /*zeroinitializer=*/true);

  Block &spawns = sproc.getSpawns().emplaceBlock();
  Block &next = sproc.getNext().emplaceBlock();

  for (Type input : inputs) {
    Type type = makeSchanType(input, true);
    spawns.addArgument(type, builder.getLoc());
    next.addArgument(type, builder.getLoc());
  }
  for (Type result : results) {
    Type type = makeSchanType(result, false);
    spawns.addArgument(type, builder.getLoc());
    next.addArgument(type, builder.getLoc());
  }

  builder.setInsertionPointToEnd(&spawns);
  builder.create<xls::YieldOp>(builder.getLoc(), spawns.getArguments());
  builder.setInsertionPointToEnd(&next);
  builder.create<xls::YieldOp>(builder.getLoc());
  return sproc;
}

xls::AfterAllOp createAfterAll(OpBuilder builder, ValueRange ts) {
  return builder.create<xls::AfterAllOp>(
      builder.getUnknownLoc(), xls::TokenType::get(builder.getContext()), ts);
}

xls::AfterAllOp createAfterAll(OpBuilder builder, Value t1, Value t2) {
  return createAfterAll(builder, ValueRange{t1, t2});
}

xls::AfterAllOp createAfterAll(OpBuilder builder, Value t1, Value t2,
                               Value t3) {
  return createAfterAll(builder, ValueRange{t1, t2, t3});
}

} // namespace

//===----------------------------------------------------------------------===//
// Handshake procs
//===----------------------------------------------------------------------===//

class HandshakeProc {
public:
  virtual std::string name() const = 0;
  virtual ~HandshakeProc() = default;
  virtual void build(OpBuilder builder) const = 0;

  HandshakeProc(const HandshakeProc &) = default;
  HandshakeProc() = default;
};

//===----------------------------------------------------------------------===//
// Fork
//===----------------------------------------------------------------------===//

class ForkProc : public HandshakeProc {
public:
  ForkProc(unsigned int width, unsigned int fanout)
      : width(width), fanout(fanout) {}
  std ::string name() const override {
    return "fork_" + std ::to_string(width) + "_x" + std ::to_string(fanout);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(ForkOp op);

private:
  unsigned int width;
  unsigned int fanout;
};

void ForkProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  TypeRange inputs{b.getIntegerType(width)};
  llvm::SmallVector<Type> outputs{};
  for (size_t i = 0; i < fanout; i++) {
    outputs.push_back(b.getIntegerType(width));
  }

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rxOp = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs.front());
  auto rxVal = rxOp.getResult();
  auto tok1 = rxOp.getTknOut();
  for (unsigned int i = 0; i < fanout; i++) {
    b.create<xls::SSendOp>(tok1, rxVal, nextArgs[1 + i]);
  }
}

std::shared_ptr<HandshakeProc> ForkProc::get(ForkOp op) {
  auto width = innerTypeWidth(op.getOperand().getType());
  auto fanout = op->getNumResults();
  return std::make_shared<ForkProc>(ForkProc(width, fanout));
}

//===----------------------------------------------------------------------===//
// Join
//===----------------------------------------------------------------------===//

class JoinProc : public HandshakeProc {
public:
  JoinProc(unsigned int width, unsigned int fanIn)
      : width(width), fanIn(fanIn) {}
  std ::string name() const override {
    return "join_" + std ::to_string(width) + "_x" + std ::to_string(fanIn);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(JoinOp op);

private:
  unsigned int width;
  unsigned int fanIn;
};

void JoinProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeVal = b.getIntegerType(width);
  auto typeToken = xls::TokenType::get(b.getContext());
  auto typeBool = b.getI1Type();

  llvm::SmallVector<Type> inputs{};
  for (size_t i = 0; i < fanIn; i++) {
    inputs.push_back(typeVal);
  }
  TypeRange outputs{typeVal};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();

  // Create first non-blocking receive/send pair:
  auto rxOp = b.create<xls::SNonblockingReceiveOp>(typeToken, typeVal, typeBool,
                                                   tok0, nextArgs[0], Value{});
  b.create<xls::SSendOp>(rxOp.getTknOut(), rxOp.getResult(), nextArgs[fanIn],
                         rxOp.getValid());

  for (unsigned int i = 1; i < fanIn; i++) {
    auto prevRxNotValid = b.create<xls::NotOp>(rxOp.getValid()).getResult();

    rxOp = b.create<xls::SNonblockingReceiveOp>(typeToken, typeVal, typeBool,
                                                rxOp.getTknOut(), nextArgs[i],
                                                prevRxNotValid);

    b.create<xls::SSendOp>(rxOp.getTknOut(), rxOp.getResult(), nextArgs[fanIn],
                           rxOp.getValid());
  }
}

std::shared_ptr<HandshakeProc> JoinProc::get(JoinOp op) {
  auto fanIn = op->getNumOperands();
  auto width = innerTypeWidth(op.getResult().getType());
  return std::make_shared<JoinProc>(JoinProc(width, fanIn));
}

//===----------------------------------------------------------------------===//
// Select
//===----------------------------------------------------------------------===//

class SelectProc : public HandshakeProc {
public:
  SelectProc(unsigned int width) : width(width) {}
  std ::string name() const override {
    return "select_" + std ::to_string(width);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(SelectOp op);

private:
  unsigned int width;
};

void SelectProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeVal = b.getIntegerType(width);
  auto typeBool = b.getI1Type();

  llvm::SmallVector<Type> inputs{typeBool, typeVal, typeVal};
  TypeRange outputs{typeVal};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  // Receive the selector and both inputs:
  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rxSel = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);
  auto rxTrueVal = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[1]);
  auto rxFalseVal = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[2]);

  auto tokResult = createAfterAll(b, rxSel.getTknOut(), rxTrueVal.getTknOut(),
                                  rxFalseVal.getTknOut())
                       .getResult();

  // Select the result:
  auto selOp = b.create<xls::SelOp>(
      typeVal, rxSel.getResult(),
      ValueRange{rxFalseVal.getResult(), rxTrueVal.getResult()});

  // Send the result:
  b.create<xls::SSendOp>(tokResult, selOp.getResult(), nextArgs[3]);
}

std::shared_ptr<HandshakeProc> SelectProc::get(SelectOp op) {
  auto width = innerTypeWidth(op.getResult().getType());
  return std::make_shared<SelectProc>(SelectProc(width));
}

//===----------------------------------------------------------------------===//
// Mux
//===----------------------------------------------------------------------===//

class MuxProc : public HandshakeProc {
public:
  MuxProc(unsigned int width, unsigned int fanIn, unsigned selectWidth)
      : width(width), fanIn(fanIn), selectWidth(selectWidth) {}
  std ::string name() const override {
    return "mux_" + std ::to_string(width) + "_sel" +
           std::to_string(selectWidth) + "_x" + std::to_string(fanIn);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(MuxOp op);

private:
  unsigned int width;
  unsigned int fanIn;
  unsigned int selectWidth;
};

void MuxProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeVal = b.getIntegerType(width);
  auto typeSelect = b.getIntegerType(selectWidth);
  auto typeToken = xls::TokenType::get(b.getContext());

  llvm::SmallVector<Type> inputs{typeSelect};
  for (unsigned int i = 0; i < fanIn; i++) {
    inputs.push_back(typeVal);
  }
  TypeRange outputs{typeVal};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  // Receive the selector:
  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rxSel = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);

  // For each possible input, check if it was selected and perform a conditional
  // receive + send:
  for (unsigned int i = 0; i < fanIn; i++) {
    auto idxLit = b.create<mlir::xls::ConstantScalarOp>(
        typeSelect, builder.getIntegerAttr(typeSelect, i));

    auto isSelected =
        b.create<mlir::xls::EqOp>(rxSel.getResult(), idxLit.getResult());

    auto rxVal = b.create<xls::SBlockingReceiveOp>(typeToken, typeVal, tok0,
                                                   nextArgs[i + 1], isSelected);

    b.create<xls::SSendOp>(rxVal.getTknOut(), rxVal.getResult(),
                           nextArgs[fanIn + 1], isSelected.getResult());
  }
}

std::shared_ptr<HandshakeProc> MuxProc::get(MuxOp op) {
  auto width = innerTypeWidth(op.getResult().getType());
  auto fanIn = op.getDataOperands().size();
  auto selectWidth = innerTypeWidth(op.getSelectOperand().getType());
  return std::make_shared<MuxProc>(MuxProc(width, fanIn, selectWidth));
}

//===----------------------------------------------------------------------===//
// Control Merge
//===----------------------------------------------------------------------===//

class CMergeProc : public HandshakeProc {
public:
  CMergeProc(unsigned int width, unsigned int fanIn, unsigned idxWidth)
      : width(width), fanIn(fanIn), idxWidth(idxWidth) {}
  std ::string name() const override {
    return "cmerge_" + std ::to_string(width) + "_idx" +
           std::to_string(idxWidth) + "_x" + std::to_string(fanIn);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(ControlMergeOp op);

private:
  unsigned int width;
  unsigned int fanIn;
  unsigned int idxWidth;
};

void CMergeProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeVal = b.getIntegerType(width);
  auto typeIdx = b.getIntegerType(idxWidth);
  auto typeToken = xls::TokenType::get(b.getContext());
  auto typeBool = b.getI1Type();

  llvm::SmallVector<Type> inputs{};
  for (unsigned int i = 0; i < fanIn; i++) {
    inputs.push_back(typeVal);
  }
  TypeRange outputs{typeVal, typeIdx};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  // For each possible input, perform a non-blocking receive. If succesfull,
  // send the result + index.

  auto tok0 = b.create<xls::AfterAllOp>().getResult();

  for (unsigned int i = 0; i < fanIn; i++) {

    auto rxOp = b.create<xls::SNonblockingReceiveOp>(
        typeToken, typeVal, typeBool, tok0, nextArgs[i], Value{});

    auto idxLit = b.create<mlir::xls::ConstantScalarOp>(
        typeIdx, builder.getIntegerAttr(typeIdx, i));

    b.create<xls::SSendOp>(rxOp.getTknOut(), rxOp.getResult(), nextArgs[fanIn],
                           rxOp.getValid());
    b.create<xls::SSendOp>(rxOp.getTknOut(), idxLit.getResult(),
                           nextArgs[fanIn + 1], rxOp.getValid());
  }
}

std::shared_ptr<HandshakeProc> CMergeProc::get(ControlMergeOp op) {
  auto width = innerTypeWidth(op.getResult().getType());
  auto fanIn = op.getDataOperands().size();
  auto idxType = innerTypeWidth(op.getIndex().getType());
  return std::make_shared<CMergeProc>(CMergeProc(width, fanIn, idxType));
}

//===----------------------------------------------------------------------===//
// Conditional Branch
//===----------------------------------------------------------------------===//

class CBranchProc : public HandshakeProc {
public:
  CBranchProc(unsigned int width) : width(width) {}
  std ::string name() const override {
    return "cbranch_" + std ::to_string(width);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(ConditionalBranchOp op);

private:
  unsigned int width;
};

void CBranchProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeVal = b.getIntegerType(width);
  auto typeBool = b.getI1Type();

  llvm::SmallVector<Type> inputs{typeBool, typeVal};
  TypeRange outputs{typeVal, typeVal};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  // Receive the selector and input:
  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rxSel = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);
  auto rxVal = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[1]);

  auto tokResult =
      createAfterAll(b, rxSel.getTknOut(), rxVal.getTknOut()).getResult();

  auto notSel = b.create<xls::NotOp>(rxSel.getResult());

  // Two conditional sends:
  b.create<xls::SSendOp>(tokResult, rxVal.getResult(), nextArgs[2],
                         rxSel.getResult());
  b.create<xls::SSendOp>(tokResult, rxVal.getResult(), nextArgs[3],
                         notSel.getResult());
}

std::shared_ptr<HandshakeProc> CBranchProc::get(ConditionalBranchOp op) {
  auto width = innerTypeWidth(op.getResult(0).getType());
  return std::make_shared<CBranchProc>(CBranchProc(width));
}

//===----------------------------------------------------------------------===//
// Source
//===----------------------------------------------------------------------===//

class SourceProc : public HandshakeProc {
public:
  SourceProc() = default;
  std ::string name() const override { return "source"; }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(SourceOp op);
};

void SourceProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  TypeRange inputs{};
  llvm::SmallVector<Type> outputs{b.getIntegerType(0)};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto lit =
      b.create<xls::ConstantScalarOp>(b.getIntegerType(0), 0).getResult();
  b.create<xls::SSendOp>(tok0, lit, nextArgs.front());
}

std::shared_ptr<HandshakeProc> SourceProc::get(SourceOp op) {
  return std::make_shared<SourceProc>(SourceProc());
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

class ConstantProc : public HandshakeProc {
public:
  ConstantProc(APInt &value) : value(value) {}
  std ::string name() const override {
    llvm::SmallString<16> valString;
    value.toString(valString, /*radix=*/16, /*signed=*/false,
                   /*formatAsCLiteral=*/true, /*UpperCase=*/false);
    return "constant_" + std::string(valString) + "_x" +
           std::to_string(value.getBitWidth());
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(ConstantOp op);

private:
  APInt value;
};

void ConstantProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto type = b.getIntegerType(value.getBitWidth());
  auto valueAttr = b.getIntegerAttr(type, value);

  TypeRange inputs{b.getIntegerType(0)};
  llvm::SmallVector<Type> outputs{type};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto tok1 = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]).getTknOut();
  auto lit = b.create<xls::ConstantScalarOp>(type, valueAttr).getResult();
  b.create<xls::SSendOp>(tok1, lit, nextArgs[1]);
}

std::shared_ptr<HandshakeProc> ConstantProc::get(ConstantOp op) {
  TypedAttr valueAttr = op.getValueAttr();

  if (auto intAttr = dyn_cast<mlir::IntegerAttr>(valueAttr)) {
    APInt value = intAttr.getValue();
    return std::make_shared<ConstantProc>(value);
  }
  if (auto floatAttr = dyn_cast<mlir::FloatAttr>(valueAttr)) {
    APInt value = floatAttr.getValue().bitcastToAPInt();
    return std::make_shared<ConstantProc>(value);
  }

  op->emitError() << "constant type must be integer or floating point.";
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Sink
//===----------------------------------------------------------------------===//

class SinkProc : public HandshakeProc {
public:
  SinkProc(unsigned int width) : width(width) {}
  std ::string name() const override {
    return "sink_x" + std::to_string(width);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(SinkOp op);

private:
  unsigned int width;
};

void SinkProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  TypeRange inputs{b.getIntegerType(width)};
  llvm::SmallVector<Type> outputs{};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  b.create<xls::SBlockingReceiveOp>(tok0, nextArgs.front());
}

std::shared_ptr<HandshakeProc> SinkProc::get(SinkOp op) {
  return std::make_shared<SinkProc>(
      SinkProc(innerTypeWidth(op.getOperand().getType())));
}

//===----------------------------------------------------------------------===//
// Integer Binary Ops (Addi, Andi, ...)
//===----------------------------------------------------------------------===//

enum class BinaryIOpKind : int {
  ADDI,
  ANDI,
  DIVSI,
  DIVUI,
  MULI,
  ORI,
  XORI,
  SHLI,
  SHRSI,
  SHRUI,
  SUBI,
};

class BinaryIOpProc : public HandshakeProc {
public:
  BinaryIOpProc(BinaryIOpKind kind, unsigned int width)
      : kind(kind), width(width) {}

  std ::string name() const override {
    const char *name;
    // clang-format off
    switch (kind) {
    case BinaryIOpKind::ADDI: { name = "addi"; break; }
    case BinaryIOpKind::ANDI: { name = "andi"; break; }
    case BinaryIOpKind::DIVSI: { name = "divsi"; break; }
    case BinaryIOpKind::DIVUI: { name = "divui"; break; }
    case BinaryIOpKind::MULI: { name = "muli"; break; }
    case BinaryIOpKind::ORI: { name = "ori"; break; }
    case BinaryIOpKind::XORI: { name = "xori"; break; }
    case BinaryIOpKind::SHLI: { name = "shli"; break; }
    case BinaryIOpKind::SHRSI: { name = "shrsi"; break; }
    case BinaryIOpKind::SHRUI: { name = "shrui"; break; }
    case BinaryIOpKind::SUBI: { name = "subi";break; }
    }
    // clang-format on
    return std::string(name) + "_x" + std ::to_string(width);
  }

  void build(OpBuilder builder) const override;

  static std::shared_ptr<HandshakeProc> get(Operation *op);

private:
  BinaryIOpKind kind;
  unsigned int width;
};

void BinaryIOpProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto type = b.getIntegerType(width);
  TypeRange inputs{type, type};
  llvm::SmallVector<Type> outputs{type};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rx1 = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);
  auto lhs = rx1.getResult();
  auto rx2 = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[1]);
  auto rhs = rx2.getResult();

  Value result;
  // clang-format off
  switch (kind) {
  case BinaryIOpKind::ADDI: { result = b.create<xls::AddOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::ANDI: { result = b.create<xls::AndOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::DIVSI: { result = b.create<xls::SdivOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::DIVUI: { result = b.create<xls::UdivOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::MULI: { result = b.create<xls::SmulOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::ORI: { result = b.create<xls::OrOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::XORI: { result = b.create<xls::XorOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::SHLI: { result = b.create<xls::ShllOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::SHRSI: { result = b.create<xls::ShraOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::SHRUI: { result = b.create<xls::ShrlOp>(lhs, rhs).getResult(); break; }
  case BinaryIOpKind::SUBI: { result = b.create<xls::SubOp>(lhs, rhs).getResult(); break; }
  }
  // clang-format on

  auto tokJoin =
      createAfterAll(b, rx1.getTknOut(), rx2.getTknOut()).getResult();
  b.create<xls::SSendOp>(tokJoin, result, nextArgs[2]);
}

std::shared_ptr<HandshakeProc> BinaryIOpProc::get(Operation *op) {
  auto kind = TypeSwitch<Operation *, std::optional<BinaryIOpKind>>(op)
                  .Case<AndIOp>([&](auto _) { return BinaryIOpKind::ANDI; })
                  .Case<AddIOp>([&](auto _) { return BinaryIOpKind::ADDI; })
                  .Case<DivSIOp>([&](auto _) { return BinaryIOpKind::DIVSI; })
                  .Case<DivUIOp>([&](auto _) { return BinaryIOpKind::DIVUI; })
                  .Case<MulIOp>([&](auto _) { return BinaryIOpKind::MULI; })
                  .Case<OrIOp>([&](auto _) { return BinaryIOpKind::ORI; })
                  .Case<XOrIOp>([&](auto _) { return BinaryIOpKind::XORI; })
                  .Case<ShLIOp>([&](auto _) { return BinaryIOpKind::SHLI; })
                  .Case<ShRSIOp>([&](auto _) { return BinaryIOpKind::SHRSI; })
                  .Case<ShRUIOp>([&](auto _) { return BinaryIOpKind::SHRUI; })
                  .Case<SubIOp>([&](auto _) { return BinaryIOpKind::SUBI; })
                  .Default([](Operation *op) { return std::nullopt; });

  if (!kind) {
    op->emitError("unknown kind");
    return nullptr;
  }
  auto width = innerTypeWidth(op->getResult(0).getType());

  return std::make_shared<BinaryIOpProc>(BinaryIOpProc(*kind, width));
}

//===----------------------------------------------------------------------===//
// Integer Comparison Ops
//===----------------------------------------------------------------------===//

class CmpIProc : public HandshakeProc {
public:
  CmpIProc(CmpIPredicate kind, unsigned int width) : kind(kind), width(width) {}

  std ::string name() const override {
    const char *name;
    // clang-format off
    switch (kind) {
    case CmpIPredicate::eq: { name = "eq"; break; }
    case CmpIPredicate::ne: { name = "ne"; break; }
    case CmpIPredicate::slt: { name = "slt"; break; }
    case CmpIPredicate::sle: { name = "sle"; break; }
    case CmpIPredicate::sgt: { name = "sgt"; break; }
    case CmpIPredicate::sge: { name = "sge"; break; }
    case CmpIPredicate::ult: { name = "ult"; break; }
    case CmpIPredicate::ule: { name = "ule"; break; }
    case CmpIPredicate::ugt: { name = "ugt"; break; }
    case CmpIPredicate::uge: { name = "uge"; break; }
    }
    // clang-format on
    return "cmpi_" + std::string(name) + "_x" + std ::to_string(width);
  }

  void build(OpBuilder builder) const override;

  static std::shared_ptr<HandshakeProc> get(CmpIOp op);

private:
  CmpIPredicate kind;
  unsigned int width;
};

void CmpIProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto type = b.getIntegerType(width);
  auto boolType = b.getI1Type();
  TypeRange inputs{type, type};
  llvm::SmallVector<Type> outputs{boolType};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rx1 = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);
  auto lhs = rx1.getResult();
  auto rx2 = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[1]);
  auto rhs = rx2.getResult();

  Value result;
  // clang-format off
  switch (kind) {
  case CmpIPredicate::eq: { result = b.create<xls::EqOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::ne: { result = b.create<xls::NeOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::slt: { result = b.create<xls::SltOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::sle: { result = b.create<xls::SleOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::sgt: { result = b.create<xls::SgtOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::sge: { result = b.create<xls::SgeOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::ult: { result = b.create<xls::UltOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::ule: { result = b.create<xls::UleOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::ugt: { result = b.create<xls::UgtOp>(lhs, rhs).getResult(); break; }
  case CmpIPredicate::uge: { result = b.create<xls::UgeOp>(lhs, rhs).getResult(); break; }
  }
  // clang-format on

  auto tokJoin =
      createAfterAll(b, rx1.getTknOut(), rx2.getTknOut()).getResult();
  b.create<xls::SSendOp>(tokJoin, result, nextArgs[2]);
}

std::shared_ptr<HandshakeProc> CmpIProc::get(CmpIOp op) {
  auto width = innerTypeWidth(op.getLhs().getType());
  return std::make_shared<CmpIProc>(CmpIProc(op.getPredicate(), width));
}

//===----------------------------------------------------------------------===//
// Bit Extension (extsi, extsu)
//===----------------------------------------------------------------------===//

class ExtendProc : public HandshakeProc {
public:
  ExtendProc(unsigned int from, unsigned int to, bool isSigned)
      : fromWidth(from), toWidth(to), isSigned(isSigned) {};

  std ::string name() const override {

    const char *name = isSigned ? "extsi" : "extsu";
    return std::string(name) + "_" + std::to_string(fromWidth) + "_" +
           std::to_string(toWidth);
  }
  void build(OpBuilder builder) const override;
  static std::shared_ptr<HandshakeProc> get(Operation *op);

private:
  unsigned int fromWidth;
  unsigned int toWidth;
  bool isSigned;
};

void ExtendProc::build(OpBuilder builder) const {
  OpBuilder::InsertionGuard guard(builder);
  auto b = ImplicitLocOpBuilder(builder.getUnknownLoc(), builder);

  auto typeIn = b.getIntegerType(fromWidth);
  auto typeOut = b.getIntegerType(toWidth);

  TypeRange inputs{typeIn};
  llvm::SmallVector<Type> outputs{typeOut};

  auto sproc = createSprocSkeleton(b, inputs, outputs, name());
  auto nextArgs = sproc.getNext().getArguments();
  Block *next = &sproc.getNext().getBlocks().front();
  b.setInsertionPointToStart(next);

  auto tok0 = b.create<xls::AfterAllOp>().getResult();
  auto rxOp = b.create<xls::SBlockingReceiveOp>(tok0, nextArgs[0]);
  Value result;
  if (isSigned) {
    result = b.create<xls::SignExtOp>(typeOut, rxOp.getResult()).getResult();
  } else {
    result = b.create<xls::ZeroExtOp>(typeOut, rxOp.getResult()).getResult();
  }
  b.create<xls::SSendOp>(rxOp.getTknOut(), result, nextArgs[1]);
}

std::shared_ptr<HandshakeProc> ExtendProc::get(Operation *op) {
  if (isa<ExtSIOp>(op)) {
    auto extOp = cast<ExtSIOp>(op);
    auto widthFrom = innerTypeWidth(extOp.getOperand().getType());
    auto widthTo = innerTypeWidth(extOp.getResult().getType());
    return std::make_shared<ExtendProc>(ExtendProc(widthFrom, widthTo, true));
  }

  if (isa<ExtUIOp>(op)) {
    auto extOp = cast<ExtUIOp>(op);
    auto widthFrom = innerTypeWidth(extOp.getOperand().getType());
    auto widthTo = innerTypeWidth(extOp.getResult().getType());
    return std::make_shared<ExtendProc>(ExtendProc(widthFrom, widthTo, false));
  }

  op->emitError("unknown op");
  return nullptr;
}

// TODO CF ops:
// BranchOp

// MergeOp
// ControlMergeOp
// LazyForkOp
// LoadOp

// StoreOp
// NotOp
// SharingWrapperOp

// TODO Misc ops:
// TruncIOp

// TODO Floats:
// AddFOp
// DivFOp
// CmpFOp
// ExtFOp
// TruncFOp
// MulFOp
// NegFOp
// SubFOp
// AbsFOp

// SIToFPOp
// FPToSIOp

//===----------------------------------------------------------------------===//
// Proc Manager
//===----------------------------------------------------------------------===//

class ProcManager {
public:
  ProcManager() = default;

  void buildRequiredProcs(OpBuilder builder) {
    for (auto const &[_, proc] : existingProcs) {
      proc->build(builder);
    }
  }

  std::shared_ptr<HandshakeProc> getProc(Operation *op) {
    auto proc =
        TypeSwitch<Operation *, std::shared_ptr<HandshakeProc>>(op)
            .Case<handshake::ForkOp>(
                [&](ForkOp op) { return ForkProc::get(op); })
            .Case<handshake::JoinOp>(
                [&](JoinOp op) { return JoinProc::get(op); })
            .Case<handshake::SelectOp>(
                [&](SelectOp op) { return SelectProc::get(op); })
            .Case<handshake::MuxOp>([&](MuxOp op) { return MuxProc::get(op); })
            .Case<handshake::ControlMergeOp>(
                [&](ControlMergeOp op) { return CMergeProc::get(op); })
            .Case<handshake::ConditionalBranchOp>(
                [&](ConditionalBranchOp op) { return CBranchProc::get(op); })
            .Case<handshake::SourceOp>(
                [&](SourceOp op) { return SourceProc::get(op); })
            .Case<handshake::SinkOp>(
                [&](SinkOp op) { return SinkProc::get(op); })
            .Case<handshake::ConstantOp>(
                [&](ConstantOp op) { return ConstantProc::get(op); })
            .Case<AndIOp, AddIOp, DivSIOp, DivUIOp, MulIOp, OrIOp, XOrIOp,
                  ShLIOp, ShRSIOp, ShRUIOp, SubIOp>(
                [&](auto op) { return BinaryIOpProc::get(op); })
            .Case<handshake::CmpIOp>(
                [&](CmpIOp op) { return CmpIProc::get(op); })
            .Case<handshake::ExtSIOp, handshake::ExtUIOp>(
                [&](auto op) { return ExtendProc::get(op); })
            .Default([](Operation *op) { return nullptr; });

    if (!proc) {
      op->emitOpError("operation not supported in XLS translation");
      return proc;
    }

    existingProcs[proc->name()] = proc;
    return proc;
  }

  std::unordered_map<std::string, std::shared_ptr<HandshakeProc>> existingProcs;
};

//===----------------------------------------------------------------------===//
// Handshake Operation Translation
//===----------------------------------------------------------------------===//

template <typename T>
class ConvertToXlsSpawn : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;
  using OpAdaptor = typename T::Adaptor;

  ConvertToXlsSpawn(MLIRContext *ctx, ProcManager &procMgr)
      : OpConversionPattern<T>(ctx), procMgr(procMgr) {}

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  ProcManager &procMgr;
};

template <typename T>
LogicalResult ConvertToXlsSpawn<T>::matchAndRewrite(
    T op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

  // Create output channels:
  SmallVector<xls::SchanOp> resultChannels = {};
  SmallVector<Value> resultChannelSenders = {};
  SmallVector<Value> resultChannelReceivers = {};

  for (auto outpt : op.getOperation()->getResults()) {
    auto innerType = convertInnerType(rewriter, outpt.getType());
    xls::SchanOp ch =
        rewriter.create<xls::SchanOp>(op.getLoc(), "ssa", innerType);
    ch.setFifoDepth(std::optional(0));
    ch.setBypass(std::optional(true));
    ch.setInputFlopKind(std::optional(xls::FlopKind::kNone));
    ch.setOutputFlopKind(std::optional(xls::FlopKind::kNone));
    ch.setRegisterPopOutputs(std::optional(false));
    ch.setRegisterPushOutputs(std::optional(false));
    resultChannels.push_back(ch);
    resultChannelSenders.push_back(ch.getIn());
    resultChannelReceivers.push_back(ch.getOut());
  }

  // Arguments to spawn are (input_ch1, ..., output_ch1, ...)
  SmallVector<Value> channels = {};
  append_range(channels, adaptor.getOperands());
  append_range(channels, resultChannelSenders);

  auto proc = procMgr.getProc(op.getOperation());
  if (!proc)
    return failure();

  rewriter.create<xls::SpawnOp>(
      op.getLoc(), ValueRange(channels),
      SymbolRefAttr::get(rewriter.getContext(), proc->name()));

  rewriter.replaceOp(op, resultChannelReceivers);
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion Driver
//===----------------------------------------------------------------------===//

class ConvertFunc : public OpConversionPattern<handshake::FuncOp> {
public:
  ConvertFunc(MLIRContext *ctx) : OpConversionPattern<handshake::FuncOp>(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertFunc::matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  if (funcOp.isExternal())
    return failure();

  rewriter.setInsertionPoint(funcOp);

  auto funcType = funcOp.getFunctionType();

  auto name = rewriter.getStringAttr(funcOp.getName()).strref();

  SmallVector<Attribute> boundaryChs = {};

  std::vector<Type> inputTypes = {};
  std::vector<Type> outputTypes = {};

  assert(funcType.getInputs().size() == funcOp.getArgNames().size());
  for (auto [input, name] : zip(funcType.getInputs(), funcOp.getArgNames())) {
    inputTypes.push_back(convertInnerType(rewriter, input));
    boundaryChs.push_back(name);
  }

  assert(funcType.getResults().size() == funcOp.getResNames().size());
  for (auto [output, name] : zip(funcType.getResults(), funcOp.getResNames())) {
    outputTypes.push_back(convertInnerType(rewriter, output));
    boundaryChs.push_back(name);
  }

  auto boundaryChsAttr = rewriter.getArrayAttr(ArrayRef(boundaryChs));
  auto sproc = rewriter.create<xls::SprocOp>(funcOp->getLoc(), name,
                                             /*is_top=*/true, boundaryChsAttr,
                                             /*zeroinitializer=*/true);

  Block &spawns = sproc.getSpawns().emplaceBlock();
  Block &next = sproc.getNext().emplaceBlock();

  rewriter.setInsertionPointToEnd(&spawns);
  rewriter.create<xls::YieldOp>(rewriter.getUnknownLoc());
  rewriter.setInsertionPointToEnd(&next);
  rewriter.create<xls::YieldOp>(rewriter.getUnknownLoc());

  SmallVector<Value> inputs = {};

  for (Type inputType : inputTypes) {
    Type type = makeSchanType(inputType, true);
    auto arg = spawns.addArgument(type, rewriter.getUnknownLoc());
    inputs.push_back(arg);
  }
  for (Type resultTypes : outputTypes) {
    Type type = makeSchanType(resultTypes, false);
    spawns.addArgument(type, rewriter.getUnknownLoc());
  }

  rewriter.inlineBlockBefore(funcOp.getBodyBlock(), spawns.getTerminator(),
                             inputs);

  rewriter.eraseOp(funcOp);

  return success();
}

class ConvertEnd : public OpConversionPattern<handshake::EndOp> {
public:
  ConvertEnd(MLIRContext *ctx) : OpConversionPattern<handshake::EndOp>(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::EndOp endOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertEnd::matchAndRewrite(handshake::EndOp endOp, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {

  // Delete both end op and the channels that feed it. Replace those channels
  // with the correct result operands of the sproc.
  auto sproc = cast<xls::SprocOp>(endOp->getParentOp());

  // First, filter out which spawn args are the outputs:
  SmallVector<Value> sprocOutputs = {};
  for (auto arg : sproc.getSpawns().getArguments()) {
    auto ch = cast<xls::SchanType>(arg.getType());
    if (ch.getIsOutput()) {
      sprocOutputs.push_back(arg);
    }
  }

  // Find all the extra channels that we will replace with the output args:
  SmallVector<xls::SchanOp> extraCh = {};
  SmallVector<Value> extraChInput = {};
  for (auto endVal : adaptor.getOperands()) {
    auto endCh = cast<xls::SchanOp>(endVal.getDefiningOp());
    extraCh.push_back(endCh);
    extraChInput.push_back(endCh.getIn());
  }
  assert(extraCh.size() == sprocOutputs.size());

  // Replace the usage of all extra channels with the output args:
  for (auto [chInp, arg] : llvm::zip(extraChInput, sprocOutputs)) {
    chInp.replaceAllUsesWith(arg);
  }

  // Delete channels:
  for (auto ch : extraCh) {
    rewriter.eraseOp(ch);
  }

  auto *parent = endOp.getOperation()->getParentOp();

  rewriter.eraseOp(endOp);

  return success();
}

class ConvertBuffer : public OpConversionPattern<handshake::BufferOp> {
public:
  ConvertBuffer(MLIRContext *ctx)
      : OpConversionPattern<handshake::BufferOp>(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::BufferOp bufOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertBuffer::matchAndRewrite(handshake::BufferOp bufOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {

  auto inp = adaptor.getOperand();
  auto ch = cast<xls::SchanOp>(adaptor.getOperand().getDefiningOp());

  auto params = bufOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
  if (!params) {
    bufOp.emitError() << "underdefined buffer (params)";
    return failure();
  }

  auto numSlotsAttr = params.getNamed(BufferOp::NUM_SLOTS_ATTR_NAME);
  if (!numSlotsAttr) {
    bufOp.emitError() << "underdefined buffer (numSlots)";
    return failure();
  }

  auto timingAttr = params.getNamed(BufferOp::TIMING_ATTR_NAME);
  if (!timingAttr) {
    bufOp.emitError() << "underdefined buffer (timing)";
    return failure();
  }

  rewriter.replaceOp(bufOp, inp);
  return success();
}

namespace {

/// Conversion pass driver. The conversion only works on modules containing
/// a single handshake function (handshake::FuncOp) at the moment. The
/// function and all the operations it contains are converted to a network
/// of XLS procs.
class HandshakeToXlsPass
    : public dynamatic::impl::HandshakeToXlsBase<HandshakeToXlsPass> {
public:
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // We only support one function per module
    handshake::FuncOp funcOp = nullptr;
    for (auto op : modOp.getOps<handshake::FuncOp>()) {
      if (funcOp) {
        modOp->emitOpError()
            << "we currently only support one handshake function per module";
        return signalPassFailure();
      }
      funcOp = op;
    }

    if (!funcOp) {
      modOp->emitOpError() << "no hanshake function found";
      return signalPassFailure();
    }

    // Check that some preconditions are met before doing anything
    if (failed(verifyIRMaterialized(funcOp))) {
      funcOp.emitError() << ERR_NON_MATERIALIZED_FUNC;
      return signalPassFailure();
    }

    OpBuilder builder(ctx);
    ProcManager procs;

    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertFunc>(ctx);

    // Basic Ops:
    patterns.insert<ConvertToXlsSpawn<ForkOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<JoinOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<SelectOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<MuxOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ControlMergeOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ConditionalBranchOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<SourceOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<SinkOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ConstantOp>>(ctx, procs);

    // BinaryIOp:
    patterns.insert<ConvertToXlsSpawn<AndIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<AddIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<DivSIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<DivUIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<MulIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<OrIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<XOrIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ShLIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ShRSIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ShRUIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<SubIOp>>(ctx, procs);

    // Comparison Ops:
    patterns.insert<ConvertToXlsSpawn<CmpIOp>>(ctx, procs);

    patterns.insert<ConvertToXlsSpawn<ExtSIOp>>(ctx, procs);
    patterns.insert<ConvertToXlsSpawn<ExtUIOp>>(ctx, procs);

    patterns.insert<ConvertBuffer>(ctx);
    patterns.insert<ConvertEnd>(ctx);

    // Everything must be converted to operations in the xls dialect
    ConversionTarget target(*ctx);
    target.addIllegalDialect<handshake::HandshakeDialect>();
    target.addLegalDialect<mlir::xls::XlsDialect>();

    // Convert all Ops:
    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();

    // Create all required handshake procs:
    builder.setInsertionPointToStart(modOp.getBody(0));
    procs.buildRequiredProcs(builder);
  }
};

} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeToXlsPass() {
  return std::make_unique<HandshakeToXlsPass>();
}
