// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @selfLoop(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i32>, %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                             %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "start"], resNames = ["end"]} {
// CHECK:           %[[VAL_3:.*]] = br %[[VAL_0]] {handshake.bb = 0 : ui32} : <i32>
// CHECK:           %[[VAL_4:.*]] = br %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32>
// CHECK:           %[[VAL_5:.*]] = br %[[VAL_2]] {handshake.bb = 0 : ui32} : <>
// CHECK:           %[[VAL_6:.*]] = mux %[[VAL_7:.*]] {{\[}}%[[VAL_3]], %[[VAL_8:.*]]] {handshake.bb = 1 : ui32} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:           %[[VAL_9:.*]] = mux %[[VAL_7]] {{\[}}%[[VAL_4]], %[[VAL_10:.*]]] {handshake.bb = 1 : ui32} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:           %[[VAL_11:.*]], %[[VAL_7]] = control_merge %[[VAL_5]], %[[VAL_12:.*]]  {handshake.bb = 1 : ui32} : [<>, <>] to <>, <i1>
// CHECK:           %[[VAL_13:.*]] = cmpi eq, %[[VAL_6]], %[[VAL_9]] {handshake.bb = 1 : ui32} : <i32>
// CHECK:           %[[VAL_8]], %[[VAL_14:.*]] = cond_br %[[VAL_13]], %[[VAL_6]] {handshake.bb = 1 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_10]], %[[VAL_15:.*]] = cond_br %[[VAL_13]], %[[VAL_9]] {handshake.bb = 1 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_12]], %[[VAL_16:.*]] = cond_br %[[VAL_13]], %[[VAL_11]] {handshake.bb = 1 : ui32} : <i1>, <>
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_16]]  {handshake.bb = 2 : ui32} : [<>] to <>, <i1>
// CHECK:           end {handshake.bb = 2 : ui32} %[[VAL_2]] : <>
// CHECK:         }
func.func @selfLoop(%arg0: i32, %arg1: i32) {
  cf.br ^bb1(%arg0: i32)
  ^bb1(%0: i32):
    %1 = arith.cmpi eq, %0, %arg1: i32
    cf.cond_br %1, ^bb1(%0: i32), ^bb2
  ^bb2:
    return
}

// -----

// CHECK-LABEL:   handshake.func @duplicateLiveOut(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !handshake.channel<i1>, %[[VAL_1:.*]]: !handshake.channel<i32>, %[[VAL_2:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "in2", "start"], resNames = ["end"]} {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_0]], %[[VAL_1]] {handshake.bb = 0 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_0]], %[[VAL_2]] {handshake.bb = 0 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_0]], %[[VAL_3]] {handshake.bb = 0 : ui32} : <i1>, <>
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_11:.*]] {{\[}}%[[VAL_4]], %[[VAL_6]]] {handshake.bb = 1 : ui32} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_6]], %[[VAL_6]]] {handshake.bb = 1 : ui32} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_4]], %[[VAL_6]]] {handshake.bb = 1 : ui32} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:           %[[VAL_14:.*]], %[[VAL_11]] = control_merge %[[VAL_8]], %[[VAL_8]]  {handshake.bb = 1 : ui32} : [<>, <>] to <>, <i1>
// CHECK:           end {handshake.bb = 1 : ui32} %[[VAL_3]] : <>
// CHECK:         }
func.func @duplicateLiveOut(%arg0: i1, %arg1: i32, %arg2: i32) {
  cf.cond_br %arg0, ^bb1(%arg1, %arg2, %arg1: i32, i32, i32), ^bb1(%arg2, %arg2, %arg2: i32, i32, i32)
  ^bb1(%0: i32, %1: i32, %2: i32):
    return
}

// ----

// CHECK-LABEL:   handshake.func @divergeSameArg(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !handshake.channel<i1>, %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "start"], resNames = ["end"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = cond_br %[[VAL_0]], %[[VAL_1]] {handshake.bb = 0 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_0]], %[[VAL_2]] {handshake.bb = 0 : ui32} : <i1>, <>
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_3]] {handshake.bb = 1 : ui32} : <i32>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = control_merge %[[VAL_5]]  {handshake.bb = 1 : ui32} : [<>] to <>, <i1>
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_8]] {handshake.bb = 1 : ui32} : <>
// CHECK:           %[[VAL_11:.*]] = merge %[[VAL_4]] {handshake.bb = 2 : ui32} : <i32>
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = control_merge %[[VAL_6]]  {handshake.bb = 2 : ui32} : [<>] to <>, <i1>
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_12]] {handshake.bb = 2 : ui32} : <>
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_10]], %[[VAL_14]]  {handshake.bb = 3 : ui32} : [<>, <>] to <>, <i1>
// CHECK:           end {handshake.bb = 3 : ui32} %[[VAL_2]] : <>
// CHECK:         }
func.func @divergeSameArg(%arg0: i1, %arg1: i32) {
  cf.cond_br %arg0, ^bb1(%arg1: i32), ^bb2(%arg1: i32)
  ^bb1(%0: i32):
    cf.br ^bb3
  ^bb2(%1: i32):
    cf.br ^bb3
  ^bb3:
    return
}
