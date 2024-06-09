// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @selfLoop(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                             %[[VAL_2:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2"], resNames = ["end"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_1]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_2]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_3]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_4]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_5]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_9:.*]] = mux %[[VAL_10:.*]] {{\[}}%[[VAL_11:.*]], %[[VAL_6]]] {handshake.bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_10]] {{\[}}%[[VAL_13:.*]], %[[VAL_7]]] {handshake.bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_10]] = control_merge %[[VAL_15:.*]], %[[VAL_8]] {handshake.bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_12]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_11]], %[[VAL_17:.*]] = cond_br %[[VAL_16]], %[[VAL_9]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_13]], %[[VAL_18:.*]] = cond_br %[[VAL_16]], %[[VAL_12]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_15]], %[[VAL_19:.*]] = cond_br %[[VAL_16]], %[[VAL_14]] {handshake.bb = 1 : ui32} : none
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = control_merge %[[VAL_19]] {handshake.bb = 2 : ui32} : none, index
// CHECK:           %[[VAL_22:.*]] = return {handshake.bb = 2 : ui32} %[[VAL_20]] : none
// CHECK:           end {handshake.bb = 2 : ui32} %[[VAL_22]] : none
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
// CHECK-SAME:                                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                     %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32,
// CHECK-SAME:                                     %[[VAL_3:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2", "in3"], resNames = ["end"]} {
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] {handshake.bb = 0 : ui32} : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_3]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_4]], %[[VAL_5]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_4]], %[[VAL_6]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_4]], %[[VAL_7]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_15:.*]] {{\[}}%[[VAL_8]], %[[VAL_10]]] {handshake.bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_15]] {{\[}}%[[VAL_10]], %[[VAL_10]]] {handshake.bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_15]] {{\[}}%[[VAL_8]], %[[VAL_10]]] {handshake.bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_15]] = control_merge %[[VAL_12]], %[[VAL_12]] {handshake.bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_19:.*]] = return {handshake.bb = 1 : ui32} %[[VAL_18]] : none
// CHECK:           end {handshake.bb = 1 : ui32} %[[VAL_19]] : none
// CHECK:         }
func.func @duplicateLiveOut(%arg0: i1, %arg1: i32, %arg2: i32) {
  cf.cond_br %arg0, ^bb1(%arg1, %arg2, %arg1: i32, i32, i32), ^bb1(%arg2, %arg2, %arg2: i32, i32, i32)
  ^bb1(%0: i32, %1: i32, %2: i32):
    return
}

// ----

// CHECK-LABEL:   handshake.func @divergeSameArg(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                   %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                   %[[VAL_2:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2"], resNames = ["end"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] {handshake.bb = 0 : ui32} : i1
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_1]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_2]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_4]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_6]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = control_merge %[[VAL_8]] {handshake.bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_11]] {handshake.bb = 1 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_7]] {handshake.bb = 2 : ui32} : i32
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_9]] {handshake.bb = 2 : ui32} : none, index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_15]] {handshake.bb = 2 : ui32} : none
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = control_merge %[[VAL_17]], %[[VAL_13]] {handshake.bb = 3 : ui32} : none, index
// CHECK:           %[[VAL_20:.*]] = return {handshake.bb = 3 : ui32} %[[VAL_18]] : none
// CHECK:           end {handshake.bb = 3 : ui32} %[[VAL_20]] : none
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
