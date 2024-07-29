// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-canonicalize --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @eraseUnconditionalBranches(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = return %[[VAL_0]] : <>
// CHECK:           end %[[VAL_1]] : <>
// CHECK:         }
handshake.func @eraseUnconditionalBranches(%start: !handshake.control<>) -> !handshake.control<> {
  %br = br %start : <>
  %returnVal = return %br : <>
  end %returnVal : <>
}

// -----

// CHECK-LABEL:   handshake.func @eraseSingleInputMerges(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !handshake.channel<i32>, %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]], %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_4:.*]] = addi %[[VAL_0]], %[[VAL_3]] : <i32>
// CHECK:           %[[VAL_5:.*]] = return %[[VAL_4]] : <i32>
// CHECK:           end %[[VAL_5]] : <i32>
// CHECK:         }
handshake.func @eraseSingleInputMerges(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %merge1 = merge %arg0 : <i32>
  %merge2 = merge %arg0, %arg1 : <i32>
  %add = handshake.addi %merge1, %merge2 : <i32>
  %returnVal = return %add : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @eraseSingleInputMuxes(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !handshake.channel<i32>, %[[VAL_1:.*]]: !handshake.channel<i32>, %[[VAL_2:.*]]: !handshake.channel<i1>,
// CHECK-SAME:                                          %[[VAL_3:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "cond", "start"], resNames = ["out0"]} {
// CHECK:           sink %[[VAL_2]] : <i1>
// CHECK:           %[[VAL_4:.*]] = mux %[[VAL_2]] {{\[}}%[[VAL_0]], %[[VAL_1]]] {handshake.bb = 0 : ui32} : <i1>, <i32>
// CHECK:           %[[VAL_5:.*]] = addi %[[VAL_0]], %[[VAL_4]] : <i32>
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : <i32>
// CHECK:           end %[[VAL_6]] : <i32>
// CHECK:         }
handshake.func @eraseSingleInputMuxes(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %cond: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %mux1 = mux %cond [%arg0] {handshake.bb = 0 : ui32} : <i1>, <i32>
  %mux2 = mux %cond [%arg0, %arg1] {handshake.bb = 0 : ui32} : <i1>, <i32>
  %add = handshake.addi %mux1, %mux2 : <i32>
  %returnVal = return %add : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @eraseSingleControlMerges(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !handshake.channel<i32>, %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                                             %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = source {handshake.bb = 0 : ui32}
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {handshake.bb = 0 : ui32, value = 0 : i32} : <i32>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = control_merge %[[VAL_0]], %[[VAL_1]]  {handshake.bb = 0 : ui32} : <i32>, <i32>
// CHECK:           %[[VAL_7:.*]] = addi %[[VAL_0]], %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_8:.*]] = addi %[[VAL_7]], %[[VAL_5]] : <i32>
// CHECK:           %[[VAL_9:.*]] = addi %[[VAL_4]], %[[VAL_6]] : <i32>
// CHECK:           %[[VAL_10:.*]] = addi %[[VAL_8]], %[[VAL_9]] : <i32>
// CHECK:           %[[VAL_11:.*]] = return %[[VAL_10]] : <i32>
// CHECK:           end %[[VAL_11]] : <i32>
// CHECK:         }
handshake.func @eraseSingleControlMerges(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %cmergeData1, %cmergeIndex1 = control_merge %arg0 {handshake.bb = 0 : ui32} : <i32>, <i32>
  %cmergeData2, %cmergeIndex2 = control_merge %arg1 {handshake.bb = 0 : ui32}: <i32>, <i32>
  %cmergeData3, %cmergeIndex3 = control_merge %arg0, %arg1 {handshake.bb = 0 : ui32} : <i32>, <i32>
  %addData1 = handshake.addi %cmergeData1, %cmergeData2 : <i32>
  %addData2 = handshake.addi %addData1, %cmergeData3 : <i32>
  %addIndex = handshake.addi %cmergeIndex1, %cmergeIndex3 : <i32>
  %add = handshake.addi %addData2, %addIndex : <i32>
  %returnVal = return %add : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @downgradeIndexlessControlMerge(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !handshake.channel<i32>, %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]], %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32>
// CHECK:           %[[VAL_4:.*]] = return %[[VAL_3]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @downgradeIndexlessControlMerge(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %cmergeData, %cmergeIndex = control_merge %arg0, %arg1 {handshake.bb = 0 : ui32} : <i32>, <i32>
  %returnVal = return %cmergeData : <i32>
  end %returnVal : <i32>
}
