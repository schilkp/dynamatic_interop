// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-set-buffering-properties="version=fpga20" --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @mergeBufferTwoInputs(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] {handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <>
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_1]]#0, %[[VAL_1]]#1 : <>
// CHECK:           end {handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} %[[VAL_2]] : <>
// CHECK:         }
handshake.func @mergeBufferTwoInputs(%start: !handshake.control<>) -> !handshake.control<> {
  %fork:2 = fork [2] %start : <>
  %merge = merge %fork#0, %fork#1 : <>
  end %merge : <>
}

// -----

// CHECK-LABEL:   handshake.func @mcUnbuffered(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: !handshake.channel<i32>, %[[VAL_2:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["memref", "addr", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = mem_controller{{\[}}%[[VAL_0]] : memref<64xi32>] %[[VAL_5:.*]]#0 (%[[VAL_6:.*]]) %[[VAL_5]]#1 {connectedBlocks = [0 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : (!handshake.channel<i32>) -> !handshake.channel<i32>
// CHECK:           %[[VAL_5]]:2 = fork [2] %[[VAL_2]] {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <>
// CHECK:           %[[VAL_6]], %[[VAL_7:.*]] = load{{\[}}%[[VAL_1]]] %[[VAL_3]] {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>, <i32>, <i32>, <i32>
// CHECK:           end %[[VAL_7]], %[[VAL_4]] : <i32>, <>
// CHECK:         }
handshake.func @mcUnbuffered(%memref: memref<64xi32>, %addr: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>) {
  %ldData1, %done = mem_controller [%memref: memref<64xi32>] %fork#0 (%ldAddrToMem) %fork#1 {connectedBlocks = [0 : i32]} : (!handshake.channel<i32>) -> !handshake.channel<i32>
  %fork:2 = fork [2] %start {handshake.bb = 0 : ui32} : <>
  %ldAddrToMem, %ldDataToSucc = load [%addr] %ldData1 {handshake.bb = 0 : ui32} : <i32>, <i32>, <i32>, <i32>
  end %ldDataToSucc, %done : <i32>, <>
}

// -----

// CHECK-LABEL:   handshake.func @lsqUnbuffered(
// CHECK-SAME:                                  %[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: !handshake.channel<i32>, %[[VAL_2:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["memref", "addr", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_3:.*]]:2 = lsq{{\[}}%[[VAL_0]] : memref<64xi32>] (%[[VAL_4:.*]]#0, %[[VAL_4]]#1, %[[VAL_5:.*]], %[[VAL_4]]#2)  {groupSizes = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
// CHECK:           %[[VAL_4]]:3 = fork [3] %[[VAL_2]] {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <>
// CHECK:           %[[VAL_5]], %[[VAL_6:.*]] = load{{\[}}%[[VAL_1]]] %[[VAL_3]]#0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>, <i32>, <i32>, <i32>
// CHECK:           end %[[VAL_6]], %[[VAL_3]]#1 : <i32>, <>
// CHECK:         }
handshake.func @lsqUnbuffered(%memref: memref<64xi32>, %addr: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>) {
  %ldData1, %done = lsq [%memref: memref<64xi32>] (%fork#0, %fork#1, %ldAddrToMem, %fork#2) {groupSizes = [1 : i32]} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
  %fork:3 = fork [3] %start {handshake.bb = 0 : ui32} : <>
  %ldAddrToMem, %ldDataToSucc = load [%addr] %ldData1 {handshake.bb = 0 : ui32} : <i32>, <i32>, <i32>, <i32>
  end %ldDataToSucc, %done : <i32>, <>
}

// -----

// CHECK-LABEL:   handshake.func @lsqBufferControlPath(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["memref", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_2:.*]]:4 = lsq{{\[}}%[[VAL_0]] : memref<64xi32>] (%[[VAL_3:.*]]#4, %[[VAL_3]]#0, %[[VAL_4:.*]], %[[VAL_5:.*]]#0, %[[VAL_6:.*]], %[[VAL_7:.*]]#0, %[[VAL_8:.*]], %[[VAL_7]]#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_1]], %[[VAL_5]]#2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [1,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <>
// CHECK:           %[[VAL_3]]:5 = lazy_fork [5] %[[VAL_9]] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <>
// CHECK:           %[[VAL_10:.*]] = constant %[[VAL_3]]#1 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, value = false} : <>, <i1>
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_3]]#2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, value = 0 : i32} : <>, <i32>
// CHECK:           %[[VAL_4]], %[[VAL_12:.*]] = load{{\[}}%[[VAL_11]]] %[[VAL_2]]#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>, <i32>, <i32>, <i32>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_10]], %[[VAL_3]]#3 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i1>, <>
// CHECK:           sink %[[VAL_12]] : <i32>
// CHECK:           %[[VAL_5]]:3 = lazy_fork [3] %[[VAL_13]] {handshake.bb = 2 : ui32} : <>
// CHECK:           %[[VAL_15:.*]] = constant %[[VAL_5]]#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, value = 1 : i32} : <>, <i32>
// CHECK:           %[[VAL_6]], %[[VAL_16:.*]] = load{{\[}}%[[VAL_15]]] %[[VAL_2]]#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>, <i32>, <i32>, <i32>
// CHECK:           sink %[[VAL_16]] : <i32>
// CHECK:           %[[VAL_7]]:3 = lazy_fork [3] %[[VAL_14]] {handshake.bb = 3 : ui32} : <>
// CHECK:           %[[VAL_17:.*]] = constant %[[VAL_7]]#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, value = 2 : i32} : <>, <i32>
// CHECK:           %[[VAL_8]], %[[VAL_18:.*]] = load{{\[}}%[[VAL_17]]] %[[VAL_2]]#2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>, <i32>, <i32>, <i32>
// CHECK:           end {handshake.bb = 3 : ui32} %[[VAL_18]], %[[VAL_2]]#3 : <i32>, <>
// CHECK:         }
handshake.func @lsqBufferControlPath(%memref: memref<64xi32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>) {
  %ldData1, %ldData2, %ldData3, %done = lsq [%memref: memref<64xi32>] (%lazyForkCtrl1#4, %lazyForkCtrl1#0, %ldAddrToMem1, %lazyForkCtrl2#0, %ldAddrToMem2, %lazyForkCtrl3#0, %ldAddrToMem3, %lazyForkCtrl3#2) {groupSizes = [1 : i32, 1 : i32, 1 : i32]} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
// ^^bb0
// ^^bb1 (from ^^bb0, ^bb2, to ^bb2, ^bb3):
  %ctrl1 = merge %start#0, %lazyForkCtrl2#2 {handshake.bb = 1 : ui32} : <>
  %lazyForkCtrl1:5 = lazy_fork [5] %ctrl1 {handshake.bb = 1 : ui32} : <>
  %cond = constant %lazyForkCtrl1#1 {value = 0 : i1, handshake.bb = 1 : ui32} : <>, <i1>
  %addr1 = constant %lazyForkCtrl1#2 {value = 0 : i32, handshake.bb = 1 : ui32} : <>, <i32>
  %ldAddrToMem1, %ldDataToSucc1 = load [%addr1] %ldData1 {handshake.bb = 1 : ui32} : <i32>, <i32>, <i32>, <i32>
  %ctrl1To2, %ctrl1To3 = cond_br %cond, %lazyForkCtrl1#3 {handshake.bb = 1 : ui32} : <i1>, <>
  sink %ldDataToSucc1 : <i32>
// ^^bb2 (from ^^bb1, to ^^bb1):
  %lazyForkCtrl2:3 = lazy_fork [3] %ctrl1To2 {handshake.bb = 2 : ui32} : <>
  %addr2 = constant %lazyForkCtrl2#1 {value = 1 : i32, handshake.bb = 2 : ui32} : <>, <i32>
  %ldAddrToMem2, %ldDataToSucc2 = load [%addr2] %ldData2 {handshake.bb = 2 : ui32} : <i32>, <i32>, <i32>, <i32>
  sink %ldDataToSucc2 : <i32>
// ^^bb3:
  %lazyForkCtrl3:3 = lazy_fork [3] %ctrl1To3 {handshake.bb = 3 : ui32} : <>
  %addr3 = constant %lazyForkCtrl3#1 {value = 2 : i32, handshake.bb = 3 : ui32} : <>, <i32>
  %ldAddrToMem3, %ldDataToSucc3 = load [%addr3] %ldData3 {handshake.bb = 3 : ui32} : <i32>, <i32>, <i32>, <i32>
  end {handshake.bb = 3 : ui32} %ldDataToSucc3, %done : <i32>, <>
}
