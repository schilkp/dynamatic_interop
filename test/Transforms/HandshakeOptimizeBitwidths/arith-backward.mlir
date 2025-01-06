// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-optimize-bitwidths --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @addiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32> to <i16>
// CHECK:           %[[VAL_4:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_5:.*]] = addi %[[VAL_4]], %[[VAL_3]] : <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @addiBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %res = addi %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @subiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32> to <i16>
// CHECK:           %[[VAL_4:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_5:.*]] = subi %[[VAL_4]], %[[VAL_3]] : <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @subiBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %res = subi %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @muliBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_0]] : <i8> to <i32>
// CHECK:           %[[VAL_4:.*]] = muli %[[VAL_3]], %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_5:.*]] = trunci %[[VAL_4]] : <i32> to <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @muliBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %res = muli %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @andiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32> to <i8>
// CHECK:           %[[VAL_4:.*]] = andi %[[VAL_0]], %[[VAL_3]] : <i8>
// CHECK:           %[[VAL_5:.*]] = extui %[[VAL_4]] : <i8> to <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @andiBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extui %arg0 : <i8> to <i32>
  %res = andi %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @oriBW(
// CHECK-SAME:                          %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                          %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                          %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = extui %[[VAL_0]] : <i8> to <i32>
// CHECK:           %[[VAL_4:.*]] = ori %[[VAL_3]], %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_5:.*]] = trunci %[[VAL_4]] : <i32> to <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @oriBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extui %arg0 : <i8> to <i32>
  %res = ori %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @xoriBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = extui %[[VAL_0]] : <i8> to <i32>
// CHECK:           %[[VAL_4:.*]] = xori %[[VAL_3]], %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_5:.*]] = trunci %[[VAL_4]] : <i32> to <i16>
// CHECK:           end %[[VAL_5]] : <i16>
// CHECK:         }
handshake.func @xoriBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extui %arg0 : <i8> to <i32>
  %res = xori %ext0, %arg1 : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @shliBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : <i32> to <i12>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]] {handshake.bb = 0 : ui32} : <i12> to <i16>
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]] {value = 4 : i32} : <>, <i32>
// CHECK:           %[[VAL_5:.*]] = trunci %[[VAL_4]] : <i32> to <i16>
// CHECK:           %[[VAL_6:.*]] = shli %[[VAL_3]], %[[VAL_5]] : <i16>
// CHECK:           end %[[VAL_6]] : <i16>
// CHECK:         }
handshake.func @shliBW(%arg0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %cst = handshake.constant %start {value = 4 : i32} : <>, <i32>
  %res = shli %arg0, %cst : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @shrsiBW(
// CHECK-SAME:                            %[[VAL_0:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : <i32> to <i20>
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]] {value = 4 : i32} : <>, <i32>
// CHECK:           %[[VAL_4:.*]] = trunci %[[VAL_3]] : <i32> to <i20>
// CHECK:           %[[VAL_5:.*]] = shrsi %[[VAL_2]], %[[VAL_4]] : <i20>
// CHECK:           %[[VAL_6:.*]] = trunci %[[VAL_5]] : <i20> to <i16>
// CHECK:           end %[[VAL_6]] : <i16>
// CHECK:         }
handshake.func @shrsiBW(%arg0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %cst = handshake.constant %start {value = 4 : i32} : <>, <i32>
  %res = shrsi %arg0, %cst : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @shruiBW(
// CHECK-SAME:                            %[[VAL_0:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : <i32> to <i20>
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]] {value = 4 : i32} : <>, <i32>
// CHECK:           %[[VAL_4:.*]] = trunci %[[VAL_3]] : <i32> to <i20>
// CHECK:           %[[VAL_5:.*]] = shrui %[[VAL_2]], %[[VAL_4]] : <i20>
// CHECK:           %[[VAL_6:.*]] = trunci %[[VAL_5]] : <i20> to <i16>
// CHECK:           end %[[VAL_6]] : <i16>
// CHECK:         }
handshake.func @shruiBW(%arg0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %cst = handshake.constant %start {value = 4 : i32} : <>, <i32>
  %res = shrui %arg0, %cst : <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}

// -----

// CHECK-LABEL:   handshake.func @selectBW(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                             %[[VAL_2:.*]]: !handshake.channel<i1>,
// CHECK-SAME:                             %[[VAL_3:.*]]: !handshake.control<>, ...) -> !handshake.channel<i16> attributes {argNames = ["arg0", "arg1", "select", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_4:.*]] = trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : <i32> to <i16>
// CHECK:           %[[VAL_5:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_6:.*]] = select %[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_4]]] : <i1>, <i16>
// CHECK:           end %[[VAL_6]] : <i16>
// CHECK:         }
handshake.func @selectBW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i32>, %select: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i16> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %res = select %select [%ext0, %arg1] : <i1>, <i32>
  %trunc = trunci %res : <i32> to <i16>
  end %trunc : <i16>
}
