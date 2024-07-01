// // NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// // The script is designed to make adding checks to
// // a test case fast, it is *not* designed to be authoritative
// // about what constitutes a good test! The CHECK should be
// // minimized and named to reflect the test intent.

// // The script is designed to make adding checks to
// // a test case fast, it is *not* designed to be authoritative
// // minimized and named to reflect the test intent.

// // NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// // RUN: dynamatic-opt --handshake-optimization %s --split-input-file | FileCheck %s

// handshake.func @removeBranchMUXPair_simple(%arg0: i32, %arg1: i1, %start: none) -> i32 {
//   %arg1_one, %arg1_two = fork [2] %arg1: i1
//   %true, %false = cond_br %arg1_one , %arg0 : i32
//   %result_mux= mux %arg1_two [%true, %false]: i1, i32
//   end %result_mux : i32
// }

// handshake.func @removeBranchMergePairloop_simple_case1(%arg0: i32, %arg1: i1, %start: none) -> (i32, index) {
//   %result_cmerge, %index = control_merge %true, %arg0: i32, index
//   %true, %false = cond_br %arg1, %result_cmerge : i32
//   end %false, %index : i32, index
// }

// handshake.func @removeBranchMergePairloop_simple_case2(%arg0: i32, %arg1: i1, %start: none) -> (i32, index) {
//   %result_cmerge, %index= control_merge %arg0, %false: i32, index
//   %true, %false = cond_br %arg1, %result_cmerge : i32
//   end %true, %index : i32, index
// }

// handshake.func @removeBranchMergePairloop_simple_case3(%arg0: i32, %arg1: i1, %start: none) -> (i32, index) {
//   %result_cmerge, %index= control_merge %arg0, %true: i32, index
//   %true, %false = cond_br %arg1, %result_cmerge : i32
//   end %false, %index : i32, index
// }

// handshake.func @removeBranchMergePairloop_simple_case4(%arg0: i32, %arg1: i1, %start: none) -> (i32, index) {
//   %result_cmerge, %index= control_merge %false, %arg0: i32, index
//   %true, %false = cond_br %arg1, %result_cmerge : i32
//   end %true, %index : i32, index
// }

// handshake.func @removeBranchCMergePairloop_multiple_doNot(%arg0: i32, %arg1: i1, %arg2: i1, %start: none) -> (i32, index) {
//   %result_cmerge_1, %index1= control_merge %true_2, %arg0: i32, index
//   %result_cmerge_2, %index2= control_merge %false_1, %result_cmerge_1: i32, index
//   %true_1, %false_1 = cond_br %arg1, %result_cmerge_2 : i32
//   %num_1 = arith.constant 1 : i32
//   %data_in = arith.addi %num_1, %true_1 : i32
//   %true_2, %false_2 = cond_br %arg2, %data_in : i32
//   end %false_2,%index2 : i32, index
// }

// handshake.func @removeBothCMerge(%arg0: i32, %arg1: i1, %arg2: i1, %arg3: i1, %start: none) -> (i32, index, index) {
//   %result_cmerge_1, %index1= control_merge %true_2, %arg0: i32, index
//   %result_cmerge_2, %index2= control_merge %false_1, %result_cmerge_1: i32, index
//   %true_1, %false_1 = cond_br %arg2, %result_cmerge_2 : i32
//   %true_2, %false_2 = cond_br %arg3, %true_1 : i32
//   end %false_2, %index1, %index2 : i32, index, index
// }

// handshake.func @removeBoth_doNot(%arg0: i32, %arg1: i1, %arg2: i1, %arg3: i1, %start: none) -> i32 {
//   %result_mux= mux %arg1 [%true_2, %arg0]: i1, i32
//   %result_cmerge_2, %index2= control_merge %false_1, %result_mux: i32, index
//   %true_1, %false_1 = cond_br %arg2, %result_cmerge_2 : i32
//   %num_2 = arith.constant 2: i32
//   %data_in= arith.muli %true, %num_2: i32
//   %true_2, %false_2 = cond_br %data_in, %false_1 : i32
//   end %false_2 : i32
// }

// // handshake.func @exampleBranchMUXPairloop(%x: i32, %j: i32, %c1: i1, %c2: i1) -> i32 {
// //   %c1_one, %c1_two = fork [2] %c1 : i1
// //   %c2_one, %c2_two = fork [2] %c2 : i1
// //   %result_mux_1= mux %c1_one [%x, %false]: i1, i32
// //   %true, %false = cond_br %c1_two, %result_mux_1 : i32
// //   %result_mux_2= mux %c2_one [%true, %false_2]: i1, i32
// //   %result_mux_2one, %result_mux_2two = fork [2] %result_mux_2 : i32
// //   %true_2, %false_2 = cond_br %c2_two, %result_mux_2two : i32
// //   sink %true_2 : i32
// //   %sta = arith.addi %j, %result_mux_2one: i32  
// //   end %sta : i32
// // }









// // handshake.func @trial(%arg0: i32, %arg2: i32, %start: none) -> (i32, index) {
  
// //   %result_mux= mux true, [%arg0, %arg2] : i32
// //   end %result_mux : i32
// // }
// handshake.func @removeSupressFork (%arg0: i1, %arg1: i32, %start: none) -> (i32, i32){
//   %true, %false= cond_br %arg0, %arg1: i32
//   %one, %two = fork [2] %false : i32
//   end %one, %two: i32, i32
// }

// handshake.func @removeSupressFork2 (%arg0: i32, %arg1: i1, %start: none) -> (i32, i32){
//   %true, %false= cond_br %arg1, %arg0: i32
//   %num1 = arith.constant 1: i32
//   %answer= arith.addi %false, %num1 : i32
//   %one, %two = fork [2] %answer: i32
//   end %one, %two: i32, i32
// }

// handshake.func @removeSupressSupressPairs (%arg0: i32, %arg1: i1, %arg2: i1, %start: none) -> i32{
//   %true, %false= cond_br %arg1, %arg0: i32
//   %true2, %false2= cond_br %arg2, %false : i32
//   end %false2: i32
// }


// handshake.func @BranchtoForkSupressPairs (%arg0: i32, %arg1: i1, %start: none) -> (i32, i32){
//   %true, %false= cond_br %arg1, %arg0: i32
//   end %true, %false: i32, i32
// }



handshake.func @removeForkForkPair(%arg0: i32, %start: none) -> (i32, i32, i32) {
  %one, %two = fork [2] %arg0: i32
  %two1, %two2 = fork [2] %two: i32
  end %one, %two1, %two2: i32, i32, i32
}

// handshake.func @removeForkForkPairMultiple(%arg0: i32, %start: none) -> (i32, i32, i32) {
//   %one, %two, %three = fork [3] %arg0: i32
//   %two1, %two2 = fork [2] %two: i32
//   %three1, %three2, %three3, %three4 = fork [4] %three: i32
//   end %one, %two1, %two2, %three1, %three2, %three3, %three4 : i32, i32, i32, i32, i32, i32, i32
// }

// handshake.func @removeForkForkPairdonot(%arg0: i32, %start: none) -> (i32, i32, i32) {
//   %one, %two, %three = fork [3] %arg0: i32
//   %num1= arith.constant 1: i32
//   %ans = arith.addi %two, %num1 : i32
//   %two1, %two2 = fork [2] %ans: i32
//   %three1, %three2, %three3, %three4 = fork [4] %three: i32
//   end %one, %two1, %two2, %three1, %three2, %three3, %three4 : i32, i32, i32, i32, i32, i32, i32
// }

// handshake.func @removeForkForkPairNested(%arg0: i32, %start: none) -> (i32, i32, i32) {
//   %one, %two= fork [2] %arg0: i32
//   %two1, %two2 = fork [2] %two: i32
//   %three1, %three2, %three3, %three4 = fork [4] %two1: i32
//   end %one, %two2, %three1, %three2, %three3, %three4 : i32, i32, i32, i32, i32, i32
// }

// handshake.func @removeForkForkPairNested_only1(%arg0: i32, %start: none) -> (i32, i32, i32) {
//   %one, %two= fork [2] %arg0: i32
//   %two1, %two2 = fork [2] %two: i32
//   %num1= arith.constant 1: i32
//   %ans = arith.addi %two1, %num1 : i32
//   %three1, %three2, %three3, %three4 = fork [4] %ans: i32
//   end %one, %three1, %three2, %three3, %three4 : i32, i32, i32, i32, i32
// }

// handshake.func @removeBranchMUXPair_multiple(%arg0: i32, %arg1: i1, %arg2: i1, %start: none) -> i32 {
//   %arg1_one, %arg1_two = fork [2] %arg1: i1
//   %arg2_one, %arg2_two = fork [2] %arg2: i1
//   %true, %false = cond_br %arg1_one, %arg0 : i32
//   %true_2, %false_2 = cond_br %arg2_one, %true : i32
//   %result_mux= mux %arg2_two [%true_2, %false_2]: i1, i32
//   %result_mux_2= mux %arg1_two [%result_mux, %false]: i1, i32
//   end %result_mux_2 : i32
// }

// handshake.func @removeForkSuppressMUX(%arg0: i32, %arg1: i1, %start: none) -> i32 {
//   %one, %two= fork [2] %arg0: i32
//   %one1, %two1, %three1 = fork [3] %arg1: i1
//   %true, %false= cond_br %one1, %one: i32
//   %num1= arith.constant 1: i1
//   %ans = arith.xori %two1, %num1 : i1
//   %true_2, %false_2= cond_br %ans, %two: i32
//   %c = mux %three1 [%false_2, %false]:i1, i32
//   end %c: i32
// }
