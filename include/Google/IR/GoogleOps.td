//===- GoogleOps.td - Google dialect ops -------------------*- tablegen -*-===//
//
// Google Dialect Operations
//
//===----------------------------------------------------------------------===//

#ifndef GOOGLE_OPS
#define GOOGLE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

//===----------------------------------------------------------------------===//
// Google Dialect Definition
//===----------------------------------------------------------------------===//

def Google_Dialect : Dialect {
  let name = "google";
  let summary = "Google dialect for custom operations";
  let description = [{
    This dialect contains custom operations for Google-specific IR.
  }];
  let cppNamespace = "::mlir::google";
}

//===----------------------------------------------------------------------===//
// Reduction Kind Enum
//===----------------------------------------------------------------------===//

def Google_ReductionKind : I32EnumAttr<"ReductionKind",
    "Reduction operation kind", [
  I32EnumAttrCase<"MAX", 0, "max">,
  I32EnumAttrCase<"MIN", 1, "min">,
  I32EnumAttrCase<"SUM", 2, "sum">,
  I32EnumAttrCase<"PRODUCT", 3, "product">,
  //I32EnumAttrCase<"AND", 4, "and">,
  //I32EnumAttrCase<"OR", 5, "or">,
  //I32EnumAttrCase<"XOR", 6, "xor">,
  I32EnumAttrCase<"MEAN", 7, "mean">,
  I32EnumAttrCase<"ARGMAX", 8, "argmax">,
  I32EnumAttrCase<"ARGMIN", 9, "argmin">
]> {
  let cppNamespace = "::mlir::google";
}

//===----------------------------------------------------------------------===//
// Base Google Operation Definition
//===----------------------------------------------------------------------===//

class Google_Op<string mnemonic, list<Trait> traits = []> :
    Op<Google_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base Unary Operation
//===----------------------------------------------------------------------===//

class Google_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Google_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Base Binary Operation
//===----------------------------------------------------------------------===//

class Google_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Google_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Google Operations
//===----------------------------------------------------------------------===//

def Google_ConstantOp : Google_Op<"constant", [Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Creates a constant tensor with a specified value.
    
    Example:
    ```mlir
    %0 = google.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    ```
  }];
  
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$resultType)>
  ];

  let hasVerifier = 1;

}

def Google_AddOp : Google_BinaryOp<"add", [Commutative]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.
    
    Example:
    ```mlir
    %result = google.add %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_MaxOp : Google_BinaryOp<"max", [Commutative]> {
  let summary = "Element-wise maximum operation";
  let description = [{
    Performs element-wise maximum of two tensors.
    
    Example:
    ```mlir
    %result = google.max %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_MinOp : Google_BinaryOp<"min", [Commutative]> {
  let summary = "Element-wise minimum operation";
  let description = [{
    Performs element-wise minimum of two tensors.
    
    Example:
    ```mlir
    %result = google.min %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_ReduceOp : Google_Op<"reduce", [Pure]> {
  let summary = "Generic reduction operation";
  let description = [{
    Performs reduction operation across specified dimensions of a tensor.
    The reduction kind is specified by the 'kind' attribute.
    
    Examples:
    ```mlir
    // Reduce all dimensions with max
    %result = google.reduce<max> %input : tensor<4x8xf32> -> tensor<f32>
    
    // Reduce along specific axes with sum
    %result = google.reduce<sum> %input axes = [0] : tensor<4x8xf32> -> tensor<8xf32>
    
    // Keep reduced dimensions with min
    %result = google.reduce<min> %input axes = [1] keepdims = true : 
              tensor<4x8xf32> -> tensor<4x1xf32>
    
    // Product reduction
    %result = google.reduce<product> %input : tensor<4x8xf32> -> tensor<f32>
    ```
  }];
  
  let arguments = (ins 
    Google_ReductionKind:$kind,
    AnyTensor:$input,
    OptionalAttr<I64ArrayAttr>:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    `<` $kind `>` $input (`axes` `=` $axes^)? (`keepdims` `=` $keepdims^)?
    attr-dict `:` type($input) `->` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, 
                   "ArrayRef<int64_t>":$axes, "bool":$keepdims, "Type":$resultType)>
  ];
  
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

def Google_DequantOp : Google_Op<"dequant",
  [Pure,
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation",
     "generateResultTileValue",
     "getIterationDomainTileFromResultTile"]>
  ]> {
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale
  );
  let results = (outs
    AnyRankedTensor:$output
  );
  let assemblyFormat = [{
    $input `,` $scale attr-dict `:` type($input) `,` type($scale) `,` type($output)
  }];
}

#endif // GOOGLE_OPS