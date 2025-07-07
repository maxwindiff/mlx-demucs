import MLX

func pad(_ x: MLXArray, axis: Int, paddings: (Int, Int)) -> MLXArray {
  let axis = axis < 0 ? x.ndim + axis : axis
  var shapeLeft = x.shape
  var shapeRight = x.shape
  shapeLeft[axis] = paddings.0
  shapeRight[axis] = paddings.1
  return MLX.concatenated([MLXArray.zeros(shapeLeft), x, MLXArray.zeros(shapeRight)], axis: axis)
}

func padReflect(_ x: MLXArray, axis: Int, paddings: (Int, Int)) -> MLXArray {
  var x = x
  let axis = axis < 0 ? x.ndim + axis : axis
  var (paddingLeft, paddingRight) = paddings

  // Reflect padding doesn't work if the input length is less than the amount to be padded. To workaround this, we
  // insert extra 0 padding before the reflection.
  let length = x.dim(axis)
  let maxPad = max(paddingLeft, paddingRight)
  if length <= maxPad {
    let extraPad = maxPad - length + 1
    let extraPadRight = min(paddingRight, extraPad)
    let extraPadLeft = extraPad - extraPadRight
    paddingLeft -= extraPadLeft
    paddingRight -= extraPadRight
    x = pad(x, axis: axis, paddings: (extraPadLeft, extraPadRight))
  }

  if paddingLeft > 0 {
    var indices = Array(repeating: MLXSlice(), count: x.ndim)
    indices[axis] = .stride(from: paddingLeft, to: 0, by: -1)
    x = MLX.concatenated([x[indices], x], axis: axis)
  }

  if paddingRight > 0 {
    var indices = Array(repeating: MLXSlice(), count: x.ndim)
    indices[axis] = .stride(from: length, to: length - paddingRight, by: -1)
    x = MLX.concatenated([x, x[indices]], axis: axis)
  }

  return x
}
