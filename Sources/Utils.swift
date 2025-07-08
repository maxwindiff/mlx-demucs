import MLX

func formatFloat(_ value: Float, decimals: Int = 6) -> String {
  return String(format: "%.\(decimals)f", value)
}

func formatComplex(_ complex: MLXArray, decimals: Int = 6) -> String {
  let real: Float = complex.realPart().item()
  let imag: Float = complex.imaginaryPart().item()
  let realStr = formatFloat(real, decimals: decimals)
  let imagStr = formatFloat(abs(imag), decimals: decimals)
  let sign = imag >= 0 ? "+" : "-"
  return "\(realStr)\(sign)\(imagStr)j"
}

func formatArray(_ array: MLXArray, _ count: Int, decimals: Int = 6) -> String {
  // Check if array is complex
  if array.dtype == .complex64 {
    // Handle complex arrays
    let flatArray = array.flattened()
    let total = flatArray.shape[0]

    if total <= 2 * count {
      // Show all elements
      let complexStrings = (0..<total).map { i in
        formatComplex(flatArray[i], decimals: decimals)
      }
      return "[\(complexStrings.joined(separator: ", "))]"
    } else {
      // Show first and last elements
      let first = (0..<count).map { i in
        formatComplex(flatArray[i], decimals: decimals)
      }
      let last = ((total - count)..<total).map { i in
        formatComplex(flatArray[i], decimals: decimals)
      }
      return "[\(first.joined(separator: ", ")), ..., \(last.joined(separator: ", "))]"
    }
  } else {
    // Handle real arrays
    let elems = array.flattened().asArray(Float.self)
    let total = elems.count
    if total <= 2 * count {
      return "[\(elems.map { formatFloat($0, decimals: decimals) }.joined(separator: ", "))]"
    } else {
      let first = elems.prefix(count).map { formatFloat($0, decimals: decimals) }
      let last = elems.suffix(count).map { formatFloat($0, decimals: decimals) }
      return "[\(first.joined(separator: ", ")), ..., \(last.joined(separator: ", "))]"
    }
  }
}

func formatStats(_ array: MLXArray, label: String = "") -> String {
  let prefix = label.isEmpty ? "" : "\(label) - "
  if array.dtype == .complex64 {
    return "\(prefix)\(array.shape), mean: \(formatComplex(MLX.mean(array))), std: \(formatComplex(MLX.std(array)))"
  } else {
    return "\(prefix)\(array.shape), mean: \(formatFloat(MLX.mean(array).item())), std: \(formatFloat(MLX.std(array).item()))"
  }
}

func printDebug(_ array: MLXArray, _ label: String = "", _ count: Int = 5, decimals: Int = 6) {
  let prefix = label.isEmpty ? "" : "\(label) - "
  print("\(prefix)\(formatStats(array)) = \(formatArray(array, count, decimals: decimals))")
}

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
