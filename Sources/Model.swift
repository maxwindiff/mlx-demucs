import Foundation
import MLX
import MLXNN

func printTensorStats(_ tensor: MLXArray, name: String) {
  let shape = tensor.shape
  let mean = MLX.mean(tensor)
  let variance = MLX.variance(tensor)
  let firstElements = tensor[0, 0..<10, 0]
  let elementValues = (0..<min(10, firstElements.size)).map { i in
    String(format: "% .6f", firstElements[i].item(Float.self))
  }
  print("Tensor: \(name)")
  print("  Shape: (\(shape.map(String.init).joined(separator: ", ")))")
  print(String(format: "  Mean: %.6f, Variance: %.6f", mean.item(Float.self), variance.item(Float.self)))
  print("  First elements: [\(elementValues.joined(separator: ", "))]")
  print("--------------------------------------------------")
}

class LSTM: Module {
  public var weight_ih: MLXArray
  public var weight_hh: MLXArray
  public let bias_ih: MLXArray
  public let bias_hh: MLXArray

  public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
    let scale = 1 / sqrt(Float(hiddenSize))
    self.weight_ih = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
    self.weight_hh = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
    self.bias_ih = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
    self.bias_hh = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
  }

  func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil) -> (
    MLXArray, MLXArray
  ) {
    var x = x
    x = addMM(bias_ih, x, weight_ih.T)

    var hidden: MLXArray! = hidden
    var cell: MLXArray! = cell
    var allHidden = [MLXArray]()
    var allCell = [MLXArray]()

    for index in 0 ..< x.dim(-2) {
      var ifgo = x[.ellipsis, index, 0...]
      if hidden != nil {
        ifgo = addMM(ifgo, hidden, weight_hh.T)
      }
      ifgo += bias_hh

      let pieces = split(ifgo, parts: 4, axis: -1)

      let i = sigmoid(pieces[0])
      let f = sigmoid(pieces[1])
      let g = tanh(pieces[2])
      let o = sigmoid(pieces[3])

      if cell != nil {
        cell = f * cell + i * g
      } else {
        cell = i * g
      }
      hidden = o * tanh(cell)

      allCell.append(cell)
      allHidden.append(hidden)
    }

    return (
      stacked(allHidden, axis: -2),
      stacked(allCell, axis: -2)
    )
  }
}

class BLSTM: Module, UnaryLayer {
  let forward: [LSTM]
  let backward: [LSTM]
  let linear: Linear

  init(inputSize: Int, numLayers: Int = 2) {
    var forwardLayers = [LSTM]()
    var backwardLayers = [LSTM]()
    for i in 0..<numLayers {
      if i == 0 {
        forwardLayers.append(LSTM(inputSize: inputSize, hiddenSize: inputSize))
        backwardLayers.append(LSTM(inputSize: inputSize, hiddenSize: inputSize))
      } else {
        forwardLayers.append(LSTM(inputSize: inputSize * 2, hiddenSize: inputSize))
        backwardLayers.append(LSTM(inputSize: inputSize * 2, hiddenSize: inputSize))
      }
    }
    self.forward = forwardLayers
    self.backward = backwardLayers
    self.linear = Linear(inputSize * 2, inputSize)
  }

  func reversed(_ x: MLXArray) -> MLXArray {
    return x[.ellipsis, stride(by: -1), 0...]
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for i in 0..<forward.count {
      let forwardOut = forward[i](hidden).0
      let backwardOut = reversed(backward[i](reversed(hidden)).0)
      hidden = MLX.concatenated([forwardOut, backwardOut], axis: -1)
    }
    //printTensorStats(hidden, name: "After self.lstm")
    let ret = linear(hidden)
    //printTensorStats(ret, name: "After self.linear")
    return ret
  }
}

class EncoderBlock: Module, UnaryLayer {
  let conv1: Conv1d
  let relu: ReLU
  let conv2: Conv1d
  let glu: GLU

  init(inChannels: Int, outChannels: Int) {
    self.conv1 = Conv1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
    self.relu = ReLU()
    self.conv2 = Conv1d(
      inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1, stride: 1)
    self.glu = GLU(axis: 2)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv1(x)
    x = relu(x)
    x = conv2(x)
    x = glu(x)
    return x
  }
}

class DecoderBlock: Module, UnaryLayer {
  let conv: Conv1d
  let glu: GLU
  let convTranspose: ConvTransposed1d
  let relu: ReLU?

  init(inChannels: Int, outChannels: Int, isLast: Bool = false) {
    self.conv = Conv1d(
      inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1)
    self.glu = GLU(axis: 2)
    self.convTranspose = ConvTransposed1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
    self.relu = isLast ? nil : ReLU()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv(x)
    x = glu(x)
    x = convTranspose(x)
    if let relu = relu {
      x = relu(x)
    }
    return x
  }
}

class DemucsModel: Module, UnaryLayer {
  let encoder: [EncoderBlock]
  let lstm: BLSTM
  let decoder: [DecoderBlock]

  override init() {
    self.encoder = [
      EncoderBlock(inChannels: 2, outChannels: 64),
      EncoderBlock(inChannels: 64, outChannels: 128),
      EncoderBlock(inChannels: 128, outChannels: 256),
      EncoderBlock(inChannels: 256, outChannels: 512),
      EncoderBlock(inChannels: 512, outChannels: 1024),
      EncoderBlock(inChannels: 1024, outChannels: 2048),
    ]
    self.lstm = BLSTM(inputSize: 2048)
    self.decoder = [
      DecoderBlock(inChannels: 2048, outChannels: 1024),
      DecoderBlock(inChannels: 1024, outChannels: 512),
      DecoderBlock(inChannels: 512, outChannels: 256),
      DecoderBlock(inChannels: 256, outChannels: 128),
      DecoderBlock(inChannels: 128, outChannels: 64),
      DecoderBlock(inChannels: 64, outChannels: 8, isLast: true),
    ]
  }

  public func validLength(_ length: Int, depth: Int = 6, kernelSize: Int = 8, context: Int = 3, stride: Int = 4) -> Int {
    var length = length

    // Forward pass through layers
    for _ in 0..<depth {
      length = Int(ceil(Double(length - kernelSize) / Double(stride))) + 1
      length = max(1, length)
      length += context - 1
    }

    // Backward pass through layers
    for _ in 0..<depth {
      length = (length - 1) * stride + kernelSize
    }

    return length
  }

  func centerTrim(_ tensor: MLXArray, reference: MLXArray) -> MLXArray {
    let referenceSize = reference.dim(-2)
    let delta = tensor.dim(-2) - referenceSize
    if delta < 0 {
      fatalError("tensor must be larger than reference. Delta is \(delta).")
    }
    if delta == 0 {
      return tensor
    }
    let startIdx = delta / 2
    let endIdx = tensor.dim(-2) - (delta - delta / 2)
    return tensor[.ellipsis, startIdx..<endIdx, 0...]
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x
    var saved = [MLXArray]()

    //print("\n=== Forward Pass Statistics ===")
    //printTensorStats(x, name: "Input")

    for (i, encode) in encoder.enumerated() {
      x = encode(x)
      //printTensorStats(x, name: "After encoder[\(i)]")
      saved.append(x)
    }

    x = lstm(x)
    //printTensorStats(x, name: "After LSTM")

    for (i, decode) in decoder.enumerated() {
      let skip = centerTrim(saved.removeLast(), reference: x)
      x = x + skip
      //printTensorStats(x, name: "After skip connection decoder[\(i)]")
      x = decode(x)
      //printTensorStats(x, name: "After decoder[\(i)]")
    }

    var shape = x.shape
    shape.removeLast()
    shape += [4, 2]
    x = x.reshaped(shape)
    print()

    return x
  }
}
