import Foundation
import MLX
import MLXNN

// Copied from https://github.com/ml-explore/mlx-swift/blob/0.25.4/Source/MLXNN/Recurrent.swift#L181
// Added bias_hh and renamed weights to make importing from pytorch easier.
class LSTM: Module {
  public var weight_ih: MLXArray
  public var weight_hh: MLXArray
  public let bias_ih: MLXArray
  public let bias_hh: MLXArray

  public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
    self.weight_ih = MLXArray.zeros([4 * hiddenSize, inputSize])
    self.weight_hh = MLXArray.zeros([4 * hiddenSize, hiddenSize])
    self.bias_ih = MLXArray.zeros([4 * hiddenSize])
    self.bias_hh = MLXArray.zeros([4 * hiddenSize])
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
  let linear: Linear?

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
    self.linear = numLayers > 1 ? Linear(inputSize * 2, inputSize) : nil
  }

  func reversed(_ x: MLXArray) -> MLXArray {
    return x[.ellipsis, stride(by: -1), 0...]
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x
    for i in 0..<forward.count {
      let forwardOut = forward[i](x).0
      let backwardOut = reversed(backward[i](reversed(x)).0)
      x = MLX.concatenated([forwardOut, backwardOut], axis: -1)
    }
    if let linear = linear {
      x = linear(x)
    }
    return x
  }
}

class EncoderBlock: Module, UnaryLayer {
  let conv1: Conv1d
  let relu: ReLU
  let conv2: Conv1d
  let glu: GLU

  init(inChannels: Int, outChannels: Int) {
    self.conv1 = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
    self.relu = ReLU()
    self.conv2 = Conv1d(inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1, stride: 1)
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
    self.conv = Conv1d(inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1)
    self.glu = GLU(axis: 2)
    self.convTranspose = ConvTransposed1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
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

class Demucs: Module, UnaryLayer {
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

  static func transformPytorch(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var transformed = [String: MLXArray]()
    for var (key, value) in weights {
      if let match = key.wholeMatch(of: /encoder\.(\d+)\.0\.weight/) {
        key = "encoder.\(match.1).conv1.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.2\.weight/) {
        key = "encoder.\(match.1).conv2.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.0\.weight/) {
        key = "decoder.\(match.1).conv.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.2\.weight/) {
        key = "decoder.\(match.1).convTranspose.weight"
        value = value.transposed(1, 2, 0)
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.0\.bias/) {
        key = "encoder.\(match.1).conv1.bias"
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.2\.bias/) {
        key = "encoder.\(match.1).conv2.bias"
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.0\.bias/) {
        key = "decoder.\(match.1).conv.bias"
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.2\.bias/) {
        key = "decoder.\(match.1).convTranspose.bias"
      } else if key.hasPrefix("lstm.lstm.") {
        let suffix = String(key.dropFirst("lstm.lstm.".count))
        var components = suffix.components(separatedBy: "_")
        let isReverse = components.last == "reverse"
        if isReverse {
          components.removeLast()
        }
        guard let layerComponent = components.last,
              layerComponent.hasPrefix("l"),
              let layerNumber = Int(String(layerComponent.dropFirst())) else {
          continue
        }
        components.removeLast()
        let paramType = components.joined(separator: "_")
        let direction = isReverse ? "backward" : "forward"
        key = "lstm.\(direction).\(layerNumber).\(paramType)"
      }
      transformed[key] = value
    }
    return transformed
  }

  func idealLength(_ length: Int, depth: Int = 6, kernelSize: Int = 8, context: Int = 3, stride: Int = 4) -> Int {
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

  // Pad the length so that no truncation happens during model execution.
  func padInput(_ input: MLXArray) -> MLXArray {
    let currentLength = input.shape[1] // samples dimension
    let totalPadding = idealLength(currentLength) - currentLength
    let leftPad = totalPadding / 2
    let rightPad = totalPadding - leftPad
    let batchSize = input.shape[0]
    let channels = input.shape[2]
    return MLX.concatenated([MLXArray.zeros([batchSize, leftPad, channels]), input, MLXArray.zeros([batchSize, rightPad, channels])], axis: 1)
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
    // Apply normalization - x has shape [batch, samples, channels]
    let ref = x.mean(axes: [0, 1]) // mean across batch and samples, keeping channels
    let refMean = ref.mean()
    let refStd = MLX.sqrt(x.variance())
    let normalizedInput = (x - refMean) / refStd
    
    // Pad input
    let paddedInput = padInput(normalizedInput)
    
    var x = paddedInput
    var saved = [MLXArray]()

    for encode in encoder {
      x = encode(x)
      saved.append(x)
    }
    x = lstm(x)
    for decode in decoder {
      let skip = centerTrim(saved.removeLast(), reference: x)
      x = x + skip
      x = decode(x)
    }

    // Reshape to 4 sources x 2 channels
    x = x.reshaped(x.shape.dropLast() + [4, 2])
    
    // Apply denormalization
    x = x * refStd + refMean
    
    return x
  }
}
