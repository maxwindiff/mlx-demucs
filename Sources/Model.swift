import Foundation
import MLX
import MLXNN

class GLU: Module, UnaryLayer {
  let dim: Int

  init(dim: Int) {
    self.dim = dim
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (a, b) = x.split(axis: dim)
    return a * MLXNN.sigmoid(b)
  }
}

class BLSTM: Module, UnaryLayer {
  let lstm: LSTM
  let linear: Linear

  init(inputSize: Int, hiddenSize: Int, numLayers: Int = 2) {
    self.lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
    self.linear = Linear(hiddenSize * 2, hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (lstmOut, _) = lstm(x)
    return linear(lstmOut)
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
    self.glu = GLU(dim: 1)
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
  let convTranspose: Conv1d
  let relu: ReLU?

  init(inChannels: Int, outChannels: Int, isLast: Bool = false) {
    self.conv = Conv1d(
      inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1)
    self.glu = GLU(dim: 1)
    self.convTranspose = Conv1d(
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
  // let lstm: BLSTM
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
    // self.lstm = BLSTM(inputSize: 2048, hiddenSize: 2048)
    self.decoder = [
      DecoderBlock(inChannels: 2048, outChannels: 1024),
      DecoderBlock(inChannels: 1024, outChannels: 512),
      DecoderBlock(inChannels: 512, outChannels: 256),
      DecoderBlock(inChannels: 256, outChannels: 128),
      DecoderBlock(inChannels: 128, outChannels: 64),
      DecoderBlock(inChannels: 64, outChannels: 8, isLast: true),
    ]
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return x
  }
}
