import MLX
import MLXNN
import MLXRandom
import Foundation

struct GLU {
  let dim: Int

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (a, b) = x.split(axis: dim, stream: .default)
    return a * MLXNN.sigmoid(b)
  }
}

struct BLSTM {
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

struct EncoderBlock {
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

struct DecoderBlock {
  let conv1: Conv1d
  let glu: GLU
  let convTranspose: Conv1d
  let reluLayer: ReLU?

  init(inChannels: Int, outChannels: Int, isLast: Bool = false) {
    self.conv1 = Conv1d(
      inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1)
    self.glu = GLU(dim: 1)
    self.convTranspose = Conv1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
    self.reluLayer = isLast ? nil : ReLU()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv1(x)
    x = glu(x)
    x = convTranspose(x)
    if let reluLayer = reluLayer {
      x = reluLayer(x)
    }
    return x
  }
}

struct DemucsModel {
  let encoder: [EncoderBlock]
  let decoder: [DecoderBlock]
  let lstm: BLSTM

  init() {
    // Encoder blocks
    self.encoder = [
      EncoderBlock(inChannels: 2, outChannels: 64),
      EncoderBlock(inChannels: 64, outChannels: 128),
      EncoderBlock(inChannels: 128, outChannels: 256),
      EncoderBlock(inChannels: 256, outChannels: 512),
      EncoderBlock(inChannels: 512, outChannels: 1024),
      EncoderBlock(inChannels: 1024, outChannels: 2048),
    ]

    // Decoder blocks
    self.decoder = [
      DecoderBlock(inChannels: 2048, outChannels: 1024),
      DecoderBlock(inChannels: 1024, outChannels: 512),
      DecoderBlock(inChannels: 512, outChannels: 256),
      DecoderBlock(inChannels: 256, outChannels: 128),
      DecoderBlock(inChannels: 128, outChannels: 64),
      DecoderBlock(inChannels: 64, outChannels: 8, isLast: true),
    ]

    // LSTM layer
    self.lstm = BLSTM(inputSize: 2048, hiddenSize: 2048)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var skips: [MLXArray] = []

    // Encoder path
    var encoded = x
    for block in encoder {
      encoded = block(encoded)
      skips.append(encoded)
    }

    // LSTM
    encoded = lstm(encoded)

    // Decoder path
    var decoded = encoded
    for (i, block) in decoder.enumerated() {
      decoded = block(decoded + skips[skips.count - 1 - i])
    }

    return decoded
  }
}
