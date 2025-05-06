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
  @ModuleInfo var lstm: LSTM
  @ModuleInfo var linear: Linear

  init(inputSize: Int, hiddenSize: Int, numLayers: Int = 2) {
    super.init()
    self.lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
    self.linear = Linear(hiddenSize * 2, hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (lstmOut, _) = lstm(x)
    return linear(lstmOut)
  }
}

class EncoderBlock: Module, UnaryLayer {
  @ModuleInfo(key: "0") var conv1: Conv1d
  @ModuleInfo(key: "1") var relu: ReLU
  @ModuleInfo(key: "2") var conv2: Conv1d
  @ModuleInfo(key: "3") var glu: GLU

  init(inChannels: Int, outChannels: Int) {
    super.init()
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
  @ModuleInfo(key: "0") var conv1: Conv1d
  @ModuleInfo(key: "1") var glu: GLU
  @ModuleInfo(key: "2") var convTranspose: Conv1d
  @ModuleInfo(key: "3") var relu: ReLU?

  init(inChannels: Int, outChannels: Int, isLast: Bool = false) {
    super.init()
    self.conv1 = Conv1d(
      inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1)
    self.glu = GLU(dim: 1)
    self.convTranspose = Conv1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
    self.relu = isLast ? nil : ReLU()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv1(x)
    x = glu(x)
    x = convTranspose(x)
    if let relu = relu {
      x = relu(x)
    }
    return x
  }
}

class DemucsModel: Module, UnaryLayer {
  @ModuleInfo var encoder: Sequential
  @ModuleInfo var lstm: BLSTM
  @ModuleInfo var decoder: Sequential

  override init() {
    super.init()
    self.encoder = Sequential {
      EncoderBlock(inChannels: 2, outChannels: 64)
      EncoderBlock(inChannels: 64, outChannels: 128)
      EncoderBlock(inChannels: 128, outChannels: 256)
      EncoderBlock(inChannels: 256, outChannels: 512)
      EncoderBlock(inChannels: 512, outChannels: 1024)
      EncoderBlock(inChannels: 1024, outChannels: 2048)
    }
    self.lstm = BLSTM(inputSize: 2048, hiddenSize: 2048)
    self.decoder = Sequential {
      DecoderBlock(inChannels: 2048, outChannels: 1024)
      DecoderBlock(inChannels: 1024, outChannels: 512)
      DecoderBlock(inChannels: 512, outChannels: 256)
      DecoderBlock(inChannels: 256, outChannels: 128)
      DecoderBlock(inChannels: 128, outChannels: 64)
      DecoderBlock(inChannels: 64, outChannels: 8, isLast: true)
    }
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let encoded = encoder(x)
    let lstmOut = lstm(encoded)
    let decoded = decoder(lstmOut)
    return decoded
  }
}
