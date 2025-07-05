import Foundation
import MLX
import MLXNN

class ScaledEmbedding: Module, UnaryLayer {
  let embedding: Embedding

  init(numEmbeddings: Int, embeddingDim: Int) {
    self.embedding = Embedding(embeddingCount: numEmbeddings, dimensions: embeddingDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    return embedding(x) * sqrt(Float(embedding.weight.shape[1]))
  }
}

class LayerScale: Module, UnaryLayer {
  let scale: MLXArray

  init(dim: Int, initValue: Float = 1e-5) {
    self.scale = MLXArray.full([dim], values: MLXArray(initValue))
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    return x * scale
  }
}

class LocalState: Module, UnaryLayer {
  let content: Conv1d
  let query: Conv1d
  let key: Conv1d
  let queryDecay: Conv1d
  let proj: Conv1d

  init(channels: Int) {
    self.content = Conv1d(
      inputChannels: channels, outputChannels: channels, kernelSize: 1, stride: 1)
    self.query = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1, stride: 1)
    self.key = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1, stride: 1)
    self.queryDecay = Conv1d(inputChannels: channels, outputChannels: 16, kernelSize: 1, stride: 1)
    self.proj = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1, stride: 1)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // TODO: Implement LocalState attention mechanism
    // This is a simplified placeholder - the actual implementation would involve
    // content-based attention with query decay mechanism
    return x
  }
}

class DConv: Module, UnaryLayer {
  let layers: [Sequential]

  init(channels: Int, depth: Int = 2, useLSTM: Bool = false) {
    var layerList = [Sequential]()
    for i in 0..<depth {
      let bottleneck = channels / 4
      let dilation = depth > 0 ? (1 << i) : 1
      let paddding = dilation // assume kernel size is 3, so (kernel // 2) == 1

      var layerComponents: [any UnaryLayer] = [
        Conv1d(
          inputChannels: channels, outputChannels: bottleneck, kernelSize: 3, stride: 1,
          padding: paddding, dilation: dilation),
        GroupNorm(groupCount: 1, dimensions: bottleneck, eps: 1e-5),
        GELU(),
      ]

      if useLSTM {
        layerComponents.append(BLSTM(inputSize: bottleneck))
        layerComponents.append(LocalState(channels: bottleneck))
      }

      layerComponents.append(contentsOf: [
        Conv1d(inputChannels: bottleneck, outputChannels: channels * 2, kernelSize: 1, stride: 1),
        GroupNorm(groupCount: 1, dimensions: channels * 2, eps: 1e-5),
        GLU(axis: 1),
        LayerScale(dim: channels),
      ])

      let layer = Sequential(layers: layerComponents)
      layerList.append(layer)
    }
    self.layers = layerList
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var output = x
    for layer in layers {
      output = layer(output)
    }
    return output
  }
}

class HEncLayer: Module, UnaryLayer {
  let conv: any UnaryLayer
  let norm1: any UnaryLayer
  let rewrite: any UnaryLayer
  let norm2: any UnaryLayer
  let dconv: DConv

  init(
    inChannels: Int,
    outChannels: Int,
    useNorm: Bool = false,
    useLSTM: Bool = false,
    pad: Bool = true,
    last: Bool = false,
  ) {
    let kernelSize = last ? 4 : 8
    let stride = last ? 2 : 4
    let padding = pad ? 2 : 0

    if last {
      self.conv = Conv1d(
        inputChannels: inChannels,
        outputChannels: outChannels,
        kernelSize: kernelSize,
        stride: stride,
        padding: padding)
      self.rewrite = Conv1d(
        inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1, stride: 1)
    } else {
      self.conv = Conv2d(
        inputChannels: inChannels,
        outputChannels: outChannels,
        kernelSize: [kernelSize, 1],
        stride: [stride, 1],
        padding: [padding, 0])
      self.rewrite = Conv2d(
        inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1, stride: 1)
    }

    self.norm1 = useNorm ? GroupNorm(groupCount: 4, dimensions: outChannels, eps: 1e-5) : Identity()
    self.norm2 =
      useNorm ? GroupNorm(groupCount: 4, dimensions: outChannels * 2, eps: 1e-5) : Identity()
    self.dconv = DConv(channels: outChannels, useLSTM: useLSTM)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv(x)
    x = norm1(x)
    x = rewrite(x)
    x = norm2(x)
    x = dconv(x)
    return x
  }
}

class HEncLayer1D: Module, UnaryLayer {
  let conv: any UnaryLayer
  let norm1: any UnaryLayer
  let rewrite: any UnaryLayer
  let norm2: any UnaryLayer
  let dconv: any UnaryLayer

  init(
    inChannels: Int,
    outChannels: Int,
    useNorm: Bool = false,
    useLSTM: Bool = false,
    last: Bool = false,
  ) {
    self.conv = Conv1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4, padding: 2)
    if last {
      self.norm1 = Identity()
      self.rewrite = Identity()
      self.norm2 = Identity()
      self.dconv = Identity()
    } else {
      self.norm1 =
        useNorm ? GroupNorm(groupCount: 4, dimensions: outChannels, eps: 1e-5) : Identity()
      self.rewrite = Conv1d(
        inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1, stride: 1)
      self.norm2 =
        useNorm ? GroupNorm(groupCount: 4, dimensions: outChannels * 2, eps: 1e-5) : Identity()
      self.dconv = DConv(channels: outChannels, useLSTM: useLSTM)
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = conv(x)
    x = norm1(x)
    x = rewrite(x)
    x = norm2(x)
    x = dconv(x)
    return x
  }
}

class HDecLayer: Module, UnaryLayer {
  let convTr: UnaryLayer
  let norm2: any UnaryLayer
  let rewrite: UnaryLayer
  let norm1: any UnaryLayer

  init(inChannels: Int, outChannels: Int, useNorm: Bool = false, last: Bool = false) {
    if last {
      self.convTr = ConvTransposed1d(
        inputChannels: inChannels, outputChannels: outChannels, kernelSize: 4, stride: 2)
      self.rewrite = Conv1d(
        inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1,
        padding: 1)
    } else {
      self.convTr = ConvTransposed2d(
        inputChannels: inChannels, outputChannels: outChannels, kernelSize: [8, 1], stride: [4, 1])
      self.rewrite = Conv2d(
        inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1,
        padding: 1)
    }
    self.norm2 = useNorm ? GroupNorm(groupCount: 4, dimensions: outChannels, eps: 1e-5) : Identity()
    self.norm1 =
      useNorm ? GroupNorm(groupCount: 4, dimensions: inChannels * 2, eps: 1e-5) : Identity()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = convTr(x)
    x = norm2(x)
    x = rewrite(x)
    x = norm1(x)
    return x
  }
}

class HDecLayer1D: Module, UnaryLayer {
  let convTr: ConvTransposed1d
  let norm2: any UnaryLayer
  let rewrite: any UnaryLayer

  init(inChannels: Int, outChannels: Int, last: Bool = false) {
    self.convTr = ConvTransposed1d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 2)
    if last {
      self.norm2 = GroupNorm(groupCount: 4, dimensions: outChannels, eps: 1e-5)
      self.rewrite = Identity()
    } else {
      self.norm2 = Identity()
      self.rewrite = Conv1d(
        inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: 3, stride: 1,
        padding: 1)
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = convTr(x)
    x = norm2(x)
    x = rewrite(x)
    return x
  }
}

class HDemucs: Module, UnaryLayer {
  let encoder: [any UnaryLayer]
  let decoder: [any UnaryLayer]
  let tencoder: [any UnaryLayer]
  let tdecoder: [HDecLayer1D]
  let freqEmb: ScaledEmbedding

  override init() {
    self.encoder = [
      HEncLayer(inChannels: 4, outChannels: 48),
      HEncLayer(inChannels: 48, outChannels: 96),
      HEncLayer(inChannels: 96, outChannels: 192),
      HEncLayer(inChannels: 192, outChannels: 384),
      HEncLayer(inChannels: 384, outChannels: 768, useNorm: true, useLSTM: true, pad: false),
      HEncLayer(inChannels: 768, outChannels: 1536, useNorm: true, useLSTM: true, last: true)
    ]

    self.decoder = [
      HDecLayer(inChannels: 1536, outChannels: 768, useNorm: true, last: true),
      HDecLayer(inChannels: 768, outChannels: 384, useNorm: true),
      HDecLayer(inChannels: 384, outChannels: 192),
      HDecLayer(inChannels: 192, outChannels: 96),
      HDecLayer(inChannels: 96, outChannels: 48),
      HDecLayer(inChannels: 48, outChannels: 16),
    ]

    self.tencoder = [
      HEncLayer1D(inChannels: 2, outChannels: 48),
      HEncLayer1D(inChannels: 48, outChannels: 96),
      HEncLayer1D(inChannels: 96, outChannels: 192),
      HEncLayer1D(inChannels: 192, outChannels: 384),
      HEncLayer1D(inChannels: 384, outChannels: 768, last: true)
    ]

    self.tdecoder = [
      HDecLayer1D(inChannels: 768, outChannels: 384, last: true),
      HDecLayer1D(inChannels: 384, outChannels: 192),
      HDecLayer1D(inChannels: 192, outChannels: 96),
      HDecLayer1D(inChannels: 96, outChannels: 48),
      HDecLayer1D(inChannels: 48, outChannels: 8),
    ]

    self.freqEmb = ScaledEmbedding(numEmbeddings: 512, embeddingDim: 48)
  }

  static func transformPytorch(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var transformed = [String: MLXArray]()

    for var (key, value) in weights {
      // Remove "models.0." prefix
      if key.hasPrefix("models.0.") {
        key = String(key.dropFirst("models.0.".count))
      }

      // Handle encoder layers
      if let match = key.wholeMatch(of: /encoder\.(\d+)\.conv\.weight/) {
        key = "encoder.\(match.1).conv.weight"
        if value.ndim == 4 {
          value = value.transposed(0, 2, 3, 1)
        } else if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.rewrite\.weight/) {
        key = "encoder.\(match.1).rewrite.weight"
        if value.ndim == 4 {
          value = value.transposed(0, 2, 3, 1)
        } else if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).weight"
        if value.ndim >= 2 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).bias"
      } else if let match = key.wholeMatch(of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.scale/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).scale"
      }

      // Handle decoder layers
      else if let match = key.wholeMatch(of: /decoder\.(\d+)\.conv_tr\.weight/) {
        key = "decoder.\(match.1).convTr.weight"
        if value.ndim == 4 {
          value = value.transposed(1, 2, 3, 0)
        } else {
          value = value.transposed(1, 2, 0)
        }
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.conv_tr\.bias/) {
        key = "decoder.\(match.1).convTr.bias"
      } else if let match = key.wholeMatch(of: /decoder\.(\d+)\.rewrite\.weight/) {
        key = "decoder.\(match.1).rewrite.weight"
        if value.ndim == 4 {
          value = value.transposed(0, 2, 3, 1)
        } else if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      }

      // Handle temporal encoder layers
      else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.conv\.weight/) {
        key = "tencoder.\(match.1).conv.weight"
        if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.rewrite\.weight/) {
        key = "tencoder.\(match.1).rewrite.weight"
        if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.weight/) {
        key = "tencoder.\(match.1).weight"
        if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.bias/) {
        key = "tencoder.\(match.1).bias"
      }

      // Handle temporal decoder layers
      else if let match = key.wholeMatch(of: /tdecoder\.(\d+)\.conv_tr\.weight/) {
        key = "tdecoder.\(match.1).convTr.weight"
        if value.ndim == 3 {
          value = value.transposed(1, 2, 0)
        }
      } else if let match = key.wholeMatch(of: /tdecoder\.(\d+)\.conv_tr\.bias/) {
        key = "tdecoder.\(match.1).convTr.bias"
      } else if let match = key.wholeMatch(of: /tdecoder\.(\d+)\.rewrite\.weight/) {
        key = "tdecoder.\(match.1).rewrite.weight"
        if value.ndim == 3 {
          value = value.transposed(0, 2, 1)
        }
      }

      // Handle frequency embedding
      else if key == "freq_emb.embedding.weight" {
        key = "freqEmb.embedding.weight"
      }

      // Handle LSTM layers
      else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.lstm\.weight_ih_l(\d+)(_reverse)?/)
      {
        let direction = match.5 != nil ? "backward" : "forward"
        let layer = match.4
        key =
          "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).\(direction).\(layer).weight_ih"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.lstm\.weight_hh_l(\d+)(_reverse)?/)
      {
        let direction = match.5 != nil ? "backward" : "forward"
        let layer = match.4
        key =
          "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).\(direction).\(layer).weight_hh"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.lstm\.bias_ih_l(\d+)(_reverse)?/)
      {
        let direction = match.5 != nil ? "backward" : "forward"
        let layer = match.4
        key =
          "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).\(direction).\(layer).bias_ih"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.lstm\.bias_hh_l(\d+)(_reverse)?/)
      {
        let direction = match.5 != nil ? "backward" : "forward"
        let layer = match.4
        key =
          "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).\(direction).\(layer).bias_hh"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.linear\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).linear.weight"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.linear\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).linear.bias"
      }

      // Handle LocalState layers
      else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.content\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).content.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.content\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).content.bias"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.query\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).query.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.query\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).query.bias"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.key\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).key.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.key\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).key.bias"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.query_decay\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).queryDecay.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.query_decay\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).queryDecay.bias"
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.proj\.weight/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).proj.weight"
        value = value.transposed(0, 2, 1)
      } else if let match = key.wholeMatch(
        of: /encoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.proj\.bias/)
      {
        key = "encoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).proj.bias"
      }

      // Handle temporal encoder DConv layers
      else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.weight/)
      {
        key = "tencoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).weight"
        if value.ndim >= 2 {
          value = value.transposed(0, 2, 1)
        }
      } else if let match = key.wholeMatch(of: /tencoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.bias/)
      {
        key = "tencoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).bias"
      } else if let match = key.wholeMatch(
        of: /tencoder\.(\d+)\.dconv\.layers\.(\d+)\.(\d+)\.scale/)
      {
        key = "tencoder.\(match.1).dconv.layers.\(match.2).layers.\(match.3).scale"
      }

      transformed[key] = value
    }
    return transformed
  }

  func idealLength(
    _ length: Int, depth: Int = 6, kernelSize: Int = 8, context: Int = 3, stride: Int = 4
  ) -> Int {
    var length = length

    for _ in 0..<depth {
      length = Int(ceil(Double(length - kernelSize) / Double(stride))) + 1
      length = max(1, length)
      length += context - 1
    }

    for _ in 0..<depth {
      length = (length - 1) * stride + kernelSize
    }

    return length
  }

  func padInput(_ input: MLXArray) -> MLXArray {
    let currentLength = input.shape[1]
    let totalPadding = idealLength(currentLength) - currentLength
    let leftPad = totalPadding / 2
    let rightPad = totalPadding - leftPad
    return MLX.concatenated(
      [MLXArray.zeros([2, leftPad]), input, MLXArray.zeros([2, rightPad])], axis: 1)
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
    // TODO: Implement full HDemucs forward pass
    // This is a skeleton implementation - the actual forward pass would involve:
    // 1. Processing through encoder layers with skip connections
    // 2. Processing through decoder layers with skip connections
    // 3. Processing through tencoder and tdecoder for temporal information
    // 4. Combining frequency and temporal embeddings
    // 5. Final output processing

    var x = x
    var saved = [MLXArray]()

    // Encoder pass
    for encode in encoder {
      x = encode(x)
      saved.append(x)
    }

    // Decoder pass with skip connections
    for decode in decoder {
      let skip = centerTrim(saved.removeLast(), reference: x)
      x = x + skip
      x = decode(x)
    }

    // TODO: Add temporal encoder/decoder processing
    // TODO: Add frequency embedding integration
    // TODO: Add final output processing

    return x
  }
}
