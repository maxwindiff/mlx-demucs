import ArgumentParser
import MLX
import MLXNN
import Foundation

@main
struct Main: ParsableCommand {
  @Option
  var modelPath = "/tmp/mlx/demucs.safetensors"

  mutating func run() {
    do {
      let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
      print("== weights ==")
      for key in weights.keys.sorted() {
        print("\(key): \(weights[key]!.shape)")
      }

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
        }
        transformed[key] = value
      }

      let model = DemucsModel()
      let shape = model.parameters().flattened()
      print("== model ==")
      for (key, value) in shape {
        print("\(key): \(value.shape)")
      }

      let parameters = ModuleParameters.unflattened(transformed)
      try model.update(parameters: parameters, verify: [.all])
    } catch {
      print("Error loading model: \(error)")
    }
  }
}
