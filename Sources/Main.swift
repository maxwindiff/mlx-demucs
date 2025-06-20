import ArgumentParser
import MLX
import MLXNN
import Foundation

@main
struct Main: ParsableCommand {
  @Option
  var modelPath = "/tmp/mlx/demucs.safetensors"

  func loadModel(from path: String) throws -> DemucsModel {
    let weights = try loadArrays(url: URL(fileURLWithPath: path))
//    print("== weights ==")
//    for key in weights.keys.sorted() {
//      print("\(key): \(weights[key]!.shape)")
//    }

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

    let model = DemucsModel()
//    print("== model ==")
//    for (key, value) in model.parameters().flattened() {
//      print("\(key): \(value.shape)")
//    }

    let parameters = ModuleParameters.unflattened(transformed)
    try model.update(parameters: parameters, verify: [.all])
    return model
  }

  mutating func run() {
    do {
      let model = try loadModel(from: modelPath)
      let input = MLXArray.zeros([1, 894292, 2])
      print("Input shape: \(input.shape)")
      let output = model(input)
      print("Output shape: \(output.shape)")
    } catch {
      print("Error running model: \(error)")
    }
  }
}
