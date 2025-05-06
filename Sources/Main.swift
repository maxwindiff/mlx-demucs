import ArgumentParser
import MLX
import Foundation

@main
struct Main: ParsableCommand {
  @Option
  var modelPath = "/tmp/mlx/demucs.safetensors"

  mutating func run() {
    do {
      let arrays = try loadArrays(url: URL(fileURLWithPath: modelPath))
      for key in arrays.keys.sorted() {
        print("- \(key)")
      }
    } catch {
      print("Error loading model: \(error)")
    }
  }
}
