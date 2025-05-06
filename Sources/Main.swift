import ArgumentParser
import MLX
import Foundation

@main
struct Demucs: ParsableCommand {
  @Option
  var model = "demucs_quantized"

  mutating func run() {
    let fileManager = FileManager.default
    let homeDir = fileManager.homeDirectoryForCurrentUser
    let checkpointsDir = homeDir.appendingPathComponent(".cache/torch/hub/checkpoints")
    print("Searching for model files in \(checkpointsDir.path)")

    do {
      let directoryContents = try fileManager.contentsOfDirectory(atPath: checkpointsDir.path)
      for file in directoryContents {
        if file.hasPrefix("\(model)-") {
          print(file)
        }
      }
    } catch {
      print("Error searching for model files: \(error.localizedDescription)")
    }
  }
}
