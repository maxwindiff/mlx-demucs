import ArgumentParser
import MLX
import MLXNN
import Foundation
import AVFoundation

@main
struct Main: ParsableCommand {
  @Option
  var modelPath = "/tmp/mlx/demucs.safetensors"

  @Option
  var inputPath: String = "input.wav"

  @Option
  var outputDir: String = "output"

  func loadWavFile(from path: String) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: audioFile.fileFormat.sampleRate,
                                     channels: 2,
                                     interleaved: false) else {
      throw NSError(domain: "AudioFormat", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
    }

    let frameCount = AVAudioFrameCount(audioFile.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw NSError(domain: "AudioBuffer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    print("Sample rate: \(audioFile.fileFormat.sampleRate) Hz")
    print("Frame count: \(frameCount)")

    try audioFile.read(into: buffer)
    guard let leftChannelData = buffer.floatChannelData?[0],
          let rightChannelData = buffer.floatChannelData?[1] else {
      throw NSError(domain: "AudioData", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get channel data"])
    }
    let sampleCount = Int(buffer.frameLength)
    var audioData = [Float]()
    audioData.reserveCapacity(sampleCount * 2)

    for i in 0..<sampleCount {
      audioData.append(leftChannelData[i])
    }
    for i in 0..<sampleCount {
      audioData.append(rightChannelData[i])
    }

    let mlxArray = MLXArray(audioData, [2, sampleCount])
    return mlxArray
  }

  func saveWavFile(data: MLXArray, path: String, sampleRate: Double = 44100.0) throws {
    let shape = data.shape
    guard shape.count == 2 else {
      throw NSError(domain: "AudioSave", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected 2D array [samples, channels]"])
    }

    let numSamples = shape[0]
    let numChannels = shape[1]

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: sampleRate,
                                     channels: AVAudioChannelCount(numChannels),
                                     interleaved: false) else {
      throw NSError(domain: "AudioFormat", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
    }

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(numSamples)) else {
      throw NSError(domain: "AudioBuffer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    buffer.frameLength = AVAudioFrameCount(numSamples)

    let audioData = data.asArray(Float.self)

    for channel in 0..<numChannels {
      guard let channelData = buffer.floatChannelData?[channel] else {
        throw NSError(domain: "AudioData", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get channel data"])
      }

      for sample in 0..<numSamples {
        channelData[sample] = audioData[sample * numChannels + channel]
      }
    }

    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
    try audioFile.write(from: buffer)
  }

  func loadModel(from path: String) throws -> DemucsModel {
    let weights = try loadArrays(url: URL(fileURLWithPath: path))

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

    let parameters = ModuleParameters.unflattened(transformed)
    try model.update(parameters: parameters, verify: [.all])
    return model
  }

  func printModelStats(_ model: DemucsModel) {
    print("\n=== Model Layer Statistics ===")

    let parameters = model.parameters().flattened()

    for (name, param) in parameters {
      let shape = param.shape
      let mean = MLX.mean(param)
      let variance = MLX.variance(param)

      let flattenedParam = param.flattened()
      let numElements = min(5, flattenedParam.size)
      let firstElements = Array(0..<numElements).map { flattenedParam[MLXArray($0)].item(Float.self) }

      print("Layer: \(name)")
      print("  Shape: (\(shape.map(String.init).joined(separator: ", ")))")
      print(String(format: "  Mean: %.6f, Variance: %.6f", mean.item(Float.self), variance.item(Float.self)))
      print("  First elements: [\(firstElements.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")
      print("--------------------------------------------------")
    }
    print()
  }

  func playAudio(from path: String, duration: TimeInterval = 10.0) throws {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)

    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()

    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: audioFile.processingFormat)

    try engine.start()

    let sampleRate = audioFile.processingFormat.sampleRate
    let framesToPlay = AVAudioFrameCount(duration * sampleRate)
    let actualFramesToPlay = min(framesToPlay, AVAudioFrameCount(audioFile.length))

    guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: actualFramesToPlay) else {
      throw NSError(domain: "AudioBuffer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
    }

    try audioFile.read(into: buffer, frameCount: actualFramesToPlay)

    print("Playing first \(duration) seconds of vocals (left channel)...")

    playerNode.scheduleBuffer(buffer, at: nil, options: []) {
      print("Playback completed.")
    }

    playerNode.play()

    let playbackDuration = Double(actualFramesToPlay) / sampleRate
    Thread.sleep(forTimeInterval: playbackDuration + 0.5)

    engine.stop()
  }

  mutating func run() {
    do {
      let model = try loadModel(from: modelPath)

      let input = try loadWavFile(from: inputPath)
      print("Input shape: \(input.shape)")

      // Normalize input
      let ref = input.mean(axis: 0)
      let refMean = ref.mean()
      let refStd = MLX.sqrt(ref.variance())
      print("ref.shape = \(ref.shape) refMean = \(refMean), refStd = \(refStd)")
      let normalizedInput = (input - refMean) / refStd

      // Padd input
      let currentLength = normalizedInput.shape[1]
      let totalPadding = model.validLength(currentLength) - currentLength
      let leftPad = totalPadding / 2
      let rightPad = totalPadding - leftPad
      let paddedInput = MLX.concatenated([MLXArray.zeros([2, leftPad]), normalizedInput, MLXArray.zeros([2, rightPad])], axis: 1)

      let permutedInput = paddedInput.expandedDimensions(axis: 0).transposed(axes: [0, 2, 1])

      var output = model(permutedInput)
      // Un-normalize output
      output = output * refStd + refMean
      print("Output shape: \(output.shape)")

      let outputShape = output.shape
      guard outputShape.count == 4 else {
        throw NSError(domain: "OutputShape", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected output shape [1, samples, instruments, channels]"])
      }

      let numInstruments = outputShape[2]
      let instrumentNames = ["drums", "bass", "other", "vocals"]

      try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
      for instrument in 3..<numInstruments {  // only save vocals for now
        let instrumentData = output[0, 0..., instrument, 0...]
        let instrumentName = instrument < instrumentNames.count ? instrumentNames[instrument] : "instrument_\(instrument)"

        let leftChannelData = instrumentData[0..., 0]
        let rightChannelData = instrumentData[0..., 1]

        let leftOutputPath = "\(outputDir)/\(instrumentName)_left.wav"
        let rightOutputPath = "\(outputDir)/\(instrumentName)_right.wav"

        try saveWavFile(data: leftChannelData.expandedDimensions(axis: 1), path: leftOutputPath)
        try saveWavFile(data: rightChannelData.expandedDimensions(axis: 1), path: rightOutputPath)

        print("Saved \(instrumentName) left channel to \(leftOutputPath)")
        print("Saved \(instrumentName) right channel to \(rightOutputPath)")

        if instrumentName == "vocals" {
          try playAudio(from: leftOutputPath, duration: 3)
        }
      }
    } catch {
      print("Error running model: \(error)")
    }
  }
}
