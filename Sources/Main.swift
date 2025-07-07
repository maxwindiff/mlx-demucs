import ArgumentParser
import MLX
import MLXNN
import Foundation
import AVFoundation

enum DemucsError: Error {
  case error(String)
}

enum ModelType: String, CaseIterable, ExpressibleByArgument {
  case demucs = "demucs"
  case hdemucs = "hdemucs"
}

@main
struct Main: ParsableCommand {
  @Option
  var modelType: ModelType = .hdemucs

  @Option
  var modelPath = ""

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
      throw DemucsError.error("Failed to create audio format")
    }

    let frameCount = AVAudioFrameCount(audioFile.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw DemucsError.error("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)
    guard let leftChannelData = buffer.floatChannelData?[0],
          let rightChannelData = buffer.floatChannelData?[1] else {
      throw DemucsError.error("Failed to get channel data")
    }

    let sampleCount = Int(buffer.frameLength)
    return MLX.stacked([
        MLXArray(UnsafeBufferPointer(start: leftChannelData, count: sampleCount)),
        MLXArray(UnsafeBufferPointer(start: rightChannelData, count: sampleCount))
    ])
  }

  func saveWavFile(data: MLXArray, path: String, sampleRate: Double = 44100.0) throws {
    guard data.shape.count == 2 else {
      throw DemucsError.error("Expected 2D array [samples, channels]")
    }
    let numSamples = data.shape[0]
    let numChannels = data.shape[1]

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: sampleRate,
                                     channels: AVAudioChannelCount(numChannels),
                                     interleaved: false) else {
      throw DemucsError.error("Failed to create audio format")
    }

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(numSamples)) else {
      throw DemucsError.error("Failed to create audio buffer")
    }
    buffer.frameLength = AVAudioFrameCount(numSamples)

    let audioData = data.transposed(1, 0).asArray(Float.self)
    for channel in 0..<numChannels {
      guard let channelData = buffer.floatChannelData?[channel] else {
        throw DemucsError.error("Failed to get channel data")
      }
      let channelOffset = channel * numSamples
      audioData.withUnsafeBufferPointer { ptr in
        channelData.update(from: ptr.baseAddress! + channelOffset, count: numSamples)
      }
    }

    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
    try audioFile.write(from: buffer)
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
      throw DemucsError.error("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer, frameCount: actualFramesToPlay)
    playerNode.scheduleBuffer(buffer, at: nil, options: [])
    playerNode.play()
    let playbackDuration = Double(actualFramesToPlay) / sampleRate
    Thread.sleep(forTimeInterval: playbackDuration + 0.5)

    engine.stop()
  }

  func loadModel(type: ModelType, from path: String) throws -> any UnaryLayer {
    var path = path
    if path == "" {
      switch type {
      case .demucs:
        path = "demucs.safetensors"
      case .hdemucs:
        path = "hdemucs.safetensors"
      }
    }
    
    var weights = try loadArrays(url: URL(fileURLWithPath: path))
    var model: UnaryLayer
    switch modelType {
    case .demucs:
      weights = Demucs.transformPytorch(weights)
      model = Demucs()
    case .hdemucs:
      weights = HDemucs.transformPytorch(weights)
      model = HDemucs()
    }

    do {
      try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    } catch {
      print("Error updating model parameters: \(error)")
      let modelParams = model.parameters().flattened().sorted(by: { $0.0 < $1.0 })
      for (key, value) in weights.sorted(by: { $0.key < $1.key }) {
        if let (_, modelParam) = modelParams.first(where: { $0.0 == key }) {
          if modelParam.shape != value.shape {
            print("Mismatch for \(key): got \(value.shape) want \(modelParam.shape)")
          }
        } else {
          print("Weight \(key) not found in model, got \(value.shape)")
        }
      }
      for (key, value) in modelParams {
        if weights[key] == nil {
          print("Model parameter \(key) not found in weights, want \(value.shape)")
        }
      }
      throw error
    }
    return model
  }

  mutating func run() {
    do {
      let model = try loadModel(type: modelType, from: modelPath)

      let input = try loadWavFile(from: inputPath)
      print("Input shape: \(input.shape)")
      let startTime = CFAbsoluteTimeGetCurrent()

      // Permute [channels, samples] to [batch=1, samples, channels]
      let permutedInput = input.expandedDimensions(axis: 0).transposed(axes: [0, 2, 1])
      // Run model
      let output = model(permutedInput)

      // Force evaluation
      eval(output)
      let endTime = CFAbsoluteTimeGetCurrent()
      print("Output shape: \(output.shape), time: \(endTime - startTime)s")

      let outputShape = output.shape
      guard outputShape.count == 4 else {
        throw DemucsError.error("Expected output shape [1, samples, instruments, channels]")
      }

      let instrumentNames = ["drums", "bass", "other", "vocals"]
      try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
      for instrument in 3..<outputShape[2] {  // only save vocals for now
        let instrumentData = output[0, 0..., instrument, 0...]
        let instrumentName = instrument < instrumentNames.count ? instrumentNames[instrument] : "instrument_\(instrument)"

        let outputPath = "\(outputDir)/\(instrumentName).wav"
        try saveWavFile(data: instrumentData, path: outputPath)
        print("Saved \(instrumentName) to \(outputPath)")

        if instrumentName == "vocals" {
          try playAudio(from: outputPath, duration: 3)
        }
      }
    } catch {
      print("Error running model: \(error)")
    }
  }
}
