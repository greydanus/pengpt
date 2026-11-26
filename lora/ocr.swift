// Handwriting OCR with Apple's Vision framework, used only to anchor transcript words to crops.
// usage: swift lora/ocr.swift page.png [page.png ...]  ->  JSON lines on stdout:
// {"image": path, "w": W, "h": H, "words": [{"text": "...", "box": [x0, y0, x1, y1]}, ...]}  (pixels, y down)
import Foundation
import Vision
import AppKit

func ocr(_ path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return ["image": path, "error": "unreadable"] }
    let W = Double(cg.width), H = Double(cg.height)
    var words: [[String: Any]] = []
    let req = VNRecognizeTextRequest { req, _ in
        for obs in (req.results as? [VNRecognizedTextObservation]) ?? [] {
            guard let cand = obs.topCandidates(1).first else { continue }
            let s = cand.string
            var idx = s.startIndex
            for part in s.split(separator: " ", omittingEmptySubsequences: true) {
                guard let r = s.range(of: part, range: idx..<s.endIndex) else { continue }
                idx = r.upperBound
                if let b = try? cand.boundingBox(for: r)?.boundingBox {
                    words.append(["text": String(part),
                                  "box": [b.minX * W, (1 - b.maxY) * H, b.maxX * W, (1 - b.minY) * H],
                                  "conf": cand.confidence])
                }
            }
        }
    }
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = ["en-US"]
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    return ["image": path, "w": W, "h": H, "words": words]
}

for path in CommandLine.arguments.dropFirst() {
    let out = ocr(path)
    if let data = try? JSONSerialization.data(withJSONObject: out), let s = String(data: data, encoding: .utf8) {
        print(s)
    }
}
