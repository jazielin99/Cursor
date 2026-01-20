//
//  GradingViewModel.swift
//  PSAGrader
//
//  ViewModel for managing grading state and API calls
//

import SwiftUI
import Combine

// MARK: - Grading Result Model
struct GradingResult: Codable {
    let success: Bool
    let grade: String?
    let gradeConfidence: Double?
    let tier: String?
    let tierConfidence: Double?
    let gradeProbabilities: [String: Double]?
    let gradingNotes: [String: Any]?
    let upgradeHint: String?
    let error: String?
    
    enum CodingKeys: String, CodingKey {
        case success
        case grade
        case gradeConfidence = "grade_confidence"
        case tier
        case tierConfidence = "tier_confidence"
        case gradeProbabilities = "grade_probabilities"
        case gradingNotes = "grading_notes"
        case upgradeHint = "upgrade_hint"
        case error
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        success = try container.decode(Bool.self, forKey: .success)
        grade = try container.decodeIfPresent(String.self, forKey: .grade)
        gradeConfidence = try container.decodeIfPresent(Double.self, forKey: .gradeConfidence)
        tier = try container.decodeIfPresent(String.self, forKey: .tier)
        tierConfidence = try container.decodeIfPresent(Double.self, forKey: .tierConfidence)
        gradeProbabilities = try container.decodeIfPresent([String: Double].self, forKey: .gradeProbabilities)
        upgradeHint = try container.decodeIfPresent(String.self, forKey: .upgradeHint)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        
        // Handle gradingNotes as a dictionary
        if let notesContainer = try? container.nestedContainer(keyedBy: DynamicCodingKeys.self, forKey: .gradingNotes) {
            var notes: [String: Any] = [:]
            for key in notesContainer.allKeys {
                if let stringValue = try? notesContainer.decode(String.self, forKey: key) {
                    notes[key.stringValue] = stringValue
                } else if let arrayValue = try? notesContainer.decode([String].self, forKey: key) {
                    notes[key.stringValue] = arrayValue
                } else if let doubleValue = try? notesContainer.decode(Double.self, forKey: key) {
                    notes[key.stringValue] = doubleValue
                }
            }
            gradingNotes = notes
        } else {
            gradingNotes = nil
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(success, forKey: .success)
        try container.encodeIfPresent(grade, forKey: .grade)
        try container.encodeIfPresent(gradeConfidence, forKey: .gradeConfidence)
        try container.encodeIfPresent(tier, forKey: .tier)
        try container.encodeIfPresent(tierConfidence, forKey: .tierConfidence)
        try container.encodeIfPresent(gradeProbabilities, forKey: .gradeProbabilities)
        try container.encodeIfPresent(upgradeHint, forKey: .upgradeHint)
        try container.encodeIfPresent(error, forKey: .error)
    }
}

struct DynamicCodingKeys: CodingKey {
    var stringValue: String
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { return nil }
    init?(intValue: Int) { return nil }
}

// MARK: - View Model
@MainActor
class GradingViewModel: ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var gradingResult: GradingResult?
    @Published var isLoading = false
    @Published var showError = false
    @Published var errorMessage = ""
    
    // Settings
    @AppStorage("apiBaseURL") var apiBaseURL = "http://localhost:8000"
    @AppStorage("enableLLMAudit") var enableLLMAudit = false
    
    private var cancellables = Set<AnyCancellable>()
    
    var hasResult: Bool {
        gradingResult != nil || isLoading
    }
    
    // MARK: - API Call
    func predictGrade() {
        guard let image = capturedImage else {
            showError(message: "No image selected")
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            showError(message: "Failed to process image")
            return
        }
        
        isLoading = true
        gradingResult = nil
        
        let url = URL(string: "\(apiBaseURL)/predict")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"card.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        request.timeoutInterval = 120
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: GradingResult.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { [weak self] completion in
                self?.isLoading = false
                if case .failure(let error) = completion {
                    self?.showError(message: "Network error: \(error.localizedDescription)")
                }
            }, receiveValue: { [weak self] result in
                self?.isLoading = false
                if result.success {
                    self?.gradingResult = result
                } else {
                    self?.showError(message: result.error ?? "Unknown error")
                }
            })
            .store(in: &cancellables)
    }
    
    // MARK: - Error Handling
    private func showError(message: String) {
        errorMessage = message
        showError = true
    }
    
    // MARK: - Clear State
    func clearResult() {
        gradingResult = nil
        capturedImage = nil
    }
}
