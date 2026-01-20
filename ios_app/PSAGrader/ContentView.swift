//
//  ContentView.swift
//  PSAGrader
//
//  Main view with camera capture and grade prediction
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var viewModel = GradingViewModel()
    @State private var showCamera = false
    @State private var showPhotoPicker = false
    @State private var showSettings = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 20) {
                    // Header
                    headerSection
                    
                    // Image Preview
                    imagePreviewSection
                    
                    // Results
                    if viewModel.hasResult {
                        resultSection
                    }
                    
                    Spacer()
                    
                    // Action Buttons
                    actionButtons
                }
                .padding()
            }
            .navigationTitle("PSA Grader")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showSettings = true }) {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showCamera) {
                CameraView(image: $viewModel.capturedImage)
            }
            .sheet(isPresented: $showPhotoPicker) {
                PhotoPicker(image: $viewModel.capturedImage)
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(viewModel: viewModel)
            }
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }
    
    // MARK: - Header Section
    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            
            Text("AI Card Grader")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Take a photo to predict PSA grade")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.top, 20)
    }
    
    // MARK: - Image Preview Section
    private var imagePreviewSection: some View {
        Group {
            if let image = viewModel.capturedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .cornerRadius(16)
                    .shadow(radius: 5)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.blue.opacity(0.3), lineWidth: 2)
                    )
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.gray.opacity(0.1))
                        .frame(height: 200)
                    
                    VStack(spacing: 12) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        
                        Text("No image selected")
                            .foregroundColor(.gray)
                    }
                }
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - Result Section
    private var resultSection: some View {
        VStack(spacing: 16) {
            if viewModel.isLoading {
                ProgressView("Analyzing card...")
                    .padding()
            } else if let result = viewModel.gradingResult {
                GradeResultCard(result: result)
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - Action Buttons
    private var actionButtons: some View {
        VStack(spacing: 12) {
            HStack(spacing: 16) {
                // Camera Button
                Button(action: { showCamera = true }) {
                    HStack {
                        Image(systemName: "camera.fill")
                        Text("Camera")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                
                // Photo Library Button
                Button(action: { showPhotoPicker = true }) {
                    HStack {
                        Image(systemName: "photo.fill")
                        Text("Library")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.purple)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
            }
            
            // Grade Button
            if viewModel.capturedImage != nil {
                Button(action: { viewModel.predictGrade() }) {
                    HStack {
                        if viewModel.isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "sparkles")
                        }
                        Text(viewModel.isLoading ? "Analyzing..." : "Predict Grade")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isLoading ? Color.gray : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(viewModel.isLoading)
            }
        }
        .padding(.horizontal)
        .padding(.bottom, 20)
    }
}

// MARK: - Grade Result Card
struct GradeResultCard: View {
    let result: GradingResult
    
    var gradeColor: Color {
        guard let grade = result.grade else { return .gray }
        let num = Int(grade.replacingOccurrences(of: "PSA_", with: "")) ?? 5
        switch num {
        case 10: return .green
        case 9: return .mint
        case 8: return .teal
        case 7: return .blue
        case 5...6: return .orange
        default: return .red
        }
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Grade Badge
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Predicted Grade")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(result.grade?.replacingOccurrences(of: "_", with: " ") ?? "Unknown")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(gradeColor)
                }
                
                Spacer()
                
                // Confidence Circle
                ZStack {
                    Circle()
                        .stroke(gradeColor.opacity(0.3), lineWidth: 8)
                        .frame(width: 70, height: 70)
                    
                    Circle()
                        .trim(from: 0, to: CGFloat(result.gradeConfidence ?? 0))
                        .stroke(gradeColor, lineWidth: 8)
                        .frame(width: 70, height: 70)
                        .rotationEffect(.degrees(-90))
                    
                    Text("\(Int((result.gradeConfidence ?? 0) * 100))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
            }
            
            Divider()
            
            // Tier Info
            HStack {
                Label(result.tier ?? "Unknown", systemImage: "tag.fill")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                if let hint = result.upgradeHint {
                    Text(hint)
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.yellow.opacity(0.2))
                        .cornerRadius(8)
                }
            }
            
            // Grading Notes
            if let notes = result.gradingNotes {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Analysis")
                        .font(.headline)
                    
                    if let centering = notes["centering"] as? String {
                        HStack(alignment: .top) {
                            Image(systemName: "arrow.left.and.right")
                                .foregroundColor(.blue)
                            Text(centering)
                                .font(.caption)
                        }
                    }
                    
                    if let summary = notes["summary"] as? String {
                        HStack(alignment: .top) {
                            Image(systemName: "doc.text")
                                .foregroundColor(.purple)
                            Text(summary)
                                .font(.caption)
                        }
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
            }
        }
        .padding()
        .background(Color.white)
        .cornerRadius(16)
        .shadow(radius: 5)
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
