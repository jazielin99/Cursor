//
//  SettingsView.swift
//  PSAGrader
//
//  Settings for API configuration
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: GradingViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var testingConnection = false
    @State private var connectionStatus: String?
    
    var body: some View {
        NavigationView {
            Form {
                // API Configuration
                Section(header: Text("API Configuration")) {
                    TextField("Server URL", text: $viewModel.apiBaseURL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .keyboardType(.URL)
                    
                    Button(action: testConnection) {
                        HStack {
                            if testingConnection {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                            } else {
                                Image(systemName: "antenna.radiowaves.left.and.right")
                            }
                            Text("Test Connection")
                        }
                    }
                    .disabled(testingConnection)
                    
                    if let status = connectionStatus {
                        Text(status)
                            .font(.caption)
                            .foregroundColor(status.contains("✓") ? .green : .red)
                    }
                }
                
                // Advanced Options
                Section(header: Text("Advanced")) {
                    Toggle("Enable LLM Visual Audit", isOn: $viewModel.enableLLMAudit)
                    
                    Text("When enabled, high-grade predictions (8-10) will be verified by GPT-4o or Gemini for improved accuracy. Requires API key on server.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // About
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Model")
                        Spacer()
                        Text("Binary Triage v2")
                            .foregroundColor(.secondary)
                    }
                    
                    Link(destination: URL(string: "https://github.com/jazielin99/Cursor")!) {
                        HStack {
                            Text("GitHub Repository")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                        }
                    }
                }
                
                // Preset Servers
                Section(header: Text("Quick Setup")) {
                    Button("Use Local Server (localhost:8000)") {
                        viewModel.apiBaseURL = "http://localhost:8000"
                    }
                    
                    Button("Use Default Cloud Server") {
                        viewModel.apiBaseURL = "https://psa-grader-api.example.com"
                    }
                }
                
                // Reset
                Section {
                    Button(role: .destructive) {
                        viewModel.apiBaseURL = "http://localhost:8000"
                        viewModel.enableLLMAudit = false
                        connectionStatus = nil
                    } label: {
                        Text("Reset to Defaults")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func testConnection() {
        testingConnection = true
        connectionStatus = nil
        
        guard let url = URL(string: "\(viewModel.apiBaseURL)/health") else {
            connectionStatus = "✗ Invalid URL"
            testingConnection = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                testingConnection = false
                
                if let error = error {
                    connectionStatus = "✗ \(error.localizedDescription)"
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        connectionStatus = "✓ Connected successfully"
                    } else {
                        connectionStatus = "✗ Server returned \(httpResponse.statusCode)"
                    }
                } else {
                    connectionStatus = "✗ Invalid response"
                }
            }
        }.resume()
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView(viewModel: GradingViewModel())
    }
}
