# Private Diabetic Retinopathy Diagnosis App

A privacy-first web application for diabetic retinopathy diagnosis using machine learning. This application performs medical image analysis locally on the user's device, ensuring patient data privacy.

## Features

-  **Privacy-First Architecture**: All image processing happens on the client side
-  **Image Upload**: Supports JPEG, JPG, and PNG retinal scan images


## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Framework**: Bootstrap 4.2.1
- **Icons**: Font Awesome 5.x
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/PyTorch (inference on device)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anjSingh453/Federated-Learning-/tree/main
cd diabetic-retinopathy-app
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

 

## Usage

1. **Upload Scan**: Click on the upload area or "Click to select retinal scan image" button
2. **Select Image**: Choose a JPEG, JPG, or PNG file from your device
3. **Analyze**: Click the "Analyze Scan" button
4. **View Results**: The diagnosis will be displayed on a new page
5. **Consult Professional**: Always follow up with a qualified healthcare provider

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

**Note**: Only retinal scan images should be uploaded for accurate diagnosis.

## Privacy & Security

This application follows privacy-first principles:

- ✅ No data is sent to external servers
- ✅ All processing occurs locally on your device
- ✅ Images are not stored after analysis
- ✅ No user tracking or analytics

 
 
 
