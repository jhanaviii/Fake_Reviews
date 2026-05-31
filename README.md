# Fake Reviews Detection Project

An AI-powered web application that detects fake reviews using deep learning. Built with DistilBERT, Flask, and Vue.js.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /Users/jhanaviagarwal/PycharmProjects/Fake_Reviews
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if not already trained):**
   ```bash
   python train.py
   ```

4. **Start the web application:**
   ```bash
   python web2.py
   ```

5. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## 🎯 Features

### Core Functionality
- **Single Review Analysis**: Real-time classification of individual reviews
- **Bulk Processing**: Upload CSV files for batch analysis
- **Interactive Dashboard**: Modern web interface with Vue.js and Tailwind CSS
- **Advanced Analytics**: Feature importance, class distribution, and performance metrics
- **RESTful API**: Easy integration with other systems

### Technical Features
- **Deep Learning Model**: DistilBERT-based classifier with custom neural network
- **Real-time Processing**: Instant analysis with confidence scores
- **Data Visualization**: Confusion matrix, training metrics, and feature importance charts
- **Error Handling**: Robust error handling and graceful degradation
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Model Architecture

### Neural Network Structure
- **Base Model**: DistilBERT (distilled version of BERT)
- **Classification Head**: 3-layer neural network (768→256→64→2 neurons)
- **Activation**: ReLU with dropout (0.1) for regularization
- **Output**: Binary classification (0=Fake, 1=Genuine)

### Feature Analysis
The system analyzes reviews for:
- **Excessive Punctuation**: Multiple exclamation/question marks
- **Emotional Language**: Overly positive or negative words
- **Generic Phrases**: Common fake review patterns
- **Length Variation**: Unusually short or long reviews
- **Specificity**: Presence of specific details and reasoning

## 🛠️ Project Structure

```
Fake_Reviews/
├── train.py                 # Model training script
├── evaluate.py              # Model evaluation and metrics
├── web2.py                  # Flask web application
├── new.py                   # Deep learning classifier
├── test_project.py          # Comprehensive test suite
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── PRESENTATION_GUIDE.md   # Technical interview guide
├── best_model2.joblib      # Trained model (763MB)
├── fake_reviews_dataset.csv # Training dataset
├── static/                 # Web interface files
│   └── index.html         # Main dashboard
├── uploads/                # Temporary file uploads
├── classification_report.txt # Model performance metrics
├── evaluation_results.csv   # Detailed evaluation results
├── training_metrics.png     # Training progress visualization
└── confusion_matrix.png     # Model confusion matrix
```

## 🔧 API Endpoints

### Core Endpoints
- `POST /analyze` - Analyze single review
- `POST /bulk_upload` - Process CSV file upload
- `GET /reviews` - Get analysis history
- `GET /metrics` - Get comprehensive metrics
- `GET /feature_importance` - Get feature importance scores
- `GET /class_distribution` - Get review distribution
- `GET /statistics` - Get summary statistics

### Example API Usage
```bash
# Analyze a single review
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing! Highly recommend!"}'

# Get metrics
curl http://localhost:5000/metrics

# Get feature importance
curl http://localhost:5000/feature_importance
```

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: >85% on test data
- **Processing Speed**: ~1-2 seconds per review
- **Memory Usage**: ~2GB RAM for inference
- **Model Size**: 763MB trained model

### Supported Features
- **Text Length**: Handles reviews from 1 word to 500+ words
- **Languages**: Currently optimized for English
- **File Formats**: CSV and TXT for bulk uploads
- **Real-time Analysis**: Instant results with confidence scores

## 🧪 Testing

### Run the Test Suite
```bash
python test_project.py
```

This will test:
- ✅ File structure and dependencies
- ✅ Model loading and initialization
- ✅ Text processing functionality
- ✅ Prediction capabilities
- ✅ Web server startup
- ✅ API endpoint functionality

### Manual Testing
1. **Single Review**: Enter text and click "Analyze Review"
2. **Bulk Upload**: Upload a CSV file with reviews
3. **Metrics Page**: Check visualizations and statistics
4. **API Testing**: Use curl commands or Postman

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Error**
```bash
# Solution: Re-train the model
python train.py
```

**2. Port Already in Use**
```bash
# Solution: Change port in web2.py
app.run(port=5001)
```

**3. Missing Dependencies**
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt
```

**4. CUDA/GPU Issues**
- The model automatically falls back to CPU
- No action needed - will work slower but still function

**5. Dataset Format Issues**
- Ensure CSV has columns: `text_` and `label`
- Check for extra spaces in column names
- Verify file location and permissions

### Error Messages
- **"Model not loaded"**: Run `python train.py` first
- **"No text column found"**: Check CSV file format
- **"Address already in use"**: Change port or kill existing process
- **"Module not found"**: Install missing dependencies

## 🚀 Deployment

### Local Development
```bash
# Development mode with auto-reload
python web2.py
```

### Production Deployment
```bash
# Production mode (disable debug)
# Edit web2.py line 192: app.run(debug=False, port=5000)
python web2.py
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "web2.py"]
```

## 📚 Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face BERT models
- **Flask**: Web framework
- **Vue.js**: Frontend framework
- **Tailwind CSS**: Styling framework
- **Chart.js**: Data visualization

### Architecture Decisions
- **DistilBERT**: Faster than full BERT while maintaining accuracy
- **Flask**: Lightweight and easy to deploy
- **Vue.js**: Reactive frontend with minimal setup
- **Joblib**: Efficient model serialization
- **Modular Design**: Separate training and serving components

## 🎓 For Technical Interviews

### Key Points to Highlight
1. **Deep Learning**: DistilBERT + custom neural network
2. **Full-Stack**: Backend API + frontend dashboard
3. **Real-world Application**: E-commerce fraud detection
4. **Scalability**: Batch processing and API design
5. **Performance**: >85% accuracy with fast inference

### Demo Script
1. Start the application: `python web2.py`
2. Open browser: `http://localhost:5000`
3. Show single review analysis
4. Demonstrate bulk upload
5. Display metrics and visualizations
6. Explain technical architecture

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_project.py`
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Include error handling

## 📄 License

This project is for educational and demonstration purposes.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test suite: `python test_project.py`
3. Review the error logs in the console
4. Check the API documentation above

---

**Happy coding! 🚀**
