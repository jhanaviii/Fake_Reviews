import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import our improved classifier
from new import DeepReviewClassifier, clean_text


def load_and_preprocess_data(file_path):
    """Load and preprocess the Kaggle fake reviews dataset"""
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Check if required columns exist
        if 'text_' not in df.columns:
            print("Warning: 'text_' column not found. Checking for alternative column names...")
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            if text_columns:
                df['text_'] = df[text_columns[0]]
                print(f"Using column '{text_columns[0]}' as text column")
            else:
                raise ValueError("No text column found in dataset")
        
        if 'label' not in df.columns:
            print("Warning: 'label' column not found. Checking for alternative column names...")
            label_columns = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]
            if label_columns:
                df['label'] = df[label_columns[0]]
                print(f"Using column '{label_columns[0]}' as label column")
            else:
                raise ValueError("No label column found in dataset")

        # Extract features and labels
        texts = df['text_'].apply(clean_text).values
        labels = (df['label'] == 'OR').astype(int).values
        
        print(f"Preprocessing complete. Genuine reviews: {sum(labels)}, Fake reviews: {len(labels) - sum(labels)}")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating sample data for testing...")
        return generate_sample_data()


def generate_sample_data():
    """Generate sample data for testing when dataset is not available"""
    print("Creating sample dataset for testing...")
    
    # Sample genuine reviews
    genuine_reviews = [
        "This product exceeded my expectations. The quality is outstanding and it works exactly as advertised. I've been using it for three months now and it's still performing perfectly. The customer service was also very helpful when I had questions about setup.",
        "I bought this item after reading several reviews and I'm very satisfied with my purchase. The build quality is solid and the features work well. It's a bit pricey but worth the investment for the quality you get.",
        "Great product! The design is modern and functional. I appreciate the attention to detail in the packaging and the included instructions were clear and easy to follow. Would definitely recommend to others.",
        "This is exactly what I was looking for. The specifications match the description perfectly and it arrived on time. The seller was professional and the product was well-packaged. Very happy with this purchase.",
        "I've been using this for about a week now and I'm impressed with the performance. The interface is intuitive and the results are consistent. It's a reliable product that does what it promises."
    ]
    
    # Sample fake reviews
    fake_reviews = [
        "AMAZING!!! BEST PRODUCT EVER!!! I LOVE IT SO MUCH!!! BUY NOW!!! DON'T WAIT!!!",
        "This is the worst thing I've ever bought. Terrible quality. Horrible service. Never buy this. Awful product.",
        "Great product! Highly recommend! Best ever! Fantastic! Wonderful! Amazing! Perfect!",
        "I hate this so much. It's terrible. The worst. Awful. Horrible. Bad. Terrible. Never buy.",
        "This product is amazing and fantastic and wonderful and perfect and the best ever and I love it so much and everyone should buy it now because it's incredible and outstanding and marvelous!"
    ]
    
    # Create balanced dataset
    all_reviews = genuine_reviews * 10 + fake_reviews * 10  # 50 genuine, 50 fake
    all_labels = [1] * 50 + [0] * 50  # 1 for genuine, 0 for fake
    
    print(f"Generated sample dataset with {len(all_reviews)} reviews")
    return np.array(all_reviews), np.array(all_labels)


def plot_training_metrics(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot learning curves
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Training', linewidth=2)
    plt.plot(history['val_acc'], label='Validation', linewidth=2)
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('static/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training metrics plot saved to static/training_metrics.png")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with enhanced styling"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Genuine'],
                yticklabels=['Fake', 'Genuine'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix plot saved to static/confusion_matrix.png")


def train_model(file_path='fake_reviews_dataset.csv',
                model_save_path='best_model2.joblib',
                batch_size=32,
                epochs=3,
                learning_rate=2e-5,
                max_length=128,
                test_size=0.2,
                random_state=42):
    """Train the model on the Kaggle fake reviews dataset"""

    print("="*60)
    print("FAKE REVIEW DETECTION MODEL TRAINING")
    print("="*60)
    
    print("\nLoading and preprocessing data...")
    texts, labels = load_and_preprocess_data(file_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"\nData split complete:")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training genuine: {sum(y_train)}, fake: {len(y_train) - sum(y_train)}")
    print(f"Test genuine: {sum(y_test)}, fake: {len(y_test) - sum(y_test)}")

    # Initialize model
    print("\nInitializing model...")
    classifier = DeepReviewClassifier(max_length=max_length)

    # Training metrics history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    print("\nStarting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Training phase
        classifier.bert.train()
        classifier.classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        # Create batches
        for i in tqdm(range(0, len(X_train), batch_size), desc=f'Training'):
            batch_texts = X_train[i:i + batch_size]
            batch_labels = torch.tensor(y_train[i:i + batch_size], dtype=torch.long).to(device)

            # Tokenize and encode batch
            encoded = classifier.tokenizer(
                batch_texts.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Forward pass
            outputs = classifier.forward(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            classifier.optimizer.step()
            classifier.optimizer.zero_grad()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # Calculate training metrics
        train_loss = total_loss / (len(X_train) / batch_size)
        train_acc = 100 * correct / total

        # Validation phase
        print("Running validation...")
        classifier.bert.eval()
        classifier.classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), desc='Validation'):
                batch_texts = X_test[i:i + batch_size]
                batch_labels = torch.tensor(y_test[i:i + batch_size], dtype=torch.long).to(device)

                encoded = classifier.tokenizer(
                    batch_texts.tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                outputs = classifier.forward(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate validation metrics
        val_loss = val_loss / (len(X_test) / batch_size)
        val_acc = 100 * val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

    # Final evaluation
    print("\n" + "="*60)
    print("GENERATING FINAL EVALUATION METRICS")
    print("="*60)
    
    y_pred = classifier.predict(X_test)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Fake', 'Genuine'])
    print(report)
    
    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to classification_report.txt")

    # Plot metrics
    print("\nGenerating visualizations...")
    plot_training_metrics(history)
    plot_confusion_matrix(y_test, y_pred)

    # Save the model
    print(f"\nSaving model to {model_save_path}...")
    import joblib
    joblib.dump(classifier, model_save_path)
    print("Model saved successfully!")

    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print(f"Visualizations saved to: static/")
    print(f"Classification report saved to: classification_report.txt")
    print("\nYou can now run the web application with: python web2.py")

    return classifier


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Train the model
    classifier = train_model(
        file_path='fake_reviews_dataset.csv',
        model_save_path='best_model2.joblib',
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,
        max_length=128
    )

