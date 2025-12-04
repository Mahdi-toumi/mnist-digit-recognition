"""
🔢 MNIST Digit Recognition - Interactive Web App
Created with Streamlit
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# -----------------------------------------------------------
# CONFIG PAGE
# -----------------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------
# SAFE PATH UTIL
# -----------------------------------------------------------
def safe_path(filename):
    base_path = os.path.dirname(os.path.abspath(__file__))  # dossier /app
    return os.path.abspath(os.path.join(base_path, "..", "models", filename))


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = safe_path("best_model.keras")
    if not os.path.exists(model_path):
        st.error(f"❌ best_model.keras introuvable : {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)


# -----------------------------------------------------------
# LOAD FINAL REPORT (pour accuracy & params)
# -----------------------------------------------------------
@st.cache_resource
def load_report():
    report_path = safe_path("final_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------
# LOAD MNIST DATA FOR ANALYSIS
# -----------------------------------------------------------
@st.cache_resource
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------
# LOAD MODEL HISTORY
# -----------------------------------------------------------
@st.cache_resource
def load_model_history():
    history_path = safe_path("model_history.json")
    if not os.path.exists(history_path):
        # Try to load from training_history.json or create empty
        training_history_path = safe_path("training_history.json")
        if os.path.exists(training_history_path):
            with open(training_history_path, "r") as f:
                return json.load(f)
        return None
    with open(history_path, "r") as f:
        return json.load(f)


model = load_model()
report = load_report()
history = load_model_history()
x_train, y_train, x_test, y_test = load_mnist_data()


# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🔢 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🎨 Draw & Predict", "📊 Model Analysis", "📈 Training History"]
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Information")

if report:
    test_acc = report["test_results"]["test_accuracy"] * 100
    params = report["model_architecture"]["total_params"]
    
    st.sidebar.metric("Test Accuracy", f"{test_acc:.2f}%")
    st.sidebar.metric("Total Parameters", f"{params:,}")
    
    # Display model summary - CORRIGÉ pour la structure réelle
    with st.sidebar.expander("📋 Model Summary"):
        st.write(f"**Model Name:** {report['model_architecture'].get('name', 'CNN')}")
        st.write(f"**Total Params:** {report['model_architecture'].get('total_params', 'N/A'):,}")
        st.write(f"**Trainable Params:** {report['model_architecture'].get('trainable_params', 'N/A'):,}")
        st.write(f"**Number of Layers:** {len(report['model_architecture'].get('layers', []))}")
        st.write(f"**Image Size:** {report['dataset_info'].get('image_size', '28x28')}")
        st.write(f"**Number of Classes:** {report['dataset_info'].get('num_classes', 10)}")
        
else:
    st.sidebar.warning("No final_report.json found.")

st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. **Draw & Predict**: Draw digits and get predictions
2. **Model Analysis**: View model architecture and performance
3. **Training History**: See training progress and metrics
""")


# -----------------------------------------------------------
# PAGE 1: DRAW & PREDICT
# -----------------------------------------------------------
if page == "🎨 Draw & Predict":
    st.title("✏️ Draw & Predict Digits")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw a Digit")
        
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        col_btn1, col_btn2 = st.columns(2)
        predict_button = col_btn1.button("🔮 Predict", type="primary")
        
        if col_btn2.button("🗑️ Clear"):
            st.rerun()
        
        # Display sample MNIST digits for reference
        with st.expander("📚 MNIST Reference Digits"):
            st.markdown("**Official MNIST samples:**")
            # Show first 10 training samples
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            axes = axes.flatten()
            for i in range(10):
                # Find first occurrence of each digit
                idx = np.where(y_train == i)[0][0]
                axes[i].imshow(x_train[idx], cmap='gray')
                axes[i].set_title(f"Label: {i}")
                axes[i].axis('off')
            st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if predict_button and canvas_result.image_data is not None:
            # ---------------------------------------------------
            # PREPROCESSING
            # ---------------------------------------------------
            img = canvas_result.image_data
            
            if img.shape[2] == 4:  # RGBA
                alpha = img[:, :, 3]
                rgb_gray = np.mean(img[:, :, :3], axis=2)
                gray = np.where(alpha > 0, 255 - rgb_gray, 0)
            else:
                gray = 255 - np.mean(img[:, :, :3], axis=2)
            
            # Resize and process
            gray_resized = cv2.resize(gray, (28, 28))
            _, gray_threshold = cv2.threshold(gray_resized, 10, 255, cv2.THRESH_BINARY)
            gray_normalized = gray_threshold.astype("float32") / 255.0
            gray_inverted = 1.0 - gray_normalized
            img_processed = gray_inverted.reshape(1, 28, 28, 1)
            
            # ---------------------------------------------------
            # PREDICTION
            # ---------------------------------------------------
            prediction = model.predict(img_processed, verbose=0)
            predicted_digit = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_digit] * 100)
            
            # Display prediction
            pred_col1, pred_col2 = st.columns([1, 2])
            
            with pred_col1:
                st.markdown(f"""
                <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:15px;'>
                    <h1 style='font-size:100px; color: white; margin:0;'>{predicted_digit}</h1>
                    <p style='font-size:24px; color:white;'>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show preprocessed image
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(gray_inverted, cmap='gray', vmin=0, vmax=1)
                ax.set_title("Input to Model")
                ax.axis('off')
                st.pyplot(fig)
            
            with pred_col2:
                # Probability distribution
                st.markdown("**Probability Distribution**")
                
                # Create interactive chart
                digits = list(range(10))
                probs = prediction[0]
                
                fig = px.bar(
                    x=digits,
                    y=probs,
                    labels={'x': 'Digit', 'y': 'Probability'},
                    title=f"Prediction Probabilities",
                    color=probs,
                    color_continuous_scale='Blues'
                )
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                fig.update_layout(
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities table
                prob_df = pd.DataFrame({
                    "Digit": digits,
                    "Probability (%)": [f"{p*100:.2f}%" for p in probs]
                })
                
                # Find the max probability
                max_prob_idx = prob_df["Probability (%)"].apply(lambda x: float(x[:-1])).idxmax()
                
                # Display with highlighting
                def highlight_max(row):
                    if row.name == max_prob_idx:
                        return ['background-color: #1a1c24'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    prob_df.style.apply(highlight_max, axis=1)
                )
            
            # ---------------------------------------------------
            # PREPROCESSING DETAILS (collapsible)
            # ---------------------------------------------------
            with st.expander("🔍 View Preprocessing Steps"):
                steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
                
                with steps_col1:
                    # Normalize original image for display
                    img_display = img.copy()
                    if img_display.dtype != np.uint8:
                        img_display = (img_display * 255).astype(np.uint8)
                    st.image(img_display, caption="1. Original Canvas", width=150)
                
                with steps_col2:
                    # Normalize grayscale for display (0-255 to 0-1)
                    gray_display = gray.astype(np.float32) / 255.0
                    st.image(gray_display, caption="2. Grayscale", clamp=True, width=150)
                
                with steps_col3:
                    # Normalize resized for display
                    gray_resized_display = gray_resized.astype(np.float32) / 255.0
                    st.image(gray_resized_display, caption="3. Resized 28x28", clamp=True, width=150)
                
                with steps_col4:
                    # Final image is already normalized 0-1
                    st.image(gray_inverted, caption="4. Final Input", clamp=True, width=150)
                
                # Pixel values
                st.markdown("**Pixel Values Matrix (10x10):**")
                pixel_matrix = pd.DataFrame(gray_inverted[:10, :10])
                st.dataframe(pixel_matrix.style.format("{:.3f}"))
        
        elif predict_button:
            st.warning("⚠️ Please draw a digit first!")
            st.info("Try drawing a clear digit in the center of the canvas.")
        else:
            st.info("👆 **Draw a digit (0-9)** in the canvas and click **Predict**")
            st.markdown("""
            **Tips for best results:**
            - Draw clear, centered digits
            - Use the full canvas space
            - Avoid very thin strokes
            - Refer to MNIST samples for style
            """)


# -----------------------------------------------------------
# PAGE 2: MODEL ANALYSIS
# -----------------------------------------------------------
elif page == "📊 Model Analysis":
    st.title("📊 Model Analysis & Performance")
    st.markdown("---")
    
    if not report:
        st.error("Model report not found. Please train the model first.")
        st.stop()
    
    # ---------------------------------------------------
    # SECTION 1: MODEL ARCHITECTURE
    # ---------------------------------------------------
    st.header("🏗️ Model Architecture")
    
    arch_col1, arch_col2 = st.columns([2, 1])
    
    with arch_col1:
        st.subheader("Model Summary")
        
        # Display layers information - CORRIGÉ pour la structure réelle
        if 'layers' in report['model_architecture']:
            layers_data = report["model_architecture"]["layers"]
            
            # Convertir en DataFrame avec des valeurs par défaut
            layers_list = []
            for layer in layers_data:
                layer_info = {
                    "Type": layer.get("type", "Unknown"),
                    "Details": ""
                }
                
                # Construire la description détaillée
                details_parts = []
                if "filters" in layer:
                    details_parts.append(f"Filters: {layer['filters']}")
                if "kernel_size" in layer:
                    details_parts.append(f"Kernel: {layer['kernel_size']}")
                if "pool_size" in layer:
                    details_parts.append(f"Pool: {layer['pool_size']}")
                if "units" in layer:
                    details_parts.append(f"Units: {layer['units']}")
                if "activation" in layer:
                    details_parts.append(f"Activation: {layer['activation']}")
                if "rate" in layer:
                    details_parts.append(f"Dropout: {layer['rate']}")
                
                layer_info["Details"] = ", ".join(details_parts) if details_parts else "-"
                layers_list.append(layer_info)
            
            layers_df = pd.DataFrame(layers_list)
            
            # Afficher le tableau sans spécifier de configuration de colonnes problématique
            st.dataframe(layers_df)
        else:
            st.warning("Layer information not available in report.")
    
    with arch_col2:
        st.subheader("Architecture Overview")
        
        # Model metrics - CORRIGÉ pour la structure réelle
        metrics_data = {
            "Metric": ["Total Parameters", "Trainable Parameters", 
                      "Model Type", "Input Size",
                      "Output Classes", "Number of Layers"],
            "Value": [
                f"{report['model_architecture'].get('total_params', 'N/A'):,}",
                f"{report['model_architecture'].get('trainable_params', 'N/A'):,}",
                report['model_architecture'].get('name', 'CNN'),
                report['dataset_info'].get('image_size', '28x28'),
                str(report['dataset_info'].get('num_classes', 10)),
                str(len(report['model_architecture'].get('layers', [])))
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)
        
        # Model type visualization
        if 'trainable_params' in report['model_architecture']:
            trainable = report['model_architecture']['trainable_params']
            total = report['model_architecture']['total_params']
            non_trainable = total - trainable
            
            if non_trainable > 0:  # Only show pie chart if there are non-trainable params
                fig = px.pie(
                    values=[trainable, non_trainable],
                    names=['Trainable', 'Non-Trainable'],
                    title='Parameter Distribution',
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ---------------------------------------------------
    # SECTION 2: TRAINING RESULTS
    # ---------------------------------------------------
    st.header("📈 Training Results")
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Loss", "Confusion Analysis", "Detailed Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Accuracy")
            train_acc = report["training_results"]["final_train_accuracy"] * 100
            st.metric("Final Training Accuracy", f"{train_acc:.2f}%")
            
            # Display validation accuracy if available
            if 'final_val_accuracy' in report['training_results']:
                val_acc = report['training_results']['final_val_accuracy'] * 100
                st.metric("Final Validation Accuracy", f"{val_acc:.2f}%")
            
            # Epoch-wise accuracy if available in history
            if history and 'accuracy' in history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['accuracy'],
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='blue', width=2)
                ))
                if 'val_accuracy' in history:
                    fig.add_trace(go.Scatter(
                        y=history['val_accuracy'],
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='red', width=2)
                    ))
                fig.update_layout(
                    title='Accuracy per Epoch',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Test Accuracy")
            test_acc = report["test_results"]["test_accuracy"] * 100
            st.metric("Final Test Accuracy", f"{test_acc:.2f}%")
            
            # Display additional test metrics
            if 'correct_predictions' in report['test_results']:
                correct = report['test_results']['correct_predictions']
                incorrect = report['test_results']['incorrect_predictions']
                total = correct + incorrect
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Correct Predictions", f"{correct:,}")
                with col2_2:
                    st.metric("Error Rate", f"{report['test_results'].get('error_rate_percent', 0):.2f}%")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Loss")
            train_loss = report["training_results"]["final_train_loss"]
            st.metric("Final Training Loss", f"{train_loss:.4f}")
            
            # Display validation loss if available
            if 'final_val_loss' in report['training_results']:
                val_loss = report['training_results']['final_val_loss']
                st.metric("Final Validation Loss", f"{val_loss:.4f}")
            
            # Epoch-wise loss if available in history
            if history and 'loss' in history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                ))
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='red', width=2)
                    ))
                fig.update_layout(
                    title='Loss per Epoch',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Test Loss")
            test_loss = report["test_results"]["test_loss"]
            st.metric("Final Test Loss", f"{test_loss:.4f}")
            
            # Display model efficiency
            if test_loss < 0.1:
                st.success("✅ Excellent model performance")
            elif test_loss < 0.3:
                st.info("📊 Good model performance")
            else:
                st.warning("⚠️ Consider improving the model")
    
    with tab3:
        st.subheader("Confusion Analysis")
        
        # Display most confused pairs if available
        if 'confusion_analysis' in report and 'most_confused_pairs' in report['confusion_analysis']:
            confused_pairs = report['confusion_analysis']['most_confused_pairs']
            
            # Create DataFrame for confused pairs
            confused_df = pd.DataFrame(confused_pairs)
            
            # Display as table
            st.dataframe(
                confused_df[['rank', 'true_label', 'predicted_label', 'count', 'percentage']]
            )
            
            # Create visualization
            fig = px.bar(
                confused_df,
                x='true_label',
                y='count',
                color='predicted_label',
                title='Most Confused Digit Pairs',
                labels={'true_label': 'True Label', 'count': 'Number of Errors', 'predicted_label': 'Predicted as'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confusion analysis data not available in report.")
    
    with tab4:
        st.subheader("Detailed Classification Metrics")
        
        # Display per-class metrics if available
        if 'per_class_metrics' in report:
            # Convert per_class_metrics to DataFrame
            metrics_list = []
            for digit, metrics in report['per_class_metrics'].items():
                metrics['digit'] = int(digit)
                metrics_list.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_list)
            
            # Format for display
            display_df = metrics_df[['digit', 'precision', 'recall', 'f1_score', 'support']].copy()
            display_df.columns = ['Digit', 'Precision', 'Recall', 'F1-Score', 'Support']
            
            # Format as percentages for display
            for col in ['Precision', 'Recall', 'F1-Score']:
                display_df[col] = (display_df[col] * 100).round(2)
            
            st.dataframe(display_df)
            
            # Create visualization of per-class performance
            fig = go.Figure()
            
            metrics_to_plot = ['precision', 'recall', 'f1_score']
            colors = ['blue', 'green', 'red']
            
            for metric, color in zip(metrics_to_plot, colors):
                fig.add_trace(go.Bar(
                    x=metrics_df['digit'],
                    y=metrics_df[metric] * 100,
                    name=metric.replace('_', ' ').title(),
                    marker_color=color
                ))
            
            fig.update_layout(
                title='Per-Class Performance Metrics',
                xaxis_title='Digit',
                yaxis_title='Score (%)',
                barmode='group',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Per-class metrics not available in report.")
    
    # ---------------------------------------------------
    # SECTION 3: CONFIDENCE ANALYSIS
    # ---------------------------------------------------
    if 'confidence_statistics' in report:
        st.header("🔍 Confidence Analysis")
        
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            st.subheader("Correct Predictions")
            correct_stats = report['confidence_statistics']['correct_predictions']
            
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Mean Confidence", f"{correct_stats['mean']*100:.1f}%")
            with col1_2:
                st.metric("Median Confidence", f"{correct_stats['median']*100:.1f}%")
            with col1_3:
                st.metric("Min Confidence", f"{correct_stats['min']*100:.1f}%")
        
        with conf_col2:
            st.subheader("Incorrect Predictions")
            incorrect_stats = report['confidence_statistics']['incorrect_predictions']
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Mean Confidence", f"{incorrect_stats['mean']*100:.1f}%")
            with col2_2:
                st.metric("Median Confidence", f"{incorrect_stats['median']*100:.1f}%")
            with col2_3:
                st.metric("Min Confidence", f"{incorrect_stats['min']*100:.1f}%")
        
        # Insight
        if correct_stats['mean'] > 0.9 and incorrect_stats['mean'] < 0.7:
            st.success("✅ Model shows good discrimination: high confidence on correct predictions, lower confidence on errors.")
        else:
            st.warning("⚠️ Model confidence doesn't always correlate with correctness. Consider calibration.")
    
    # ---------------------------------------------------
    # SECTION 4: TRAINING CONFIGURATION
    # ---------------------------------------------------
    st.header("⚙️ Training Configuration")
    
    if 'training_config' in report:
        config_items = [
            ("Optimizer", report['training_config'].get('optimizer', 'N/A')),
            ("Learning Rate", report['training_config'].get('learning_rate_initial', 'N/A')),
            ("Batch Size", report['training_config'].get('batch_size', 'N/A')),
            ("Epochs Completed", report['training_config'].get('epochs_completed', 'N/A')),
            ("Training Duration", f"{report['training_config'].get('training_duration_minutes', 0):.2f} minutes"),
            ("Early Stopping", "Yes" if report['training_config'].get('early_stopping', False) else "No")
        ]
        
        for label, value in config_items:
            st.write(f"**{label}:** {value}")


# -----------------------------------------------------------
# PAGE 3: TRAINING HISTORY
# -----------------------------------------------------------
elif page == "📈 Training History":
    st.title("📈 Training History & Evolution")
    st.markdown("---")
    
    if not history:
        st.warning("Training history not found. Displaying available information from report...")
        
        if report and 'training_results' in report:
            # Display final results from report
            st.header("Final Training Results")
            
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                train_acc = report["training_results"]["final_train_accuracy"] * 100
                train_loss = report["training_results"]["final_train_loss"]
                
                st.metric("Training Accuracy", f"{train_acc:.2f}%")
                st.metric("Training Loss", f"{train_loss:.4f}")
            
            with results_col2:
                if 'final_val_accuracy' in report['training_results']:
                    val_acc = report['training_results']['final_val_accuracy'] * 100
                    val_loss = report['training_results']['final_val_loss']
                    
                    st.metric("Validation Accuracy", f"{val_acc:.2f}%")
                    st.metric("Validation Loss", f"{val_loss:.4f}")
            
            # Display training configuration
            if 'training_config' in report:
                st.header("Training Configuration")
                config_items = [
                    ("Optimizer", report['training_config'].get('optimizer', 'N/A')),
                    ("Learning Rate", report['training_config'].get('learning_rate_initial', 'N/A')),
                    ("Batch Size", report['training_config'].get('batch_size', 'N/A')),
                    ("Epochs Completed", report['training_config'].get('epochs_completed', 'N/A')),
                    ("Training Duration", f"{report['training_config'].get('training_duration_minutes', 0):.2f} minutes"),
                    ("Early Stopping", "Yes" if report['training_config'].get('early_stopping', False) else "No")
                ]
                
                for label, value in config_items:
                    st.write(f"**{label}:** {value}")
        
        else:
            st.error("No training history or report data available.")
            st.info("Please train the model to generate history data.")
    
    else:
        # ---------------------------------------------------
        # SECTION 1: TRAINING PROGRESS VISUALIZATION
        # ---------------------------------------------------
        st.header("Training Progress Over Time")
        
        # Create metrics comparison
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Accuracy Evolution", "Loss Evolution", "Combined View"])
        
        with metrics_tab1:
            if 'accuracy' in history:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['accuracy']))),
                    y=history['accuracy'],
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6)
                ))
                
                if 'val_accuracy' in history:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(history['val_accuracy']))),
                        y=history['val_accuracy'],
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                        marker=dict(size=6)
                    ))
                
                fig.update_layout(
                    title='Model Accuracy Evolution',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    template='plotly_white',
                    hovermode='x unified',
                    height=500
                )
                
                # Add annotations for best accuracy
                if 'val_accuracy' in history:
                    best_epoch = np.argmax(history['val_accuracy'])
                    best_acc = history['val_accuracy'][best_epoch]
                    fig.add_annotation(
                        x=best_epoch,
                        y=best_acc,
                        text=f'Best: {best_acc:.3f}',
                        showarrow=True,
                        arrowhead=2
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Accuracy history not available")
        
        with metrics_tab2:
            if 'loss' in history:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['loss']))),
                    y=history['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=6)
                ))
                
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(history['val_loss']))),
                        y=history['val_loss'],
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='#2ca02c', width=3, dash='dash'),
                        marker=dict(size=6)
                    ))
                
                fig.update_layout(
                    title='Model Loss Evolution',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_white',
                    hovermode='x unified',
                    height=500
                )
                
                # Add annotations for best loss
                if 'val_loss' in history:
                    best_epoch = np.argmin(history['val_loss'])
                    best_loss = history['val_loss'][best_epoch]
                    fig.add_annotation(
                        x=best_epoch,
                        y=best_loss,
                        text=f'Best: {best_loss:.3f}',
                        showarrow=True,
                        arrowhead=2
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loss history not available")
        
        with metrics_tab3:
            # Combined view
            fig = go.Figure()
            
            has_data = False
            
            if 'accuracy' in history:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['accuracy']))),
                    y=history['accuracy'],
                    mode='lines',
                    name='Train Accuracy',
                    yaxis='y1',
                    line=dict(color='blue', width=2)
                ))
                has_data = True
            
            if 'val_accuracy' in history:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['val_accuracy']))),
                    y=history['val_accuracy'],
                    mode='lines',
                    name='Val Accuracy',
                    yaxis='y1',
                    line=dict(color='lightblue', width=2)
                ))
                has_data = True
            
            if 'loss' in history:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['loss']))),
                    y=history['loss'],
                    mode='lines',
                    name='Train Loss',
                    yaxis='y2',
                    line=dict(color='red', width=2)
                ))
                has_data = True
            
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history['val_loss']))),
                    y=history['val_loss'],
                    mode='lines',
                    name='Val Loss',
                    yaxis='y2',
                    line=dict(color='orange', width=2)
                ))
                has_data = True
            
            if has_data:
                fig.update_layout(
                    title='Combined Training Metrics',
                    xaxis_title='Epoch',
                    yaxis=dict(
                        title='Accuracy',
                        titlefont=dict(color='blue'),
                        tickfont=dict(color='blue')
                    ),
                    yaxis2=dict(
                        title='Loss',
                        titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        overlaying='y',
                        side='right'
                    ),
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history data available for combined view")
        
        # ---------------------------------------------------
        # SECTION 2: STATISTICAL ANALYSIS
        # ---------------------------------------------------
        if history and ('accuracy' in history or 'loss' in history):
            st.header("📊 Statistical Analysis")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                if 'accuracy' in history:
                    final_train_acc = history['accuracy'][-1]
                    initial_train_acc = history['accuracy'][0]
                    improvement = (final_train_acc - initial_train_acc) * 100
                    
                    st.metric(
                        "Training Accuracy Improvement",
                        f"{improvement:.1f}%",
                        delta=f"{final_train_acc:.3f}"
                    )
            
            with stat_col2:
                if 'val_accuracy' in history and 'accuracy' in history:
                    overfitting_gap = (history['accuracy'][-1] - history['val_accuracy'][-1]) * 100
                    
                    st.metric(
                        "Overfitting Gap",
                        f"{overfitting_gap:.2f}%",
                        delta="Low" if overfitting_gap < 5 else "High",
                        delta_color="normal" if overfitting_gap < 5 else "inverse"
                    )
            
            with stat_col3:
                if 'loss' in history:
                    final_loss = history['loss'][-1]
                    initial_loss = history['loss'][0]
                    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
                    
                    st.metric(
                        "Loss Reduction",
                        f"{loss_reduction:.1f}%",
                        delta=f"{final_loss:.4f}"
                    )
            
            # ---------------------------------------------------
            # SECTION 3: RAW HISTORY DATA
            # ---------------------------------------------------
            with st.expander("📋 View Raw Training History Data"):
                history_df = pd.DataFrame(history)
                st.dataframe(history_df)
                
                # Option to download history
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download History as CSV",
                    data=csv,
                    file_name="model_training_history.csv",
                    mime="text/csv"
                )


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using TensorFlow and Streamlit • MNIST Digit Recognition System</p>
    <p><small>Model Accuracy: {:.2f}% • Parameters: {:,}</small></p>
</div>
""".format(
    report["test_results"]["test_accuracy"] * 100 if report else 0,
    report["model_architecture"]["total_params"] if report else 0
), unsafe_allow_html=True)