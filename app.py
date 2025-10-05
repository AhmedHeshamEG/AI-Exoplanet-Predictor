import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the trained model and scaler
with open('isolation_forest_model.pkl', 'rb') as f:
    iso_forest = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_normality(st_teff, st_mass, st_rad, st_met, st_lum, st_age, st_vsin):
    """
    Predict the Probability score for a stellar system.
    Returns a score from 0 (Not having exoplanets) to 1 (Has exoplanets).
    All values must be provided.
    """
    # Check if any value is None
    values = [st_teff, st_mass, st_rad, st_met, st_lum, st_age, st_vsin]
    if any(v is None for v in values):
        return None, "❌ ERROR: You must insert all data in order to calculate the probability score. Please fill in all fields."
    
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'st_teff': [st_teff],
        'st_mass': [st_mass],
        'st_rad': [st_rad],
        'st_met': [st_met],
        'st_lum': [st_lum],
        'st_age': [st_age],
        'st_vsin': [st_vsin]
    })
    
    # Get anomaly score from the model
    anomaly_score = -iso_forest.score_samples(input_data)
    anomaly_score = anomaly_score.reshape(-1, 1)
    
    # Normalize and invert the score
    normalized_score = scaler.transform(anomaly_score)
    normality_score = 1 - normalized_score[0][0]
    
    # Create output dataframe
    output_df = input_data.copy()
    output_df['probability_score'] = normality_score
    
    return output_df, f"✅ Probability Score: {normality_score:.4f}"

def predict_from_csv(file):
    """
    Predict probability scores for multiple stellar systems of having exoplanets from a CSV file.
    Rows with missing values will be dropped.
    """
    if file is None:
        return None, "❌ Please upload a CSV file", None
    
    # Read the CSV file
    df = pd.read_csv(file.name)
    
    # Required columns
    required_columns = ['st_teff', 'st_mass', 'st_rad', 'st_met', 'st_lum', 'st_age', 'st_vsin']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return None, f"❌ ERROR: CSV is missing required columns: {', '.join(missing_columns)}\n\nRequired columns: st_teff, st_mass, st_rad, st_met, st_lum, st_age, st_vsin", None
    
    # Store original row count
    original_count = len(df)
    
    # Select only required columns
    df_input = df[required_columns].copy()
    
    # Drop rows with any missing values
    df_input = df_input.dropna()
    df = df.loc[df_input.index]  # Keep only the same rows in original df
    
    dropped_count = original_count - len(df_input)
    
    if len(df_input) == 0:
        return None, "❌ ERROR: All rows contain missing values. You must provide complete data for all columns.", None
    
    # Get anomaly scores
    anomaly_scores = -iso_forest.score_samples(df_input)
    anomaly_scores = anomaly_scores.reshape(-1, 1)
    
    # Normalize and invert
    normalized_scores = scaler.transform(anomaly_scores)
    normality_scores = 1 - normalized_scores.flatten()
    
    # Add scores to original dataframe
    result_df = df.copy()
    result_df['probability_score'] = normality_scores
    
    # Sort by probability score (highest first)
    result_df = result_df.sort_values(by='probability_score', ascending=False)
    
    # Save to CSV for download
    output_csv_path = 'predictions_with_scores.csv'
    result_df.to_csv(output_csv_path, index=False)
    
    status_msg = f"✅ Predictions complete!\n\n"
    status_msg += f"Total rows processed: {len(result_df)}\n"
    if dropped_count > 0:
        status_msg += f"Rows dropped (missing data): {dropped_count}\n"
    
    return result_df, status_msg, output_csv_path

# Create Gradio interface
with gr.Blocks(title="Stellar System Probability Predictor") as demo:
    gr.Markdown("# Stellar System Probability Predictor")
    gr.Markdown("### ⚠️ **Important: You must insert all data in order to calculate the probability score**")
    gr.Markdown("Predict The Probability score for a stellar system to host at least 1 exoplanet (1 = Certainly, 0 = Impossible)")
    
    with gr.Tabs():
        with gr.Tab("Single Prediction"):
            gr.Markdown("### Fill in all fields below:")
            with gr.Row():
                with gr.Column():
                    st_teff = gr.Number(label="Stellar Effective Temperature (K) *", value=None)
                    st_mass = gr.Number(label="Stellar Mass (Solar masses) *", value=None)
                    st_rad = gr.Number(label="Stellar Radius (Solar radii) *", value=None)
                
                with gr.Column():
                    st_met = gr.Number(label="Stellar Metallicity [dex] *", value=None)
                    st_lum = gr.Number(label="Stellar Luminosity (log(Solar)) *", value=None)
                    st_age = gr.Number(label="Stellar Age (Gyr) *", value=None)
                    st_vsin = gr.Number(label="Stellar Rotation Velocity (km/s) *", value=None)
            
            gr.Markdown("**All fields marked with * are required**")
            
            predict_btn = gr.Button("Predict Probability Score", variant="primary")
            
            with gr.Row():
                normality_output = gr.Textbox(label="Result", interactive=False)
            
            output_table = gr.Dataframe(label="Complete Results")
            
            predict_btn.click(
                fn=predict_normality,
                inputs=[st_teff, st_mass, st_rad, st_met, st_lum, st_age, st_vsin],
                outputs=[output_table, normality_output]
            )
        
        with gr.Tab("Batch Prediction from CSV"):
            gr.Markdown("### Upload a CSV file with stellar parameters")
            gr.Markdown("⚠️ **Required columns:** st_teff, st_mass, st_rad, st_met, st_lum, st_age, st_vsin")
            gr.Markdown("**Note:** Rows with missing values will be automatically dropped.")
            
            csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            csv_predict_btn = gr.Button("Predict from CSV", variant="primary")
            
            csv_status = gr.Textbox(label="Status", interactive=False)
            csv_output_table = gr.Dataframe(label="Results with Probability Scores")
            csv_download = gr.File(label="Download Results as CSV")
            
            csv_predict_btn.click(
                fn=predict_from_csv,
                inputs=[csv_input],
                outputs=[csv_output_table, csv_status, csv_download]
            )
    
    gr.Markdown("---")
    gr.Markdown("### 📊 Score Interpretation")
    gr.Markdown("""
    | Score Range | Interpretation | Description |
    |------------|----------------|-------------|
    | **0.6 - 1.0** | ✅ Normal stellar system | High probability of hosting exoplanets |
    | **0.4 - 0.6** | ⚠️ Moderately unusual | Moderate probability |
    | **0.0 - 0.4** | ❌ Highly anomalous | Low probability of hosting exoplanets |
    """)

if __name__ == "__main__":
    demo.launch()