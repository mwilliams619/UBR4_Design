from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import py3Dmol
from pathlib import Path
import json
import os
import tempfile
import subprocess
import asyncio
import re
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Set float32 matmul precision to 'medium' for better performance with Tensor Cores
torch.set_float32_matmul_precision('medium')
# Configure PyTorch Lightning logger
pl.seed_everything(42)
logger = TensorBoardLogger("lightning_logs", name="boltz_prediction")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a permanent output directory
BASE_OUTPUT_DIR = "protein_predictions"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

class ProteinSequence(BaseModel):
    sequence: str

@app.post("/predict")
async def predict_structure(protein: ProteinSequence):
    output_dir = None
    try:
        # Create timestamped directory for this prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"prediction_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Created output directory: {output_dir}")
        
        # Create input FASTA file
        input_file = "input.fasta"
        input_path = os.path.join(output_dir, input_file)
        with open(input_path, "w") as f:
            f.write(f">A|protein|empty\n{protein.sequence}")
        
        print(f"Created input file at: {input_path}")
        
        # Run Boltz prediction
        cmd = [
            "boltz", "predict", input_path,
            "--out_dir", output_dir,
            "--use_msa_server",
            "--output_format", "pdb",  # Changed back to PDB format
            "--recycling_steps", "3",
            "--sampling_steps", "200"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the prediction and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Initialize status messages
        status_messages = []
        
        # Monitor the process output
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                status_messages.append(output.strip())
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            raise Exception(f"Prediction failed with return code {rc}\n" + "\n".join(status_messages))

        # Updated path structure based on the actual output
        results_dir = os.path.join(output_dir, "boltz_results_input", "predictions", "input")
        if not os.path.exists(results_dir):
            raise Exception(f"Results directory not found at {results_dir}")
            
        print(f"Results directory contents: {os.listdir(results_dir)}")
        
        # Look for PDB file
        model_file = os.path.join(results_dir, "input_model_0.pdb")
        confidence_file = os.path.join(results_dir, "confidence_input_model_0.json")
        
        print(f"Looking for model file at: {model_file}")
        print(f"Looking for confidence file at: {confidence_file}")

        if not os.path.exists(model_file):
            raise Exception(f"Model file not found at {model_file}")
        if not os.path.exists(confidence_file):
            raise Exception(f"Confidence file not found at {confidence_file}")

        # Read the structure and confidence data
        with open(model_file, "r") as f:
            structure_lines = f.readlines()
            
        with open(confidence_file, "r") as f:
            confidence_data = json.load(f)
            print("\nConfidence data keys:", confidence_data.keys())
            
        # Get pLDDT scores from confidence data
        plddt_scores = confidence_data.get('plddt', [])
        if plddt_scores:
            print(f"\nFound {len(plddt_scores)} pLDDT scores")
            print(f"Score range: {min(plddt_scores):.2f} to {max(plddt_scores):.2f}")
        
        modified_structure_lines = []
        residue_index = 0

        for line in structure_lines:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == "CA":  # Only modify B-factors for CA atoms
                    if residue_index < len(plddt_scores):
                        plddt_score = plddt_scores[residue_index]
                        # PDB format: B-factor is columns 61-66
                        modified_line = f"{line[:60]}{plddt_score:6.2f}{line[66:]}"
                        modified_structure_lines.append(modified_line)
                        residue_index += 1
                    else:
                        modified_structure_lines.append(line)
                else:
                    modified_structure_lines.append(line)
            else:
                modified_structure_lines.append(line)
        
        # Join the modified structure lines
        modified_structure = ''.join(modified_structure_lines)
        
        # Print sample of modified structure
        print("\nSample of modified structure (first 500 characters):")
        print(modified_structure[:500])
        
        # Generate visualization with the modified structure
        view = create_3d_visualization(modified_structure)
        
        # Add path information to response
        return {
            "html": view,
            "confidence_scores": confidence_data,
            "status_messages": status_messages,
            "output_directory": output_dir,
            "model_file": model_file,
            "confidence_file": confidence_file
        }
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if output_dir and os.path.exists(output_dir):
            # Print directory structure for debugging
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")
        raise HTTPException(status_code=500, detail=str(e))

def create_3d_visualization(structure):
    structure_escaped = structure.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    
    viewer_html = f"""
    <div style="height: 600px; width: 100%; position: relative;" id='viewer_3dmol'>
        <div id="gldiv" style="width: 100%; height: 100%;"></div>
        <div class="plddt-legend" style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
            <div style="font-weight: bold; margin-bottom: 5px;">pLDDT Score</div>
            <div class="legend-item" style="display: flex; align-items: center; margin: 2px 0;">
                <div class="legend-color" style="width: 20px; height: 20px; background-color: #0857D3; margin-right: 5px;"></div>
                <span>Very high (90-100)</span>
            </div>
            <div class="legend-item" style="display: flex; align-items: center; margin: 2px 0;">
                <div class="legend-color" style="width: 20px; height: 20px; background-color: #6ACBF1; margin-right: 5px;"></div>
                <span>Confident (70-90)</span>
            </div>
            <div class="legend-item" style="display: flex; align-items: center; margin: 2px 0;">
                <div class="legend-color" style="width: 20px; height: 20px; background-color: #FED936; margin-right: 5px;"></div>
                <span>Low (50-70)</span>
            </div>
            <div class="legend-item" style="display: flex; align-items: center; margin: 2px 0;">
                <div class="legend-color" style="width: 20px; height: 20px; background-color: #FD7D4D; margin-right: 5px;"></div>
                <span>Very low (<50)</span>
            </div>
        </div>
    </div>
    <script>
        $(document).ready(function() {{
            let viewer = $3Dmol.createViewer('viewer_3dmol', {{
                backgroundColor: 'white'
            }});
            
            let model = viewer.addModel('{structure_escaped}', "pdb");  // Changed to PDB format
            
            // Set default cartoon representation
            viewer.setStyle({{}}, {{cartoon: {{}}}}); 
            
            // Color by B-factor (pLDDT scores)
            viewer.setStyle({{atom: "CA"}}, {{
                cartoon: {{
                    colorfunc: function(atom) {{
                        let score = atom.b;
                        if (score >= 90) return '#0857D3';      // Very high
                        else if (score >= 70) return '#6ACBF1'; // Confident
                        else if (score >= 50) return '#FED936'; // Low
                        else return '#FD7D4D';                  // Very low
                    }},
                    thickness: 0.4
                }}
            }});
            
            viewer.zoomTo();
            viewer.render();
            
            // Debug logging
            console.log('Model loaded');
            let atoms = model.selectedAtoms({{atom: "CA"}});
            console.log('Number of CA atoms:', atoms.length);
            if (atoms.length > 0) {{
                console.log('Sample B-factors:', atoms.slice(0,5).map(a => a.b));
            }}
        }});
    </script>
    """
    return viewer_html

@app.get("/")
async def root():
    return FileResponse('static/index.html')