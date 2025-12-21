#!/usr/bin/env python3
"""Simple PDF generator for the report"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re

def escape_html(text):
    """Escape HTML special characters"""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def process_markdown_inline(text):
    """Process inline markdown: bold and code"""
    # Escape first
    text = escape_html(text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Code - using simple formatting
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" color="red">\1</font>', text)
    # Checkmarks
    text = text.replace('✅', '<font color="green">✓</font>')
    text = text.replace('❌', '<font color="red">✗</font>')
    return text

def create_pdf():
    pdf_filename = 'hw3_report_1155249290.pdf'
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            rightMargin=60, leftMargin=60,
                            topMargin=60, bottomMargin=30)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, spaceAfter=20, textColor=colors.grey)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13, spaceAfter=10, spaceBefore=14, textColor=colors.HexColor('#2c3e50'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, spaceAfter=8, spaceBefore=10)
    body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9, leading=13, alignment=TA_JUSTIFY, spaceAfter=6)
    bullet_style = ParagraphStyle('Bullet', parent=styles['BodyText'], fontSize=9, leading=12, leftIndent=15, spaceAfter=3)
    
    # Title
    elements.append(Paragraph("ROSE 5710 Assignment 3 Report", title_style))
    elements.append(Paragraph("Student ID: 1155249290 | Date: December 21, 2025", subtitle_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Section 1
    elements.append(Paragraph("1. Why Create Multiple Target Configurations per Reference Joint Angle?", h1_style))
    elements.append(Paragraph("Creating multiple target configurations (8 perturbations) for each reference joint angle set serves several critical purposes:", body_style))
    elements.append(Paragraph("• <b>Data augmentation</b>: Increases dataset size from 5,000 to 40,000 samples without requiring additional FK computations", bullet_style))
    elements.append(Paragraph("• <b>Local workspace coverage</b>: Perturbing ±2° ensures dense sampling in neighborhoods, helping learn smooth mappings", bullet_style))
    elements.append(Paragraph("• <b>Robustness</b>: Network learns to handle small variations essential for real-world robot control", bullet_style))
    elements.append(Paragraph("• <b>Generalization</b>: Multiple nearby targets help interpolate between training samples rather than memorizing", bullet_style))
    
    # Section 2
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("2. Why Normalize Input Features?", h1_style))
    elements.append(Paragraph("<b>Why normalize:</b>", body_style))
    elements.append(Paragraph("• Features with different scales (angles in [-2,2] vs positions in [-100,100]) cause unbalanced gradients", bullet_style))
    elements.append(Paragraph("• Standardized features (mean=0, std=1) allow consistent optimizer step sizes across dimensions", bullet_style))
    elements.append(Paragraph("• Prevents numerical overflow/underflow in activations and weight updates", bullet_style))
    elements.append(Paragraph("<b>Without normalization:</b> Training would be extremely slow or fail; large-scale features would dominate the loss; risk of vanishing/exploding gradients increases.", body_style))
    
    # Section 3
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("3. Why Split Data into Train/Test Sets?", h1_style))
    elements.append(Paragraph("<b>Training set (80%, 32,000 samples)</b>: Used to compute loss and update weights via backpropagation", body_style))
    elements.append(Paragraph("<b>Test/Validation set (20%, 8,000 samples)</b>: Evaluates performance on unseen data; monitors generalization; triggers early stopping; provides unbiased real-world performance estimate", body_style))
    elements.append(Paragraph("This separation prevents falsely believing the model generalizes when it's merely memorizing training data.", body_style))
    
    # Section 4
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("4. Why Compute Normalization Scalers from Training Data Only?", h1_style))
    elements.append(Paragraph("Critical principle to prevent <b>data leakage</b>:", body_style))
    elements.append(Paragraph("• Test set represents future/unseen data the model has never encountered", bullet_style))
    elements.append(Paragraph("• Including test statistics in normalization indirectly gives the model information about the test set", bullet_style))
    elements.append(Paragraph("• In real deployment, we won't have access to future data statistics—only training statistics", bullet_style))
    elements.append(Paragraph("<b>Correct approach</b>: Compute scaler.fit() on training data only; apply scaler.transform() to both train and test.", body_style))
    
    # Section 5
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("5. Risk of Overfitting and Prevention Strategies", h1_style))
    elements.append(Paragraph("<b>Overfitting</b> occurs when a model learns training data too well, including noise, failing to generalize.", body_style))
    elements.append(Paragraph("<b>Our prevention strategies:</b>", body_style))
    elements.append(Paragraph("• <b>Early Stopping</b> (patience=50): Monitors validation loss, stops when plateaus, restores best weights", bullet_style))
    elements.append(Paragraph("• <b>Train/Test Split</b>: 20% held-out data provides continuous overfitting detection", bullet_style))
    elements.append(Paragraph("• <b>Model Size Constraint</b>: Limited to &lt;20,000 parameters (used 19,126)", bullet_style))
    elements.append(Paragraph("• <b>Large Dataset</b>: 40,000 diverse samples make memorization difficult", bullet_style))
    elements.append(Paragraph("<b>Result</b>: Training MAE (1.48°) ≈ Validation MAE (1.49°) shows minimal overfitting.", body_style))
    
    # Section 6
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("6. Network Architecture Design", h1_style))
    elements.append(Paragraph("<b>Architecture:</b>", body_style))
    elements.append(Paragraph("Input Layer: 13 features (6 reference joints + 4 quaternions + 3 positions)", bullet_style))
    elements.append(Paragraph("Hidden Layer 1: 128 neurons, ReLU activation", bullet_style))
    elements.append(Paragraph("Hidden Layer 2: 96 neurons, tanh activation", bullet_style))
    elements.append(Paragraph("Hidden Layer 3: 48 neurons, ReLU activation", bullet_style))
    elements.append(Paragraph("Output Layer: 6 neurons, linear activation (target joint angles)", bullet_style))
    elements.append(Paragraph("Total Parameters: 19,126 (&lt;20,000 requirement ✓)", bullet_style))
    elements.append(Paragraph("<b>Design rationale:</b>", body_style))
    elements.append(Paragraph("• <b>Funnel architecture</b> (128→96→48→6): Progressively compresses high-dimensional input into 6 angles", bullet_style))
    elements.append(Paragraph("• <b>First layer (128, ReLU)</b>: Large capacity for complex nonlinear IK relationships; ReLU for efficient gradients", bullet_style))
    elements.append(Paragraph("• <b>Second layer (96, tanh)</b>: Symmetric output [-1,1] helps with normalized data; provides activation diversity", bullet_style))
    elements.append(Paragraph("• <b>Third layer (48, ReLU)</b>: Further refinement with ReLU's sparsity", bullet_style))
    elements.append(Paragraph("• <b>Output layer (6, linear)</b>: No activation constraint—allows any continuous joint angle value", bullet_style))
    
    # Section 7
    elements.append(PageBreak())
    elements.append(Paragraph("7. Activation Functions: ReLU, tanh, and Linear", h1_style))
    elements.append(Paragraph("<b>ReLU</b> (f(x) = max(0, x)): Zero for negative inputs, linear for positive. Computationally efficient, alleviates vanishing gradient, induces sparsity. Used in hidden layers 1 & 3.", body_style))
    elements.append(Paragraph("<b>tanh</b> (f(x) = (e^x - e^-x)/(e^x + e^-x)): Output range [-1,1], symmetric, smooth. Zero-centered outputs better than sigmoid. Used in hidden layer 2 for activation diversity.", body_style))
    elements.append(Paragraph("<b>Linear</b> (f(x) = x): No transformation, direct pass-through. Used in output layer because:", body_style))
    elements.append(Paragraph("• Joint angles are continuous values, not bounded to [0,1] or [-1,1]", bullet_style))
    elements.append(Paragraph("• Need to predict any angle within joint limits (e.g., -120° to +90°)", bullet_style))
    elements.append(Paragraph("• Non-linear activation would artificially constrain predictions", bullet_style))
    elements.append(Paragraph("<b>Critical</b>: If we used ReLU, all negative angles would be clipped to zero; if tanh, outputs bounded to [-1,1], preventing accurate angle prediction.", body_style))
    
    # Section 8
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("8. Why MSE (Mean Squared Error) is Appropriate", h1_style))
    elements.append(Paragraph("MSE = (1/n) Σ(y_pred - y_true)²", body_style))
    elements.append(Paragraph("<b>Why MSE for this regression task:</b>", body_style))
    elements.append(Paragraph("• <b>Continuous output</b>: Joint angles are real-valued, not discrete classes", bullet_style))
    elements.append(Paragraph("• <b>Penalizes large errors</b>: Squaring amplifies big mistakes—5° error is 25× worse than 1°, crucial for robot safety", bullet_style))
    elements.append(Paragraph("• <b>Smooth gradients</b>: Differentiable everywhere, providing stable gradient descent", bullet_style))
    elements.append(Paragraph("• <b>Physical interpretation</b>: Squared error relates to energy/torque in robotics", bullet_style))
    elements.append(Paragraph("• <b>Statistical optimality</b>: MSE estimates the conditional mean", bullet_style))
    elements.append(Paragraph("<b>MAE as metric</b>: Used alongside MSE for interpretability. MAE=0.026 rad directly tells us average error is 1.49°.", body_style))
    
    # Section 9
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("9. Results and Analysis", h1_style))
    elements.append(Paragraph("9.a Final Performance Metrics", h2_style))
    
    # Table
    table_data = [
        ['Metric', 'Training', 'Validation'],
        ['Loss (MSE)', '0.001440', '0.001516'],
        ['MAE (radians)', '0.025904', '0.026074'],
        ['MAE (degrees)', '1.4842°', '1.4939°']
    ]
    t = Table(table_data, colWidths=[2*inch, 1.3*inch, 1.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Best epoch: 483 (out of 500)", body_style))
    elements.append(Paragraph("<b>Is this acceptable for a robot arm?</b>", body_style))
    elements.append(Paragraph("<font color='green'>✓</font> <b>Yes, highly acceptable:</b>", body_style))
    elements.append(Paragraph("• Meets requirement: 1.49° &lt; 2° threshold (0.026 &lt; 0.035 radians)", bullet_style))
    elements.append(Paragraph("• Practical accuracy: ±1.5° error is excellent for most manipulation tasks", bullet_style))
    elements.append(Paragraph("• For 50cm reach, 1.5° error translates to ~1.3cm positional error at end-effector", bullet_style))
    elements.append(Paragraph("• Validation MAE ≈ Training MAE indicates robust performance on unseen configurations", bullet_style))
    elements.append(Paragraph("• Well below typical robot joint encoder resolution (often 0.1-0.5°)", bullet_style))
    elements.append(Paragraph("<b>Context</b>: Industrial robots typically require &lt;5° for general tasks, &lt;2° for precision tasks. Our 1.49° places this model in the \"precision\" category.", body_style))
    
    # Section 9.b
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("9.b Future Improvements", h2_style))
    elements.append(Paragraph("If I were to improve this model further, I would try:", body_style))
    elements.append(Paragraph("1. <b>Residual connections</b>: Add skip connections for better gradient flow in deeper networks", bullet_style))
    elements.append(Paragraph("2. <b>Learning rate scheduling</b>: Start with higher LR (1e-3), decay to 1e-4 after plateau using ReduceLROnPlateau", bullet_style))
    elements.append(Paragraph("3. <b>Data augmentation enhancements</b>: Add Gaussian noise to simulate sensor errors; vary perturbation range", bullet_style))
    elements.append(Paragraph("4. <b>Ensemble methods</b>: Train 5 models with different seeds, average predictions (typical 10-15% improvement)", bullet_style))
    elements.append(Paragraph("5. <b>Architecture search</b>: Try [96→80→64] funnel; experiment with dropout (0.1-0.2); test Batch Normalization", bullet_style))
    elements.append(Paragraph("6. <b>Physics-informed loss</b>: Add FK constraint: penalize when FK(predicted_joints) ≠ target_pose", bullet_style))
    elements.append(Paragraph("7. <b>Attention mechanisms</b>: Learn which input features are most important for each output joint", bullet_style))
    
    # Summary
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("Summary", h1_style))
    elements.append(Paragraph("This neural network-based inverse kinematics solver successfully achieves:", body_style))
    elements.append(Paragraph("<font color='green'>✓</font> Validation MAE: <b>1.49°</b> (requirement: &lt;2°)", bullet_style))
    elements.append(Paragraph("<font color='green'>✓</font> Model parameters: <b>19,126</b> (requirement: &lt;20,000)", bullet_style))
    elements.append(Paragraph("<font color='green'>✓</font> Strong generalization with minimal overfitting", bullet_style))
    elements.append(Paragraph("<font color='green'>✓</font> Production-ready accuracy for robotic manipulation tasks", bullet_style))
    elements.append(Paragraph("The combination of careful data generation (40k samples with local perturbations), proper normalization, balanced architecture design, and overfitting prevention strategies resulted in a compact yet accurate IK solver suitable for real-time robot control.", body_style))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ PDF report generated: {pdf_filename}")
    import os
    print(f"✓ File size: {os.path.getsize(pdf_filename) / 1024:.1f} KB")

if __name__ == '__main__':
    create_pdf()
