import os
import io
import torch
import numpy as np
import nibabel as nib
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# =====================
# DATABASE MODELS
# =====================

class Patient(db.Model):
    __tablename__ = 'patients'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    age         = db.Column(db.Integer, nullable=False)
    gender      = db.Column(db.String(10), nullable=False)
    mobile      = db.Column(db.String(15), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    scans       = db.relationship('Scan', backref='patient', lazy=True)

class Scan(db.Model):
    __tablename__ = 'scans'
    id          = db.Column(db.Integer, primary_key=True)
    patient_id  = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    scan_date   = db.Column(db.DateTime, default=datetime.utcnow)
    results     = db.relationship('Result', backref='scan', lazy=True)

class Result(db.Model):
    __tablename__ = 'results'
    id               = db.Column(db.Integer, primary_key=True)
    scan_id          = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False)
    dice_wt          = db.Column(db.Float)
    dice_tc          = db.Column(db.Float)
    dice_et          = db.Column(db.Float)
    average_dice     = db.Column(db.Float)
    tumor_volume     = db.Column(db.Float)
    severity_level   = db.Column(db.Integer)
    severity_label   = db.Column(db.String(20))
    prediction_image = db.Column(db.String(200))
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)

# =====================
# U-NET MODEL
# =====================

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()
        self.encoder1   = ConvBlock(in_channels, 64)
        self.encoder2   = ConvBlock(64, 128)
        self.encoder3   = ConvBlock(128, 256)
        self.encoder4   = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.upconv4    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4   = ConvBlock(1024, 512)
        self.upconv3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3   = ConvBlock(512, 256)
        self.upconv2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2   = ConvBlock(256, 128)
        self.upconv1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1   = ConvBlock(128, 64)
        self.out        = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.sigmoid(self.out(d1))

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = UNet(in_channels=4, out_channels=3).to(device)
checkpoint = torch.load(app.config['MODEL_PATH'], map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded on {device}")

# =====================
# HELPER FUNCTIONS
# =====================

def normalize(img):
    if img.max() == 0:
        return img
    return (img - img.min()) / (img.max() - img.min())

def dice_score(pred, gt):
    intersection = (pred * gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return float(2 * intersection) / float(pred.sum() + gt.sum())

def get_severity(tumor_volume, et_ratio):
    if tumor_volume < 5 or et_ratio < 0.20:
        return 1, 'Low Risk'
    elif tumor_volume < 20 or et_ratio < 0.40:
        return 2, 'Moderate Risk'
    else:
        return 3, 'High Risk'

def find_seg_file(flair_path):
    seg_path = flair_path.replace('_flair.nii.gz', '_seg.nii.gz')
    if os.path.exists(seg_path):
        return seg_path
    patient_id = os.path.basename(flair_path).replace('_flair.nii.gz', '')
    seg_path   = os.path.join(BASE_DIR, 'data', 'raw', patient_id, f'{patient_id}_seg.nii.gz')
    if os.path.exists(seg_path):
        return seg_path
    return None

def run_prediction(flair_path, t1_path, t1ce_path, t2_path):
    flair = nib.load(flair_path).get_fdata()
    t1    = nib.load(t1_path).get_fdata()
    t1ce  = nib.load(t1ce_path).get_fdata()
    t2    = nib.load(t2_path).get_fdata()

    combined   = flair + t1ce
    slice_sums = combined.sum(axis=(0, 1))
    best_slice = np.argmax(slice_sums)

    flair_s = normalize(flair[:, :, best_slice])
    t1_s    = normalize(t1[:, :, best_slice])
    t1ce_s  = normalize(t1ce[:, :, best_slice])
    t2_s    = normalize(t2[:, :, best_slice])

    input_tensor = np.stack([flair_s, t1_s, t1ce_s, t2_s], axis=0)
    input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    pred_np = prediction.squeeze(0).cpu().numpy()
    pred_wt = (pred_np[0] > 0.5).astype(np.uint8)
    pred_tc = (pred_np[1] > 0.5).astype(np.uint8)
    pred_et = (pred_np[2] > 0.5).astype(np.uint8)

    tumor_volume = float(pred_wt.sum() * 1.0 / 1000)
    et_ratio     = float(pred_et.sum() / (pred_wt.sum() + 1e-6))
    severity_level, severity_label = get_severity(tumor_volume, et_ratio)

    seg_path = find_seg_file(flair_path)
    d_wt = d_tc = d_et = d_avg = None
    if seg_path:
        seg   = nib.load(seg_path).get_fdata()
        gt_wt = (seg[:, :, best_slice] > 0).astype(np.uint8)
        gt_tc = (np.isin(seg[:, :, best_slice], [1, 4])).astype(np.uint8)
        gt_et = (seg[:, :, best_slice] == 4).astype(np.uint8)
        d_wt  = round(dice_score(pred_wt, gt_wt) * 100, 2)
        d_tc  = round(dice_score(pred_tc, gt_tc) * 100, 2)
        d_et  = round(dice_score(pred_et, gt_et) * 100, 2)
        d_avg = round((d_wt + d_tc + d_et) / 3, 2)
        print(f"Dice scores — WT: {d_wt}, TC: {d_tc}, ET: {d_et}, Avg: {d_avg}")
    else:
        print("Segmentation file not found — dice score skipped")

    return {
        'pred_wt'        : pred_wt,
        'pred_tc'        : pred_tc,
        'pred_et'        : pred_et,
        'flair_slice'    : flair_s,
        'best_slice'     : int(best_slice),
        'tumor_volume'   : round(tumor_volume, 3),
        'et_ratio'       : round(et_ratio, 3),
        'severity_level' : severity_level,
        'severity_label' : severity_label,
        'dice_wt'        : d_wt,
        'dice_tc'        : d_tc,
        'dice_et'        : d_et,
        'average_dice'   : d_avg,
    }

def save_result_image(result_data, filename):
    bg      = '#1a1a2e'
    flair_s = result_data['flair_slice']
    pred_wt = result_data['pred_wt']
    pred_tc = result_data['pred_tc']
    pred_et = result_data['pred_et']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=bg)
    titles    = ['FLAIR', 'Whole Tumor', 'Tumor Core', 'Enhancing Tumor']

    axes[0].set_facecolor(bg)
    axes[0].imshow(flair_s, cmap='gray')
    axes[0].set_title(titles[0], color='white', fontsize=11)
    axes[0].axis('off')

    axes[1].set_facecolor(bg)
    axes[1].imshow(flair_s, cmap='gray')
    axes[1].imshow(pred_wt, cmap='Reds', alpha=0.5)
    axes[1].set_title(titles[1], color='white', fontsize=11)
    axes[1].axis('off')

    axes[2].set_facecolor(bg)
    axes[2].imshow(flair_s, cmap='gray')
    axes[2].imshow(pred_tc, cmap='Blues', alpha=0.5)
    axes[2].set_title(titles[2], color='white', fontsize=11)
    axes[2].axis('off')

    axes[3].set_facecolor(bg)
    axes[3].imshow(flair_s, cmap='gray')
    axes[3].imshow(pred_et, cmap='Greens', alpha=0.5)
    axes[3].set_title(titles[3], color='white', fontsize=11)
    axes[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=bg)
    plt.close()
    return filename

# =====================
# ROUTES
# =====================

@app.route('/')
def index():
    total_patients = Patient.query.count()
    total_scans    = Scan.query.count()
    high_risk      = Result.query.filter_by(severity_level=3).count()
    stats = {
        'total_patients' : total_patients,
        'total_scans'    : total_scans,
        'high_risk'      : high_risk,
    }
    return render_template('index.html', stats=stats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        patient = Patient(
            name   = request.form['name'],
            age    = request.form['age'],
            gender = request.form['gender'],
            mobile = request.form['mobile'],
        )
        db.session.add(patient)
        db.session.commit()
        return redirect(url_for('upload', patient_id=patient.id))
    return render_template('register.html')

@app.route('/upload/<int:patient_id>', methods=['GET', 'POST'])
def upload(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if request.method == 'POST':
        files     = request.files
        scan      = Scan(patient_id=patient_id)
        db.session.add(scan)
        db.session.commit()

        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(scan.id))
        os.makedirs(upload_dir, exist_ok=True)

        paths = {}
        for key in ['flair', 't1', 't1ce', 't2']:
            f         = files[key]
            save_path = os.path.join(upload_dir, f.filename)
            f.save(save_path)
            paths[key] = save_path

        result_data  = run_prediction(paths['flair'], paths['t1'], paths['t1ce'], paths['t2'])
        img_filename = f"result_{scan.id}.png"
        save_result_image(result_data, img_filename)

        result = Result(
            scan_id          = scan.id,
            tumor_volume     = result_data['tumor_volume'],
            severity_level   = result_data['severity_level'],
            severity_label   = result_data['severity_label'],
            prediction_image = img_filename,
            dice_wt          = result_data['dice_wt'],
            dice_tc          = result_data['dice_tc'],
            dice_et          = result_data['dice_et'],
            average_dice     = result_data['average_dice'],
        )
        db.session.add(result)
        db.session.commit()

        return redirect(url_for('results', result_id=result.id))

    return render_template('upload.html', patient=patient)

@app.route('/results/<int:result_id>')
def results(result_id):
    result  = Result.query.get_or_404(result_id)
    scan    = result.scan
    patient = scan.patient
    return render_template('results.html', result=result, scan=scan, patient=patient)

@app.route('/download_report/<int:result_id>')
def download_report(result_id):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    result  = Result.query.get_or_404(result_id)
    scan    = result.scan
    patient = scan.patient

    buffer   = io.BytesIO()
    doc      = SimpleDocTemplate(buffer, pagesize=letter)
    styles   = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('title', fontSize=20, textColor=colors.HexColor('#0099cc'),
                                  spaceAfter=10, fontName='Helvetica-Bold', alignment=1)
    elements.append(Paragraph("NeuroScan AI - Brain Tumor Analysis Report", title_style))
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph("Patient Information", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    patient_data = [
        ['Patient Name', patient.name],
        ['Age',          str(patient.age)],
        ['Gender',       patient.gender],
        ['Mobile',       patient.mobile],
        ['Scan Date',    scan.scan_date.strftime('%d %B %Y, %I:%M %p')],
        ['Report ID',    f'RPT-{result.id:04d}'],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (0,-1), colors.HexColor('#e8f4fd')),
        ('TEXTCOLOR',      (0,0), (0,-1), colors.HexColor('#0099cc')),
        ('FONTNAME',       (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING',        (0,0), (-1,-1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph("Analysis Results", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    results_data = [
        ['Tumor Volume',    f"{result.tumor_volume} cm³"],
        ['Severity Level',  f"Level {result.severity_level}"],
        ['Risk Assessment', result.severity_label],
        ['Detection Method','AI - U-Net Deep Learning'],
        ['Model Accuracy',  '82.02% (Dice Score)'],
    ]
    if result.dice_wt:
        results_data += [
            ['Dice WT',      f"{result.dice_wt}%"],
            ['Dice TC',      f"{result.dice_tc}%"],
            ['Dice ET',      f"{result.dice_et}%"],
            ['Average Dice', f"{result.average_dice}%"],
        ]
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (0,-1), colors.HexColor('#e8f4fd')),
        ('TEXTCOLOR',      (0,0), (0,-1), colors.HexColor('#0099cc')),
        ('FONTNAME',       (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING',        (0,0), (-1,-1), 8),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 0.3*inch))

    img_path = os.path.join(app.config['RESULTS_FOLDER'], result.prediction_image)
    if os.path.exists(img_path):
        elements.append(Paragraph("Segmentation Results", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Image(img_path, width=6.5*inch, height=1.8*inch))
        elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph("Clinical Recommendations", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    if result.severity_level == 1:
        rec = "Early stage tumor detected. Regular monitoring every 3-6 months recommended. Consult neurologist for follow-up."
    elif result.severity_level == 2:
        rec = "Intermediate stage tumor detected. Medical consultation advised within 2 weeks. Consider MRI follow-up in 1-3 months."
    else:
        rec = "Advanced stage tumor detected. Immediate medical attention required. Urgent referral to neurosurgeon recommended."
    elements.append(Paragraph(rec, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    footer_style = ParagraphStyle('footer', fontSize=9, textColor=colors.grey, alignment=1)
    elements.append(Paragraph("This report is generated by NeuroScan AI and is for reference purposes only.", footer_style))
    elements.append(Paragraph("Please consult a qualified medical professional for diagnosis and treatment.", footer_style))

    doc.build(elements)
    buffer.seek(0)
    filename = f"NeuroScan_Report_{patient.name}_{result.id}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

# =====================
# DELETE PATIENT ROUTE  
# =====================

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # Pehle saare results aur unki image files delete karo
    for scan in patient.scans:
        for result in scan.results:
            if result.prediction_image:
                img_path = os.path.join(app.config['RESULTS_FOLDER'], result.prediction_image)
                if os.path.exists(img_path):
                    os.remove(img_path)
            db.session.delete(result)
        db.session.delete(scan)
    
    # Ab patient delete karo
    db.session.delete(patient)
    db.session.commit()
    
    return redirect(url_for('history'))

@app.route('/history')
def history():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template('history.html', patients=patients)

# =====================
# MAIN
# =====================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created")
    app.run(debug=True, port=5000)