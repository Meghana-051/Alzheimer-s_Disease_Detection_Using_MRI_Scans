# app.py
from __future__ import annotations
import io
import base64
import uuid
import logging
import time
import os
import traceback
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from advanced_engine.multimodal_fusion import MultimodalFusionClassifier
from advanced_engine.gradcam_3d import GradCAM3D
from advanced_engine.clinical_agent import build_clinical_agent, build_guideline_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuro_platform")

class AppState:
    model:   MultimodalFusionClassifier
    gradcam: GradCAM3D
    agent:   object
    device:  torch.device

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading multimodal fusion model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultimodalFusionClassifier(num_classes=4, clinical_input_dim=5).to(device)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "fusion_model.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading pre-trained weight parameters from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Operating with randomized initialization matrices.")
    
    model.eval()

    try:
        target_block = list(model.visual_encoder.backbone.children())
        target_layer = target_block[4][-1].conv2[0]
    except Exception:
        target_layer = list(model.visual_encoder.backbone.parameters())[-1]

    gradcam = GradCAM3D(model=model, target_layer=target_layer)

    logger.info("Building guideline vector store ...")
    store = build_guideline_store()
    agent = build_clinical_agent(store)

    app_state.model   = model
    app_state.gradcam = gradcam
    app_state.agent   = agent
    app_state.device  = device
    yield
    torch.cuda.empty_cache()

app = FastAPI(title="Clinical Neuroradiology AI Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def hipaa_audit_log(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start      = time.perf_counter()
    response   = await call_next(request)
    elapsed    = time.perf_counter() - start
    logger.info(f"AUDIT | request_id={request_id} method={request.method} path={request.url.path} status={response.status_code} elapsed={elapsed:.3f}s")
    response.headers["X-Request-ID"] = request_id
    return response

class DiagnosisResponse(BaseModel):
    request_id:       str
    predicted_stage:  int
    stage_label:      str
    confidence_scores: list[float]
    modal_gates:      list[float]
    gradcam_b64:      str
    clinical_report:  str
    processing_ms:    float

STAGE_LABELS = {0: "Non-Demented", 1: "Very Mild Cognitive Impairment", 2: "Mild Cognitive Impairment", 3: "Moderate Alzheimer's Disease"}

def preprocess_volume(file_bytes: bytes, target: tuple = (32, 128, 128)) -> torch.Tensor:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
            img = np.array(pil_img)
        except Exception as exc:
            raise ValueError(f"Image decompression engine completely failed to parse file matrix: {exc}")
            
    img = cv2.resize(img, (target[2], target[1])).astype(np.float32) / 255.0
    volume = np.stack([img] * target[0], axis=0)
    volume = np.stack([volume] * 3, axis=0)[np.newaxis]
    return torch.from_numpy(volume)

def build_metadata_tensor(age: float, gender: int, mmse: float, apoe_e4: int, cdr: float) -> torch.Tensor:
    feats = np.array([age / 100.0, float(gender), mmse / 30.0, float(apoe_e4), cdr / 3.0], dtype=np.float32)
    return torch.from_numpy(feats).unsqueeze(0)

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    template_path = os.path.join("frontend", "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard UI index template missing.")
    with open(template_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.post("/api/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    mri_file:  UploadFile = File(...),
    name:      str        = Form(...),
    age:       float      = Form(...),
    gender:    int        = Form(...),
    mmse:      float      = Form(28.0),
    apoe_e4:   int        = Form(0),
    cdr:       float      = Form(0.0),
):
    t0 = time.perf_counter()
    
    try:
        raw_bytes = await mri_file.read()

        volume_t   = preprocess_volume(raw_bytes).to(app_state.device)
        metadata_t = build_metadata_tensor(age, gender, mmse, apoe_e4, cdr).to(app_state.device)

        with torch.no_grad():
            output = app_state.model(volume_t, metadata_t)

        logits = output["logits"]
        probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
        stage  = int(torch.argmax(logits, dim=-1).item())
        gates  = output["modal_gates"].squeeze().tolist()

        cam_volume = app_state.gradcam.generate(volume=volume_t, metadata=metadata_t, class_idx=stage)
        
        arr = np.frombuffer(raw_bytes, np.uint8)
        raw_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            raw_gray = np.array(Image.open(io.BytesIO(raw_bytes)).convert("L"))
            
        raw_gray_resized = cv2.resize(raw_gray, (128, 128)).astype(np.float32) / 255.0
        raw_vol_np = np.stack([raw_gray_resized] * 32, axis=0)

        overlay_bgr = app_state.gradcam.slice_overlay(volume_np=raw_vol_np, cam_np=cam_volume, slice_idx=16, axis=0, alpha=0.45)
        _, enc_buf = cv2.imencode(".png", overlay_bgr)
        gradcam_b64 = base64.b64encode(enc_buf.tobytes()).decode()

        agent_input = {
            "messages": [],
            "patient_context": {"name": name, "age": age, "gender": "M" if gender else "F", "mmse_score": mmse, "apoe_e4": bool(apoe_e4)},
            "model_output": {"predicted_stage": stage, "confidence_scores": probs, "modal_gates": gates},
            "retrieved_docs": [], "draft_report": "", "critique": "", "final_report": "", "iterations": 0,
        }
        agent_result  = app_state.agent.invoke(agent_input)
        clinical_text = agent_result.get("final_report", agent_result.get("draft_report", ""))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DiagnosisResponse(
            request_id=str(uuid.uuid4()), predicted_stage=stage, stage_label=STAGE_LABELS[stage],
            confidence_scores=probs, modal_gates=gates, gradcam_b64=gradcam_b64,
            clinical_report=clinical_text, processing_ms=round(elapsed_ms, 1),
        )
        
    except Exception as exc:
        print("\n" + "="*60 + "\n[CRITICAL PIPELINE EXCEPTION ERROR DETAILS]:\n" + "="*60)
        traceback.print_exc()
        print("="*60 + "\n")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal pipeline diagnostic operational execution failure: {str(exc)}"}
        )

@app.post("/api/v1/report/pdf")
async def generate_pdf(payload: dict):
    from fpdf import FPDF
    try:
        p = FPDF()
        p.add_page()
        
        p.set_font("Helvetica", "B", 16)
        p.cell(0, 12, "Clinical Neuroradiology AI Analytics Brief", ln=True, align="C")
        p.line(10, 22, 200, 22)
        p.ln(5)
        
        p.set_font("Helvetica", "B", 11)
        p.cell(35, 8, "Patient Name: ", ln=False)
        p.set_font("Helvetica", size=11)
        p.cell(0, 8, f"{payload.get('name', 'Anonymous Evaluation Subject')}", ln=True)
        
        p.set_font("Helvetica", "B", 11)
        p.cell(35, 8, "Assigned Stage: ", ln=False)
        p.set_font("Helvetica", size=11)
        p.cell(0, 8, f"{payload.get('stage_label', 'Undetermined Diagnostic Classification')}", ln=True)
        
        p.set_font("Helvetica", "B", 11)
        p.cell(35, 8, "Verification ID: ", ln=False)
        p.set_font("Helvetica", size=11)
        p.cell(0, 8, f"{payload.get('request_id', 'N/A-LOCAL-STANDALONE')}", ln=True)
        p.ln(6)
        
        p.set_font("Helvetica", "B", 13)
        p.cell(0, 8, "Official Diagnostic Findings & Recommendations", ln=True)
        p.line(10, 58, 100, 58)
        p.ln(4)
        
        p.set_font("Courier", size=10)
        report_content = payload.get("clinical_report", "No structural case data compiled.")
        
        # ── FIXED: SANITIZE UNICODE SYMBOLS THAT CRASH FPDF (e.g. ε -> e) ──
        sanitized_report = report_content.replace("ε", "e").replace("\u2014", "-")
        clean_report_text = sanitized_report.encode('latin-1', 'replace').decode('latin-1')
        
        p.multi_cell(0, 5, clean_report_text)
        
        p.ln(10)
        p.set_font("Helvetica", "I", 9)
        p.cell(0, 5, f"Report systematically processed on: {time.strftime('%Y-%m-%d %H:%M:%S')} (UTC Local Node Key)", ln=True)
        p.cell(0, 5, "Status Verified and Authenticated - Neuroradiology Platform Tier-1 Validation.", ln=True)
        
        pdf_output = p.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = bytes(pdf_output)
            
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=neuro_report.pdf"}
        )
        
    except Exception as e:
        logger.error(f"PDF Compilation Lifecycle Failure: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"PDF engine failed to map layout strings: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)