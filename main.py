"""
FastAPI Main Application - AI-Powered Metadata Tagging System
Entry point for the complete processing pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import logging
import os
import shutil
from pathlib import Path
import asyncio
import json
from utils.file_cleanup import cleanup_all_files, cleanup_file, schedule_delayed_cleanup, get_cleanup_info
from datetime import datetime
from typing import Dict, Any, Optional
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse


# Import backend modules
from backend.api.routes import router as api_router
from backend.core.file_handler import main as process_files
from backend.core.nlp_validator import process_all_scripts
from backend.core.llm_processor import main as process_llm
from backend.core.visualization_engine import create_visualization_data
from ml.cross_validator import create_cross_validation_data
from config import settings


logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Metadata Tagging System",
    description="Dual AI validation system for script analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes  
app.include_router(api_router, prefix="/api/v1")

@app.post("/api/process-uploaded")
async def process_uploaded_files(background_tasks: BackgroundTasks):
    global processing_status
    
    if processing_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Already processing")
    
    uploads_dir = Path("data/uploads")
    txt_files = list(uploads_dir.glob("*.txt"))
    
    if not txt_files:
        raise HTTPException(status_code=400, detail="No files found")
    
    processing_status.update({
        "is_processing": True,
        "current_step": "Starting",
        "progress": 5,
        "start_time": datetime.now().isoformat()
    })
    
    background_tasks.add_task(run_pipeline)
    return {"status": "started"}

app.mount("/css", StaticFiles(directory="frontend/CSS"), name="css")
app.mount("/js", StaticFiles(directory="frontend/JS"), name="js")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")

async def run_pipeline():
    global processing_status
    try:
        processing_status.update({"current_step": "Preprocessing", "progress": 25})
        await asyncio.to_thread(process_files)
        
        processing_status.update({"current_step": "NLP", "progress": 50})
        await asyncio.to_thread(process_all_scripts)
        
        processing_status.update({"current_step": "LLM", "progress": 75})
        await asyncio.to_thread(process_llm)
        
        processing_status.update({"is_processing": False, "progress": 100})
    except Exception as e:
        processing_status.update({"is_processing": False, "error": str(e)})

# Global processing status
processing_status = {
    "is_processing": False,
    "current_step": "",
    "progress": 0,
    "error": None,
    "start_time": None,
    "files_uploaded": []
}

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload transcript file and trigger processing pipeline"""
    
    global processing_status
    
    # Check if processing is already running
    if processing_status["is_processing"]:
        raise HTTPException(
            status_code=400, 
            detail="Processing already in progress. Please wait."
        )
    
    # Validate file
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    try:
        # Save uploaded file
        upload_path = Path("data/uploads") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update processing status
        processing_status.update({
            "is_processing": True,
            "current_step": "File uploaded",
            "progress": 10,
            "error": None,
            "start_time": datetime.now().isoformat(),
            "files_uploaded": [file.filename]
        })
        
        # Start background processing
        background_tasks.add_task(run_complete_pipeline, file.filename)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "status": "processing_started",
            "processing_id": processing_status["start_time"]
        }
        
    except Exception as e:
        processing_status.update({
            "is_processing": False,
            "error": str(e)
        })
        if 'upload_path' in locals() and upload_path.exists():
            cleanup_file(file.filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/status")
async def get_processing_status():
    """Get current processing status"""
    return processing_status

@app.get("/api/results/{filename}")
async def get_results(filename: str):
    """Get processing results for a specific file"""
    
    base_name = Path(filename).stem
    
    # Check for all result files
    result_files = {
        "llm_analysis": f"data/processed/processed_llm_analyzer_jsons/analyzed_{base_name}.json",
        "nlp_analysis": f"data/processed/processed_nlp_validator_jsons/{base_name}_analysis.json",
        "visualization_data": "data/processed/visualization_engine/visualization_data.json",
        "cross_validation": "data/processed/cross_validator/cross_validation_output.json"
    }
    
    results = {}
    
    for result_type, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[result_type] = json.load(f)
            except Exception as e:
                results[result_type] = {"error": f"Failed to load {result_type}: {str(e)}"}
        else:
            results[result_type] = {"error": f"File not found: {file_path}"}
    
    return results

@app.post("/api/export-pdf")
async def export_pdf(request: Request):
    """Generate PDF report from dashboard data"""
    try:
        from backend.core.pdf_generator import generate_pdf_report
        
        # Get JSON data from request
        data = await request.json()
        
        # Generate PDF
        pdf_path = generate_pdf_report(data)
        
        # Return PDF file
        return StreamingResponse(
            open(pdf_path, "rb"), 
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=metadata-report.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get("/api/files")
async def list_processed_files():
    """List all processed files"""
    
    uploads_dir = Path("data/uploads")
    processed_files = []
    
    if uploads_dir.exists():
        for file in uploads_dir.glob("*.txt"):
            base_name = file.stem
            
            # Check processing status
            llm_result = Path(f"data/processed/processed_llm_analyzer_jsons/analyzed_{base_name}.json")
            nlp_result = Path(f"data/processed/processed_nlp_validator_jsons/{base_name}_analysis.json")
            
            processed_files.append({
                "filename": file.name,
                "upload_time": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                "llm_processed": llm_result.exists(),
                "nlp_processed": nlp_result.exists(),
                "fully_processed": llm_result.exists() and nlp_result.exists()
            })
    
    return {"files": processed_files}




async def run_complete_pipeline(filename: str):
    """Run the complete processing pipeline for a single file"""
    
    global processing_status
    
    try:
        print(f"🔄 Starting pipeline for: {filename}")
        base_name = Path(filename).stem
        
        # Step 1: File preprocessing
        processing_status.update({
            "current_step": "Preprocessing files",
            "progress": 20
        })
        await asyncio.to_thread(process_files)
        
        # Step 2: NLP Analysis
        processing_status.update({
            "current_step": "Running NLP analysis",
            "progress": 40
        })
        await asyncio.to_thread(run_nlp_analysis)
        
        # Step 3: LLM Analysis
        processing_status.update({
            "current_step": "Running LLM analysis",
            "progress": 60
        })
        await asyncio.to_thread(run_llm_analysis)
        
        # Step 4: Cross Validation
        processing_status.update({
            "current_step": "Cross validation",
            "progress": 80
        })
        await asyncio.to_thread(run_cross_validation)
        
        # Step 5: Generate Visualization Data
        processing_status.update({
            "current_step": "Generating visualizations",
            "progress": 90
        })
        await asyncio.to_thread(generate_visualization_data)
        
        # Complete
        processing_status.update({
            "is_processing": False,
            "current_step": "Processing complete",
            "progress": 100,
            "error": None
        })
        
        print(f"✅ Complete pipeline finished for {filename}")
        schedule_delayed_cleanup(delay_minutes=7)
        
    except Exception as e:
        processing_status.update({
            "is_processing": False,
            "current_step": "Processing failed",
            "error": str(e),
            "progress": 0
        })
        print(f"❌ Pipeline failed for {filename}: {str(e)}")
        cleanup_file(filename)
        
    except Exception as e:
        processing_status.update({
            "is_processing": False,
            "current_step": "Processing failed",
            "error": str(e),
            "progress": 0
        })
        print(f"❌ Pipeline failed: {str(e)}")
        
        # Clean up files immediately on error (no delay needed)
        cleanup_all_files()
        
@app.post("/api/process-uploaded")
async def process_uploaded_files(background_tasks: BackgroundTasks):
    global processing_status
    
    if processing_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    uploads_dir = Path("data/uploads")
    txt_files = list(uploads_dir.glob("*.txt"))
    
    if not txt_files:
        raise HTTPException(status_code=404, detail="No .txt files found")
    
    processing_status.update({
        "is_processing": True,
        "current_step": "Starting batch processing",
        "progress": 5,
        "start_time": datetime.now().isoformat(),
        "files_uploaded": [f.name for f in txt_files]
    })
    
    background_tasks.add_task(run_complete_pipeline_batch, [f.name for f in txt_files])
    
    return {
        "message": f"Started processing {len(txt_files)} files",
        "files": [f.name for f in txt_files]
    }

async def run_complete_pipeline_batch(filenames: List[str]):
    global processing_status
    
    try:
        processing_status.update({"current_step": "Preprocessing files", "progress": 20})
        await asyncio.to_thread(process_files)
        
        processing_status.update({"current_step": "Running NLP analysis", "progress": 40})
        await asyncio.to_thread(run_nlp_analysis)
        
        processing_status.update({"current_step": "Running LLM analysis", "progress": 60})
        await asyncio.to_thread(run_llm_analysis)
        
        processing_status.update({"current_step": "Cross validation", "progress": 80})
        await asyncio.to_thread(run_cross_validation)
        
        processing_status.update({"current_step": "Generating visualizations", "progress": 90})
        await asyncio.to_thread(generate_visualization_data)
        
        processing_status.update({
                    "is_processing": False,
                    "current_step": "Complete",
                    "progress": 100
                })
                
                # Schedule cleanup after batch processing (7 minutes for multiple files)
        schedule_delayed_cleanup(delay_minutes=7)
        
    except Exception as e:
            processing_status.update({
                "is_processing": False,
                "current_step": "Batch processing failed",
                "error": str(e),
                "progress": 0
            })
            print(f"❌ Batch processing failed: {str(e)}")
            cleanup_all_files()

def run_nlp_analysis():
    """Run NLP analysis step"""
    try:
        print("🔄 Starting NLP analysis...")
        
        # Import and run NLP processor
        from backend.core.nlp_validator import process_all_scripts
        process_all_scripts()
        
        print("✅ NLP analysis completed")
    except Exception as e:
        print(f"❌ NLP analysis failed: {str(e)}")
        raise e

def run_llm_analysis():
    """Run LLM analysis step"""
    try:
        print("🔄 Starting LLM analysis...")
        
        # Import and run LLM processor
        from backend.core.llm_processor import main as llm_main
        llm_main()
        
        print("✅ LLM analysis completed")
    except Exception as e:
        print(f"❌ LLM analysis failed: {str(e)}")
        raise e

def run_cross_validation():
    """Run cross validation step"""
    try:
        print("🔄 Starting cross validation...")
        
        llm_dir = "data/processed/processed_llm_analyzer_jsons"
        nlp_dir = "data/processed/processed_nlp_validator_jsons"
        
        import glob
        
        def get_latest_file(folder):
            files = glob.glob(os.path.join(folder, "*.json"))
            if not files:
                return None
            return max(files, key=os.path.getctime)
        
        llm_file = get_latest_file(llm_dir)
        nlp_file = get_latest_file(nlp_dir)
        
        if llm_file and nlp_file:
            # Use the imported function directly
            cross_val_data = create_cross_validation_data(llm_file, nlp_file)
            
            if "error" in cross_val_data:
                raise Exception(cross_val_data["error"])
            
            # Save results
            output_path = "data/processed/cross_validator/cross_validation_output.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cross_val_data, f, indent=2, ensure_ascii=False)
            
            print("✅ Cross validation completed with real data")
        else:
            raise Exception("Could not find LLM or NLP analysis files")
            
    except Exception as e:
        print(f"❌ Cross validation failed: {str(e)}")
        raise e

def generate_visualization_data():
    """Generate visualization data step"""
    try:
        print("🔄 Generating visualization data...")
        
        # Get latest files
        llm_dir = "data/processed/processed_llm_analyzer_jsons"
        nlp_dir = "data/processed/processed_nlp_validator_jsons"
        
        import glob
        
        def get_latest_file(folder):
            files = glob.glob(os.path.join(folder, "*.json"))
            if not files:
                return None
            return max(files, key=os.path.getctime)
        
        llm_file = get_latest_file(llm_dir)
        nlp_file = get_latest_file(nlp_dir)
        
        if llm_file and nlp_file:
            viz_data = create_visualization_data(llm_file, nlp_file)
            
            # Save visualization data
            output_path = "data/processed/visualization_engine/visualization_data.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            print("✅ Visualization data generated")
        else:
            raise Exception("Could not find analysis files for visualization")
            
    except Exception as e:
        print(f"❌ Visualization generation failed: {str(e)}")
        raise e
    
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")   

@app.post("/api/process-uploaded")
async def process_uploaded_files(background_tasks: BackgroundTasks):
    global processing_status
    
    if processing_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    uploads_dir = Path("data/uploads")
    txt_files = list(uploads_dir.glob("*.txt"))
    
    if not txt_files:
        raise HTTPException(status_code=404, detail="No .txt files found")
    
    processing_status.update({
        "is_processing": True,
        "current_step": "Starting batch processing",
        "progress": 5,
        "start_time": datetime.now().isoformat(),
        "files_uploaded": [f.name for f in txt_files]
    })
    
    background_tasks.add_task(run_complete_pipeline_batch, [f.name for f in txt_files])
    
    return {"message": f"Started processing {len(txt_files)} files", "files": [f.name for f in txt_files]}

@app.get("/api/results/{filename}")
async def get_results(filename: str):
    """Get processing results for a specific file"""
    
    # Handle URL encoding and get base name
    import urllib.parse
    decoded_filename = urllib.parse.unquote(filename)
    base_name = Path(decoded_filename).stem
    
    # Check for all result files with different naming patterns
    result_files = {
        "llm_analysis": f"data/processed/processed_llm_analyzer_jsons/analyzed_{base_name}.json",
        "nlp_analysis": f"data/processed/processed_nlp_validator_jsons/{base_name}_analysis.json", 
        "visualization_data": "data/processed/visualization_engine/visualization_data.json",
        "cross_validation": "data/processed/cross_validator/cross_validation_output.json"
    }
    
    # Also try without 'analyzed_' prefix for LLM files
    if not os.path.exists(result_files["llm_analysis"]):
        result_files["llm_analysis"] = f"data/processed/processed_llm_analyzer_jsons/{base_name}.json"
    
    results = {}
    
    for result_type, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[result_type] = json.load(f)
            except Exception as e:
                results[result_type] = {"error": f"Failed to load {result_type}: {str(e)}"}
        else:
            results[result_type] = {"error": f"File not found: {file_path}"}
    
    return results


async def run_complete_pipeline_batch(filenames: List[str]):
    global processing_status
    
    try:
        processing_status.update({"current_step": "Preprocessing files", "progress": 20})
        await asyncio.to_thread(process_files)
        
        processing_status.update({"current_step": "Running NLP analysis", "progress": 40})
        await asyncio.to_thread(run_nlp_analysis)
        
        processing_status.update({"current_step": "Running LLM analysis", "progress": 60})
        await asyncio.to_thread(run_llm_analysis)
        
        processing_status.update({"current_step": "Cross validation", "progress": 80})
        await asyncio.to_thread(run_cross_validation)
        
        processing_status.update({"current_step": "Generating visualizations", "progress": 90})
        await asyncio.to_thread(generate_visualization_data)
        
        # Complete
        processing_status.update({
            "is_processing": False,
            "current_step": "Processing complete",
            "progress": 100,
            "error": None
        })
        
        print(f"✅ Complete pipeline finished for batch: {', '.join(filenames)}")
        
        # Schedule cleanup after 1 minute to allow downloads
        schedule_delayed_cleanup(delay_minutes=7)
        
        
    except Exception as e:
        processing_status.update({"is_processing": False, "error": str(e)})

@app.get("/api/cleanup")
async def manual_cleanup():
    """Manually trigger file cleanup"""
    try:
        # Get info before cleanup
        before_info = get_cleanup_info()
        
        # Perform cleanup
        success = cleanup_all_files()
        
        # Get info after cleanup
        after_info = get_cleanup_info()
        
        return {
            "success": success,
            "before": before_info,
            "after": after_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/api/cleanup/info")
async def get_file_cleanup_info():
    """Get information about files that can be cleaned up"""
    try:
        info = get_cleanup_info()
        total_files = sum(data["count"] for data in info.values())
        total_size = sum(data["total_size_mb"] for data in info.values())
        
        return {
            "directories": info,
            "summary": {
                "total_files": total_files,
                "total_size_mb": round(total_size, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup info: {str(e)}")

@app.post("/api/cleanup/{filename}")
async def cleanup_specific_file(filename: str):
    """Clean up files for a specific uploaded file"""
    try:
        import urllib.parse
        decoded_filename = urllib.parse.unquote(filename)
        
        success = cleanup_file(decoded_filename)
        
        return {
            "success": success,
            "filename": decoded_filename,
            "message": f"Cleanup completed for {decoded_filename}" if success else "Cleanup failed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File cleanup failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 AI-Powered Metadata Tagging System Starting...")
    print(f"📁 Project root: {os.getcwd()}")
    
    # Create necessary directories
    directories = [
        "data/uploads",
        "data/processed/LLM_jsons",
        "data/processed/NLP_jsons", 
        "data/processed/processed_llm_analyzer_jsons",
        "data/processed/processed_nlp_validator_jsons",
        "data/processed/cross_validator",
        "data/processed/visualization_engine"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Application initialized successfully")
    print("🧹 File cleanup manager ready")
    
    # Check if cleanup directories exist (simple check)
    cleanup_dirs = [
        "data/uploads",
        "data/processed/LLM_jsons",
        "data/processed/NLP_jsons",
        "data/processed/processed_llm_analyzer_jsons",
        "data/processed/processed_nlp_validator_jsons", 
        "data/processed/cross_validator",
        "data/processed/visualization_engine",
    ]
    
    total_files = 0
    for cleanup_dir in cleanup_dirs:
        if os.path.exists(cleanup_dir):
            try:
                files = list(Path(cleanup_dir).glob("*"))
                total_files += len([f for f in files if f.is_file()])
            except Exception as e:
                print(f"⚠️ Could not check directory {cleanup_dir}: {e}")
    
    if total_files > 0:
        print(f"📁 Found {total_files} existing files in cleanup directories")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🛑 Application shutting down...")
    
    # Perform immediate cleanup on shutdown
    success = cleanup_all_files()
    
    if success:
        print("🧹 Cleanup completed on shutdown")
    else:
        print("⚠️ Cleanup encountered issues during shutdown")

if __name__ == "__main__":
    print("🎬 Starting AI-Powered Metadata Tagging System (Backend Only)")
    print("🔗 API available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🔄 Upload endpoint: POST http://localhost:8000/api/upload")
    print("📊 Status endpoint: GET http://localhost:8000/api/status")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )