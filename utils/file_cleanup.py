"""
File Cleanup Utility - Automatic cleanup of user uploaded and processed files
Handles cleanup on process completion and application shutdown
"""

import os
import shutil
import threading
import time
import atexit
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileCleanupManager:
    """Manages automatic cleanup of user uploaded and processed files"""
    
    def __init__(self):
        self.cleanup_dirs = [
            "data/uploads",
            "data/processed/LLM_jsons",
            "data/processed/NLP_jsons",
            "data/processed/processed_llm_analyzer_jsons",
            "data/processed/processed_nlp_validator_jsons",
            "data/processed/cross_validator",
            "data/processed/visualization_engine",
        ]
        self.cleanup_enabled = True
        self.cleanup_lock = threading.Lock()
        
        # Register cleanup on application exit
        atexit.register(self.cleanup_on_exit)
        
    def cleanup_user_files(self, delay_seconds: int = 0) -> bool:
        """
        Clean up all user uploaded and processed files
        
        Args:
            delay_seconds: Optional delay before cleanup (useful for allowing downloads)
            
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        if not self.cleanup_enabled:
            return False
            
        def cleanup_task():
            if delay_seconds > 0:
                time.sleep(delay_seconds)
                
            with self.cleanup_lock:
                try:
                    cleanup_count = 0
                    
                    for cleanup_dir in self.cleanup_dirs:
                        if os.path.exists(cleanup_dir):
                            # Get files before cleanup for logging
                            files = list(Path(cleanup_dir).glob("*"))
                            
                            if files:
                                for file_path in files:
                                    try:
                                        if file_path.is_file():
                                            os.remove(file_path)
                                            cleanup_count += 1
                                            logger.info(f"Cleaned up: {file_path}")
                                    except Exception as e:
                                        logger.error(f"Failed to remove {file_path}: {e}")
                                
                                # Remove empty directories if they exist
                                try:
                                    if not any(Path(cleanup_dir).iterdir()):
                                        # Don't remove the directory itself, just ensure it's empty
                                        pass
                                except Exception as e:
                                    logger.error(f"Error checking directory {cleanup_dir}: {e}")
                    
                    if cleanup_count > 0:
                        logger.info(f"Successfully cleaned up {cleanup_count} files")
                        return True
                    else:
                        logger.info("No files to clean up")
                        return True
                        
                except Exception as e:
                    logger.error(f"Cleanup failed: {e}")
                    return False
        
        # Run cleanup in background thread if delay is specified
        if delay_seconds > 0:
            cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
            cleanup_thread.start()
            return True
        else:
            return cleanup_task()
    
    def cleanup_specific_file(self, filename: str) -> bool:
        """
        Clean up files related to a specific uploaded file
        
        Args:
            filename: Name of the original uploaded file
            
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        if not self.cleanup_enabled:
            return False
            
        base_name = Path(filename).stem
        
        with self.cleanup_lock:
            try:
                cleanup_count = 0
                
                # Define file patterns to clean up
                file_patterns = [
                    f"data/uploads/{filename}",
                    f"data/processed/processed_llm_analyzer_jsons/analyzed_{base_name}.json",
                    f"data/processed/processed_llm_analyzer_jsons/{base_name}.json",
                    f"data/processed/processed_nlp_validator_jsons/{base_name}_analysis.json",
                ]
                
                for file_pattern in file_patterns:
                    if os.path.exists(file_pattern):
                        try:
                            os.remove(file_pattern)
                            cleanup_count += 1
                            logger.info(f"Cleaned up specific file: {file_pattern}")
                        except Exception as e:
                            logger.error(f"Failed to remove {file_pattern}: {e}")
                
                logger.info(f"Cleaned up {cleanup_count} files for {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Specific file cleanup failed for {filename}: {e}")
                return False
    
    def schedule_cleanup_after_processing(self, delay_minutes: int = 1):
        """
        Schedule cleanup to run after processing is complete
        Useful for allowing time for users to download results
        
        Args:
            delay_minutes: Minutes to wait before cleanup
        """
        delay_seconds = delay_minutes * 60
        logger.info(f"Scheduling cleanup in {delay_minutes} minute(s)")
        self.cleanup_user_files(delay_seconds=delay_seconds)
    
    def cleanup_on_exit(self):
        """Cleanup function called on application exit"""
        logger.info("Application shutting down - performing cleanup")
        self.cleanup_user_files()
    
    def disable_cleanup(self):
        """Disable automatic cleanup (for development/debugging)"""
        self.cleanup_enabled = False
        logger.info("File cleanup disabled")
    
    def enable_cleanup(self):
        """Enable automatic cleanup"""
        self.cleanup_enabled = True
        logger.info("File cleanup enabled")
    
    def get_file_info(self) -> dict:
        """Get information about current files in cleanup directories"""
        file_info = {}
        
        for cleanup_dir in self.cleanup_dirs:
            if os.path.exists(cleanup_dir):
                files = list(Path(cleanup_dir).glob("*"))
                file_info[cleanup_dir] = {
                    "count": len(files),
                    "files": [f.name for f in files],
                    "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024)
                }
            else:
                file_info[cleanup_dir] = {
                    "count": 0,
                    "files": [],
                    "total_size_mb": 0
                }
        
        return file_info

# Global instance
cleanup_manager = FileCleanupManager()

# Convenience functions for easy integration
def cleanup_all_files(delay_seconds: int = 0) -> bool:
    """Clean up all user files"""
    return cleanup_manager.cleanup_user_files(delay_seconds)

def cleanup_file(filename: str) -> bool:
    """Clean up files for specific filename"""
    return cleanup_manager.cleanup_specific_file(filename)

def schedule_delayed_cleanup(delay_minutes: int = 1):
    """Schedule cleanup after delay"""
    cleanup_manager.schedule_cleanup_after_processing(delay_minutes)

def disable_cleanup():
    """Disable cleanup for development"""
    cleanup_manager.disable_cleanup()

def enable_cleanup():
    """Enable cleanup"""
    cleanup_manager.enable_cleanup()

def get_cleanup_info() -> dict:
    """Get current file information"""
    return cleanup_manager.get_file_info()

# Integration hooks for your existing code
class ProcessingCleanupHooks:
    """Hooks to integrate cleanup with your existing processing pipeline"""
    
    @staticmethod
    def on_processing_complete(filename: Optional[str] = None, delay_minutes: int = 1):
        """
        Call this when processing is complete
        
        Args:
            filename: Specific filename if cleaning up one file, None for all files
            delay_minutes: Minutes to wait before cleanup (allows time for downloads)
        """
        if filename:
            # Clean up specific file after delay
            def delayed_cleanup():
                time.sleep(delay_minutes * 60)
                cleanup_file(filename)
            
            threading.Thread(target=delayed_cleanup, daemon=True).start()
            logger.info(f"Scheduled cleanup for {filename} in {delay_minutes} minute(s)")
        else:
            # Clean up all files after delay
            schedule_delayed_cleanup(delay_minutes)
    
    @staticmethod
    def on_processing_error(filename: Optional[str] = None):
        """
        Call this when processing fails
        Immediate cleanup since there are no results to download
        
        Args:
            filename: Specific filename if cleaning up one file, None for all files
        """
        if filename:
            cleanup_file(filename)
            logger.info(f"Cleaned up files for failed processing: {filename}")
        else:
            cleanup_all_files()
            logger.info("Cleaned up all files due to processing error")
    
    @staticmethod
    def on_user_disconnect():
        """Call this when user disconnects or session ends"""
        # Immediate cleanup when user leaves
        cleanup_all_files()
        logger.info("User disconnected - performed cleanup")

# Export hooks for easy access
hooks = ProcessingCleanupHooks()

if __name__ == "__main__":
    # Test the cleanup manager
    print("File Cleanup Manager Test")
    print("Current file info:")
    info = get_cleanup_info()
    for directory, data in info.items():
        print(f"  {directory}: {data['count']} files ({data['total_size_mb']:.2f} MB)")
    
    print("\nTesting cleanup...")
    result = cleanup_all_files()
    print(f"Cleanup result: {result}")