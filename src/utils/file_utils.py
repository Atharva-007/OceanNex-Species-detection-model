"""File management utilities."""

import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from .logger import get_logger


logger = get_logger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


class FileManager:
    """Manages file operations for the fish classifier."""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize file manager.
        
        Args:
            base_path: Base directory path
        """
        self.base_path = Path(base_path)
        self.logger = get_logger(self.__class__.__name__)
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved JSON to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded data or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.warning(f"JSON file not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded JSON from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """
        Copy file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            destination = Path(destination)
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            
            self.logger.info(f"Copied {source} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source} to {destination}: {e}")
            return False
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """
        Move file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            destination = Path(destination)
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            
            self.logger.info(f"Moved {source} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {source} to {destination}: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete file.
        
        Args:
            file_path: File path to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file: {file_path}")
            else:
                self.logger.warning(f"File not found for deletion: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    def list_files(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                self.logger.warning(f"Directory not found: {directory}")
                return []
            
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            # Filter to only files (not directories)
            files = [f for f in files if f.is_file()]
            
            self.logger.info(f"Found {len(files)} files in {directory}")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list files in {directory}: {e}")
            return []
    
    def get_file_size(self, file_path: Union[str, Path]) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            file_path: File path
            
        Returns:
            File size in bytes or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if file_path.exists():
                return file_path.stat().st_size
            else:
                self.logger.warning(f"File not found: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get file size for {file_path}: {e}")
            return None
    
    def clean_directory(self, directory: Union[str, Path], keep_subdirs: bool = True) -> bool:
        """
        Clean directory contents.
        
        Args:
            directory: Directory to clean
            keep_subdirs: Whether to keep subdirectories
            
        Returns:
            True if successful, False otherwise
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                self.logger.warning(f"Directory not found: {directory}")
                return True
            
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir() and not keep_subdirs:
                    shutil.rmtree(item)
            
            self.logger.info(f"Cleaned directory: {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clean directory {directory}: {e}")
            return False