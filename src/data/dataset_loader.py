"""
Dataset loading module for fish species classification.

This module provides functionality for loading datasets from various sources
and formats including directory structures, CSV files, and archive formats.
"""

import os
import csv
import zipfile
import tarfile
import shutil
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

from ..utils.logger import get_logger
from ..utils.exceptions import DatasetError, ValidationError
from ..utils.file_utils import ensure_directory


class DatasetLoader:
    """
    Versatile dataset loader supporting multiple formats and sources.
    
    Supports:
    - Directory-based datasets (ImageNet style)
    - CSV-based datasets with image paths
    - Archive files (zip, tar)
    - Multiple dataset merging
    """
    
    def __init__(self):
        """Initialize the dataset loader."""
        self.logger = get_logger(__name__)
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.supported_archive_extensions = {'.zip', '.tar', '.tar.gz', '.tgz'}
    
    def load_from_directory(self,
                           dataset_path: str,
                           expected_structure: str = "class_folders") -> Tuple[List[str], List[str]]:
        """
        Load dataset from directory structure.
        
        Args:
            dataset_path: Path to dataset directory
            expected_structure: Expected structure ('class_folders' or 'flat')
            
        Returns:
            Tuple of (image_paths, class_names)
        """
        try:
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                raise DatasetError(f"Dataset path does not exist: {dataset_path}")
            
            if expected_structure == "class_folders":
                return self._load_class_folder_structure(dataset_path)
            elif expected_structure == "flat":
                return self._load_flat_structure(dataset_path)
            else:
                raise ValidationError(f"Unsupported structure type: {expected_structure}")
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset from directory: {str(e)}")
    
    def _load_class_folder_structure(self, dataset_path: Path) -> Tuple[List[str], List[str]]:
        """Load dataset with class-based folder structure."""
        image_paths = []
        class_names = []
        
        # Find all subdirectories as classes
        class_dirs = [d for d in dataset_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        if not class_dirs:
            raise DatasetError("No class directories found")
        
        # Sort for consistent ordering
        class_dirs = sorted(class_dirs, key=lambda x: x.name)
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            self.logger.debug(f"Processing class: {class_name}")
            
            # Find all image files in class directory
            class_images = []
            for file_path in class_dir.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.supported_image_extensions):
                    class_images.append(str(file_path))
            
            if class_images:
                image_paths.extend(class_images)
                class_names.extend([class_name] * len(class_images))
                self.logger.debug(f"Found {len(class_images)} images in class '{class_name}'")
            else:
                self.logger.warning(f"No images found in class directory: {class_dir}")
        
        self.logger.info(f"Loaded {len(image_paths)} images from {len(set(class_names))} classes")
        return image_paths, class_names
    
    def _load_flat_structure(self, dataset_path: Path) -> Tuple[List[str], List[str]]:
        """Load dataset with flat structure (all images in one directory)."""
        image_paths = []
        
        # Find all image files
        for file_path in dataset_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_image_extensions):
                image_paths.append(str(file_path))
        
        if not image_paths:
            raise DatasetError("No image files found in dataset directory")
        
        # For flat structure, we don't have class information from directories
        class_names = ['unknown'] * len(image_paths)
        
        self.logger.info(f"Loaded {len(image_paths)} images from flat structure")
        return image_paths, class_names
    
    def load_from_csv(self,
                     csv_file: str,
                     image_column: str = 'image_path',
                     label_column: str = 'label',
                     base_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Load dataset from CSV file.
        
        Args:
            csv_file: Path to CSV file
            image_column: Name of column containing image paths
            label_column: Name of column containing labels
            base_path: Base path to prepend to relative image paths
            
        Returns:
            Tuple of (image_paths, class_names)
        """
        try:
            if not os.path.exists(csv_file):
                raise DatasetError(f"CSV file not found: {csv_file}")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            if image_column not in df.columns:
                raise ValidationError(f"Image column '{image_column}' not found in CSV")
            if label_column not in df.columns:
                raise ValidationError(f"Label column '{label_column}' not found in CSV")
            
            # Extract image paths and labels
            image_paths = df[image_column].tolist()
            class_names = df[label_column].tolist()
            
            # Prepend base path if provided
            if base_path:
                base_path = Path(base_path)
                image_paths = [str(base_path / path) for path in image_paths]
            
            # Validate image files exist
            valid_paths = []
            valid_labels = []
            
            for path, label in zip(image_paths, class_names):
                if os.path.exists(path):
                    valid_paths.append(path)
                    valid_labels.append(str(label))
                else:
                    self.logger.warning(f"Image file not found: {path}")
            
            if not valid_paths:
                raise DatasetError("No valid image files found from CSV")
            
            self.logger.info(f"Loaded {len(valid_paths)} images from CSV file")
            return valid_paths, valid_labels
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset from CSV: {str(e)}")
    
    def load_from_archive(self,
                         archive_path: str,
                         extract_path: str,
                         expected_structure: str = "class_folders") -> Tuple[List[str], List[str]]:
        """
        Load dataset from archive file (zip, tar).
        
        Args:
            archive_path: Path to archive file
            extract_path: Path to extract archive contents
            expected_structure: Expected structure after extraction
            
        Returns:
            Tuple of (image_paths, class_names)
        """
        try:
            archive_path = Path(archive_path)
            extract_path = Path(extract_path)
            
            if not archive_path.exists():
                raise DatasetError(f"Archive file not found: {archive_path}")
            
            # Create extraction directory
            ensure_directory(str(extract_path))
            
            # Extract archive based on type
            if archive_path.suffix.lower() == '.zip':
                self._extract_zip(archive_path, extract_path)
            elif archive_path.suffix.lower() in {'.tar', '.tar.gz', '.tgz'}:
                self._extract_tar(archive_path, extract_path)
            else:
                raise ValidationError(f"Unsupported archive format: {archive_path.suffix}")
            
            # Load from extracted directory
            return self.load_from_directory(str(extract_path), expected_structure)
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset from archive: {str(e)}")
    
    def _extract_zip(self, archive_path: Path, extract_path: Path):
        """Extract ZIP archive."""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            self.logger.info(f"Extracted ZIP archive to: {extract_path}")
        except Exception as e:
            raise DatasetError(f"Failed to extract ZIP archive: {str(e)}")
    
    def _extract_tar(self, archive_path: Path, extract_path: Path):
        """Extract TAR archive."""
        try:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)
            self.logger.info(f"Extracted TAR archive to: {extract_path}")
        except Exception as e:
            raise DatasetError(f"Failed to extract TAR archive: {str(e)}")
    
    def merge_datasets(self,
                      datasets: List[Dict[str, Union[str, List]]],
                      output_path: str,
                      merge_strategy: str = "combine") -> Tuple[List[str], List[str]]:
        """
        Merge multiple datasets into one.
        
        Args:
            datasets: List of dataset dictionaries with 'paths' and 'labels' or 'path' key
            output_path: Path for merged dataset output
            merge_strategy: Strategy for merging ('combine', 'copy_files')
            
        Returns:
            Tuple of (image_paths, class_names)
        """
        try:
            merged_paths = []
            merged_labels = []
            
            for i, dataset in enumerate(datasets):
                self.logger.info(f"Processing dataset {i+1}/{len(datasets)}")
                
                if 'paths' in dataset and 'labels' in dataset:
                    # Dataset already loaded
                    paths = dataset['paths']
                    labels = dataset['labels']
                elif 'path' in dataset:
                    # Load dataset from path
                    structure = dataset.get('structure', 'class_folders')
                    paths, labels = self.load_from_directory(dataset['path'], structure)
                else:
                    raise ValidationError(f"Invalid dataset format at index {i}")
                
                if merge_strategy == "copy_files":
                    # Copy files to new location with class organization
                    copied_paths = self._copy_dataset_files(paths, labels, output_path, f"dataset_{i}")
                    merged_paths.extend(copied_paths)
                else:
                    # Just combine paths (files stay in original locations)
                    merged_paths.extend(paths)
                
                merged_labels.extend(labels)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            unique_labels = []
            
            for path, label in zip(merged_paths, merged_labels):
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)
                    unique_labels.append(label)
            
            self.logger.info(f"Merged {len(datasets)} datasets into {len(unique_paths)} images")
            return unique_paths, unique_labels
            
        except Exception as e:
            raise DatasetError(f"Failed to merge datasets: {str(e)}")
    
    def _copy_dataset_files(self,
                           image_paths: List[str],
                           labels: List[str],
                           output_path: str,
                           dataset_prefix: str) -> List[str]:
        """Copy dataset files to organized structure."""
        try:
            output_path = Path(output_path)
            copied_paths = []
            
            for path, label in zip(image_paths, labels):
                if not os.path.exists(path):
                    self.logger.warning(f"Source file not found: {path}")
                    continue
                
                # Create class directory
                class_dir = output_path / label
                ensure_directory(str(class_dir))
                
                # Generate unique filename
                source_path = Path(path)
                filename = f"{dataset_prefix}_{source_path.name}"
                dest_path = class_dir / filename
                
                # Copy file
                shutil.copy2(path, dest_path)
                copied_paths.append(str(dest_path))
            
            return copied_paths
            
        except Exception as e:
            raise DatasetError(f"Failed to copy dataset files: {str(e)}")
    
    def convert_to_tensorflow_format(self,
                                   image_paths: List[str],
                                   labels: List[str],
                                   output_path: str,
                                   class_mapping: Optional[Dict[str, int]] = None) -> str:
        """
        Convert dataset to TensorFlow-friendly format.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            output_path: Output directory path
            class_mapping: Optional mapping of class names to indices
            
        Returns:
            Path to the converted dataset
        """
        try:
            output_path = Path(output_path)
            ensure_directory(str(output_path))
            
            # Create class mapping if not provided
            if class_mapping is None:
                unique_labels = sorted(set(labels))
                class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            
            # Create metadata file
            metadata = {
                'num_classes': len(class_mapping),
                'class_names': list(class_mapping.keys()),
                'class_mapping': class_mapping,
                'total_samples': len(image_paths)
            }
            
            # Save metadata
            import json
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create CSV file with paths and numeric labels
            csv_data = []
            for path, label in zip(image_paths, labels):
                if os.path.exists(path):
                    csv_data.append({
                        'image_path': path,
                        'label': label,
                        'label_index': class_mapping.get(label, -1)
                    })
            
            # Save CSV
            csv_path = output_path / 'dataset.csv'
            with open(csv_path, 'w', newline='') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            self.logger.info(f"Converted dataset to TensorFlow format: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise DatasetError(f"Failed to convert to TensorFlow format: {str(e)}")
    
    def validate_dataset_structure(self,
                                 dataset_path: str,
                                 expected_structure: str = "class_folders") -> Dict[str, any]:
        """
        Validate dataset structure and return validation report.
        
        Args:
            dataset_path: Path to dataset directory
            expected_structure: Expected structure type
            
        Returns:
            Dictionary with validation results
        """
        try:
            dataset_path = Path(dataset_path)
            
            validation_report = {
                'valid': True,
                'structure_type': expected_structure,
                'issues': [],
                'warnings': [],
                'statistics': {
                    'total_files': 0,
                    'image_files': 0,
                    'non_image_files': 0,
                    'empty_directories': 0,
                    'classes_found': 0
                }
            }
            
            if not dataset_path.exists():
                validation_report['valid'] = False
                validation_report['issues'].append(f"Dataset path does not exist: {dataset_path}")
                return validation_report
            
            # Count files and analyze structure
            all_files = list(dataset_path.rglob('*'))
            image_files = [f for f in all_files 
                          if f.is_file() and f.suffix.lower() in self.supported_image_extensions]
            non_image_files = [f for f in all_files 
                              if f.is_file() and f.suffix.lower() not in self.supported_image_extensions]
            
            validation_report['statistics']['total_files'] = len([f for f in all_files if f.is_file()])
            validation_report['statistics']['image_files'] = len(image_files)
            validation_report['statistics']['non_image_files'] = len(non_image_files)
            
            if expected_structure == "class_folders":
                class_dirs = [d for d in dataset_path.iterdir() 
                             if d.is_dir() and not d.name.startswith('.')]
                validation_report['statistics']['classes_found'] = len(class_dirs)
                
                if not class_dirs:
                    validation_report['valid'] = False
                    validation_report['issues'].append("No class directories found")
                
                # Check for empty class directories
                empty_dirs = []
                for class_dir in class_dirs:
                    class_images = [f for f in class_dir.rglob('*') 
                                   if f.is_file() and f.suffix.lower() in self.supported_image_extensions]
                    if not class_images:
                        empty_dirs.append(class_dir.name)
                
                validation_report['statistics']['empty_directories'] = len(empty_dirs)
                if empty_dirs:
                    validation_report['warnings'].append(f"Empty class directories: {empty_dirs}")
            
            # Check for non-image files
            if non_image_files:
                validation_report['warnings'].append(f"Found {len(non_image_files)} non-image files")
            
            # Check minimum requirements
            if len(image_files) == 0:
                validation_report['valid'] = False
                validation_report['issues'].append("No image files found")
            
            return validation_report
            
        except Exception as e:
            raise DatasetError(f"Dataset structure validation failed: {str(e)}")