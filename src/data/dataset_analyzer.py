"""
Dataset analysis module for fish species classification.

This module provides comprehensive dataset analysis functionality including
statistical analysis, visualization, and quality assessment.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import Counter
import cv2
from PIL import Image

from ..utils.logger import get_logger
from ..utils.exceptions import DatasetError, ValidationError
from ..utils.visualization import save_plot


class DatasetAnalyzer:
    """
    Comprehensive dataset analyzer for statistical analysis and visualization.
    
    Features:
    - Class distribution analysis
    - Image quality assessment
    - Statistical summaries
    - Visualization generation
    - Data quality reports
    """
    
    def __init__(self,
                 image_paths: List[str],
                 labels: List[str],
                 class_names: Optional[List[str]] = None):
        """
        Initialize the dataset analyzer.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            class_names: Optional list of class names (if None, derived from labels)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.logger = get_logger(__name__)
        
        # Validate inputs
        if len(image_paths) != len(labels):
            raise ValidationError("Number of image paths must match number of labels")
        
        # Set up class information
        if class_names is None:
            self.class_names = sorted(list(set(labels)))
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Analysis results storage
        self.analysis_results: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, Any] = {}
        
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of classes in the dataset.
        
        Returns:
            Dictionary containing class distribution statistics
        """
        try:
            self.logger.info("Analyzing class distribution...")
            
            # Count samples per class
            class_counts = Counter(self.labels)
            
            # Calculate statistics
            counts = list(class_counts.values())
            total_samples = len(self.labels)
            
            distribution_stats = {
                'total_samples': total_samples,
                'num_classes': self.num_classes,
                'class_counts': dict(class_counts),
                'min_samples': min(counts),
                'max_samples': max(counts),
                'mean_samples': np.mean(counts),
                'median_samples': np.median(counts),
                'std_samples': np.std(counts),
                'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
                'class_percentages': {
                    class_name: (count / total_samples) * 100
                    for class_name, count in class_counts.items()
                }
            }
            
            # Identify imbalanced classes
            mean_count = np.mean(counts)
            underrepresented = [
                name for name, count in class_counts.items()
                if count < 0.5 * mean_count
            ]
            overrepresented = [
                name for name, count in class_counts.items()
                if count > 2.0 * mean_count
            ]
            
            distribution_stats['underrepresented_classes'] = underrepresented
            distribution_stats['overrepresented_classes'] = overrepresented
            
            # Calculate Gini coefficient for inequality measure
            gini_coefficient = self._calculate_gini_coefficient(counts)
            distribution_stats['gini_coefficient'] = gini_coefficient
            
            self.analysis_results['class_distribution'] = distribution_stats
            
            self.logger.info(f"Class distribution analysis completed. "
                           f"Imbalance ratio: {distribution_stats['imbalance_ratio']:.2f}")
            
            return distribution_stats
            
        except Exception as e:
            raise DatasetError(f"Class distribution analysis failed: {str(e)}")
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        try:
            sorted_values = sorted(values)
            n = len(sorted_values)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        except:
            return 0.0
    
    def analyze_image_properties(self,
                                sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze image properties like dimensions, file sizes, and formats.
        
        Args:
            sample_size: Number of images to sample for analysis (None for all)
            
        Returns:
            Dictionary containing image property statistics
        """
        try:
            self.logger.info("Analyzing image properties...")
            
            # Sample images if needed
            if sample_size and sample_size < len(self.image_paths):
                indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
                sample_paths = [self.image_paths[i] for i in indices]
            else:
                sample_paths = self.image_paths
            
            # Collect image properties
            widths, heights, file_sizes, formats, channels = [], [], [], [], []
            aspect_ratios = []
            corrupted_files = []
            
            for i, path in enumerate(sample_paths):
                try:
                    if not os.path.exists(path):
                        corrupted_files.append(path)
                        continue
                    
                    # Get file size
                    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                    file_sizes.append(file_size)
                    
                    # Get image properties
                    with Image.open(path) as img:
                        width, height = img.size
                        format_type = img.format
                        mode = img.mode
                        
                        widths.append(width)
                        heights.append(height)
                        formats.append(format_type)
                        
                        # Convert mode to channel count
                        if mode in ['L', 'P']:
                            channel_count = 1
                        elif mode in ['RGB', 'YCbCr', 'LAB', 'HSV']:
                            channel_count = 3
                        elif mode in ['RGBA', 'CMYK']:
                            channel_count = 4
                        else:
                            channel_count = len(mode)
                        
                        channels.append(channel_count)
                        aspect_ratios.append(width / height)
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze image {path}: {str(e)}")
                    corrupted_files.append(path)
                
                # Progress logging
                if (i + 1) % 1000 == 0:
                    self.logger.debug(f"Analyzed {i + 1}/{len(sample_paths)} images")
            
            # Calculate statistics
            property_stats = {
                'sample_size': len(sample_paths),
                'analyzed_images': len(widths),
                'corrupted_files': len(corrupted_files),
                'width_stats': self._calculate_stats(widths),
                'height_stats': self._calculate_stats(heights),
                'file_size_stats': self._calculate_stats(file_sizes),
                'aspect_ratio_stats': self._calculate_stats(aspect_ratios),
                'format_distribution': dict(Counter(formats)),
                'channel_distribution': dict(Counter(channels)),
                'common_resolutions': self._find_common_resolutions(widths, heights),
                'corrupted_file_paths': corrupted_files[:10]  # Show first 10
            }
            
            self.analysis_results['image_properties'] = property_stats
            
            self.logger.info(f"Image properties analysis completed. "
                           f"Analyzed {property_stats['analyzed_images']} images, "
                           f"found {property_stats['corrupted_files']} corrupted files")
            
            return property_stats
            
        except Exception as e:
            raise DatasetError(f"Image properties analysis failed: {str(e)}")
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {'count': 0}
        
        return {
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _find_common_resolutions(self,
                               widths: List[int],
                               heights: List[int]) -> Dict[str, int]:
        """Find most common image resolutions."""
        resolutions = [f"{w}x{h}" for w, h in zip(widths, heights)]
        resolution_counts = Counter(resolutions)
        
        # Return top 10 most common resolutions
        return dict(resolution_counts.most_common(10))
    
    def analyze_data_quality(self,
                           check_duplicates: bool = True,
                           check_corruption: bool = True) -> Dict[str, Any]:
        """
        Analyze data quality issues in the dataset.
        
        Args:
            check_duplicates: Whether to check for duplicate images
            check_corruption: Whether to check for corrupted images
            
        Returns:
            Dictionary containing data quality metrics
        """
        try:
            self.logger.info("Analyzing data quality...")
            
            quality_issues = {
                'duplicate_files': [],
                'corrupted_files': [],
                'missing_files': [],
                'empty_files': [],
                'invalid_labels': [],
                'quality_score': 0.0
            }
            
            # Check for missing files
            for path in self.image_paths:
                if not os.path.exists(path):
                    quality_issues['missing_files'].append(path)
            
            # Check for empty files
            for path in self.image_paths:
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    quality_issues['empty_files'].append(path)
            
            # Check for corrupted files
            if check_corruption:
                quality_issues['corrupted_files'] = self._find_corrupted_images()
            
            # Check for duplicate files
            if check_duplicates:
                quality_issues['duplicate_files'] = self._find_duplicate_images()
            
            # Check for invalid labels
            valid_labels = set(self.class_names)
            for i, label in enumerate(self.labels):
                if label not in valid_labels:
                    quality_issues['invalid_labels'].append({
                        'index': i,
                        'label': label,
                        'path': self.image_paths[i]
                    })
            
            # Calculate quality score
            total_issues = (
                len(quality_issues['duplicate_files']) +
                len(quality_issues['corrupted_files']) +
                len(quality_issues['missing_files']) +
                len(quality_issues['empty_files']) +
                len(quality_issues['invalid_labels'])
            )
            
            quality_issues['quality_score'] = max(0.0, 1.0 - (total_issues / len(self.image_paths)))
            quality_issues['total_issues'] = total_issues
            
            self.quality_metrics = quality_issues
            
            self.logger.info(f"Data quality analysis completed. "
                           f"Quality score: {quality_issues['quality_score']:.3f}, "
                           f"Total issues: {total_issues}")
            
            return quality_issues
            
        except Exception as e:
            raise DatasetError(f"Data quality analysis failed: {str(e)}")
    
    def _find_corrupted_images(self) -> List[str]:
        """Find corrupted or unreadable images."""
        corrupted = []
        
        for path in self.image_paths:
            try:
                if os.path.exists(path):
                    with Image.open(path) as img:
                        img.verify()  # Verify image integrity
            except Exception:
                corrupted.append(path)
        
        return corrupted
    
    def _find_duplicate_images(self) -> List[Tuple[str, str]]:
        """Find duplicate images based on file size and basic comparison."""
        # This is a simplified duplicate detection
        # For more robust detection, consider using image hashing
        
        duplicates = []
        size_groups = {}
        
        # Group files by size
        for path in self.image_paths:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(path)
        
        # Check for potential duplicates (same size)
        for size, paths in size_groups.items():
            if len(paths) > 1:
                # For files with same size, add as potential duplicates
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        duplicates.append((paths[i], paths[j]))
        
        return duplicates
    
    def generate_visualizations(self,
                              output_dir: str,
                              save_plots: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive visualizations of the dataset.
        
        Args:
            output_dir: Directory to save visualization plots
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        try:
            self.logger.info("Generating dataset visualizations...")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_paths = {}
            
            # Class distribution bar plot
            plot_paths['class_distribution'] = self._plot_class_distribution(
                output_dir, save_plots
            )
            
            # Class distribution pie chart
            plot_paths['class_distribution_pie'] = self._plot_class_distribution_pie(
                output_dir, save_plots
            )
            
            # Image properties histograms
            if 'image_properties' in self.analysis_results:
                plot_paths['image_properties'] = self._plot_image_properties(
                    output_dir, save_plots
                )
            
            # Sample images grid
            plot_paths['sample_images'] = self._plot_sample_images(
                output_dir, save_plots
            )
            
            self.logger.info(f"Generated {len(plot_paths)} visualizations")
            return plot_paths
            
        except Exception as e:
            raise DatasetError(f"Visualization generation failed: {str(e)}")
    
    def _plot_class_distribution(self,
                               output_dir: Path,
                               save_plot: bool) -> Optional[str]:
        """Plot class distribution as bar chart."""
        try:
            if 'class_distribution' not in self.analysis_results:
                self.analyze_class_distribution()
            
            class_counts = self.analysis_results['class_distribution']['class_counts']
            
            plt.figure(figsize=(max(12, len(self.class_names) * 0.5), 8))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
            plt.xlabel('Classes', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_plot:
                plot_path = output_dir / 'class_distribution.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to plot class distribution: {str(e)}")
            return None
    
    def _plot_class_distribution_pie(self,
                                   output_dir: Path,
                                   save_plot: bool) -> Optional[str]:
        """Plot class distribution as pie chart."""
        try:
            if 'class_distribution' not in self.analysis_results:
                self.analyze_class_distribution()
            
            class_counts = self.analysis_results['class_distribution']['class_counts']
            
            plt.figure(figsize=(12, 8))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # Show only top 15 classes in pie chart to avoid clutter
            if len(classes) > 15:
                top_indices = np.argsort(counts)[-15:]
                classes = [classes[i] for i in top_indices]
                counts = [counts[i] for i in top_indices]
                other_count = sum(self.analysis_results['class_distribution']['class_counts'].values()) - sum(counts)
                if other_count > 0:
                    classes.append('Others')
                    counts.append(other_count)
            
            plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            plt.title('Class Distribution (Pie Chart)', fontsize=16, fontweight='bold')
            plt.axis('equal')
            
            if save_plot:
                plot_path = output_dir / 'class_distribution_pie.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to plot pie chart: {str(e)}")
            return None
    
    def _plot_image_properties(self,
                             output_dir: Path,
                             save_plot: bool) -> Optional[str]:
        """Plot image properties histograms."""
        try:
            props = self.analysis_results['image_properties']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Width distribution
            if props['width_stats']['count'] > 0:
                axes[0, 0].hist(range(int(props['width_stats']['min']), 
                                    int(props['width_stats']['max'])), 
                               bins=30, alpha=0.7, color='blue')
                axes[0, 0].set_title('Width Distribution')
                axes[0, 0].set_xlabel('Width (pixels)')
                axes[0, 0].set_ylabel('Frequency')
            
            # Height distribution
            if props['height_stats']['count'] > 0:
                axes[0, 1].hist(range(int(props['height_stats']['min']), 
                                    int(props['height_stats']['max'])), 
                               bins=30, alpha=0.7, color='green')
                axes[0, 1].set_title('Height Distribution')
                axes[0, 1].set_xlabel('Height (pixels)')
                axes[0, 1].set_ylabel('Frequency')
            
            # File size distribution
            if props['file_size_stats']['count'] > 0:
                axes[1, 0].hist([props['file_size_stats']['min'], props['file_size_stats']['max']], 
                               bins=30, alpha=0.7, color='red')
                axes[1, 0].set_title('File Size Distribution')
                axes[1, 0].set_xlabel('Size (MB)')
                axes[1, 0].set_ylabel('Frequency')
            
            # Format distribution
            if props['format_distribution']:
                formats = list(props['format_distribution'].keys())
                counts = list(props['format_distribution'].values())
                axes[1, 1].bar(formats, counts, alpha=0.7, color='orange')
                axes[1, 1].set_title('Format Distribution')
                axes[1, 1].set_xlabel('Format')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = output_dir / 'image_properties.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to plot image properties: {str(e)}")
            return None
    
    def _plot_sample_images(self,
                          output_dir: Path,
                          save_plot: bool,
                          samples_per_class: int = 3) -> Optional[str]:
        """Plot sample images from each class."""
        try:
            # Calculate grid size
            cols = min(samples_per_class, 5)
            rows = min(len(self.class_names), 10)  # Limit to 10 classes for visualization
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            if cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, class_name in enumerate(self.class_names[:rows]):
                # Find images for this class
                class_indices = [j for j, label in enumerate(self.labels) if label == class_name]
                
                # Sample images
                sample_indices = np.random.choice(
                    class_indices, 
                    min(samples_per_class, len(class_indices)), 
                    replace=False
                )
                
                for j in range(cols):
                    if j < len(sample_indices):
                        # Load and display image
                        try:
                            img_path = self.image_paths[sample_indices[j]]
                            img = Image.open(img_path)
                            axes[i, j].imshow(img)
                            axes[i, j].set_title(f'{class_name}', fontsize=8)
                        except Exception as e:
                            axes[i, j].text(0.5, 0.5, 'Failed to load', 
                                           ha='center', va='center', transform=axes[i, j].transAxes)
                    else:
                        axes[i, j].text(0.5, 0.5, 'No image', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
                    
                    axes[i, j].axis('off')
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = output_dir / 'sample_images.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to plot sample images: {str(e)}")
            return None
    
    def generate_report(self, output_file: str) -> str:
        """
        Generate comprehensive dataset analysis report.
        
        Args:
            output_file: Path to output report file
            
        Returns:
            Path to generated report
        """
        try:
            self.logger.info("Generating comprehensive dataset report...")
            
            # Ensure all analyses are completed
            if 'class_distribution' not in self.analysis_results:
                self.analyze_class_distribution()
            
            if 'image_properties' not in self.analysis_results:
                self.analyze_image_properties()
            
            if not self.quality_metrics:
                self.analyze_data_quality()
            
            # Generate report
            report = {
                'dataset_summary': {
                    'total_images': len(self.image_paths),
                    'num_classes': self.num_classes,
                    'class_names': self.class_names
                },
                'class_distribution': self.analysis_results['class_distribution'],
                'image_properties': self.analysis_results['image_properties'],
                'data_quality': self.quality_metrics,
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Dataset analysis report generated: {output_file}")
            return output_file
            
        except Exception as e:
            raise DatasetError(f"Report generation failed: {str(e)}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Class imbalance recommendations
        if 'class_distribution' in self.analysis_results:
            imbalance_ratio = self.analysis_results['class_distribution']['imbalance_ratio']
            if imbalance_ratio > 5:
                recommendations.append(
                    f"High class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                    "Consider data augmentation or class weighting."
                )
        
        # Data quality recommendations
        if self.quality_metrics:
            quality_score = self.quality_metrics['quality_score']
            if quality_score < 0.9:
                recommendations.append(
                    f"Data quality score is {quality_score:.3f}. "
                    "Consider cleaning corrupted or duplicate files."
                )
        
        # Sample size recommendations
        if len(self.image_paths) < 1000:
            recommendations.append(
                "Dataset is relatively small. Consider data augmentation to improve model performance."
            )
        
        return recommendations