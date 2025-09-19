"""
Comprehensive Dataset Analysis for Fish Species Classification
============================================================

This script analyzes all datasets in the project:
1. FishImgDataset/ - Main dataset with 31 species
2. archive/ - Additional dataset with XML annotations
3. fishesdataser2/ - Additional dataset folder

Purpose: 
- Understand complete data structure
- Count images per species across all datasets
- Identify overlap and conflicts
- Prepare for unified training
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDatasetAnalyzer:
    """Analyze all fish datasets in the project"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.datasets = {
            'FishImgDataset': self.base_dir / 'FishImgDataset',
            'archive': self.base_dir / 'archive', 
            'fishesdataser2': self.base_dir / 'fishesdataser2'
        }
        self.results = {}
        self.unified_species = set()
        self.total_images = 0
        
    def analyze_fishimg_dataset(self):
        """Analyze the main FishImgDataset"""
        print("🔍 Analyzing FishImgDataset...")
        dataset_path = self.datasets['FishImgDataset']
        
        if not dataset_path.exists():
            print(f"❌ FishImgDataset not found at {dataset_path}")
            return
            
        analysis = {
            'type': 'classification_folders',
            'splits': {},
            'species': set(),
            'total_images': 0,
            'species_distribution': defaultdict(int)
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                split_analysis = self._analyze_classification_split(split_path)
                analysis['splits'][split] = split_analysis
                analysis['species'].update(split_analysis['species'])
                analysis['total_images'] += split_analysis['total_images']
                
                # Update species distribution
                for species, count in split_analysis['species_distribution'].items():
                    analysis['species_distribution'][species] += count
        
        self.results['FishImgDataset'] = analysis
        self.unified_species.update(analysis['species'])
        self.total_images += analysis['total_images']
        
        print(f"✅ FishImgDataset: {len(analysis['species'])} species, {analysis['total_images']} images")
        
    def analyze_archive_dataset(self):
        """Analyze the archive dataset with XML annotations"""
        print("🔍 Analyzing archive dataset...")
        dataset_path = self.datasets['archive']
        
        if not dataset_path.exists():
            print(f"❌ Archive dataset not found at {dataset_path}")
            return
            
        analysis = {
            'type': 'xml_annotations',
            'splits': {},
            'species': set(),
            'total_images': 0,
            'species_distribution': defaultdict(int),
            'xml_files': 0
        }
        
        # Analyze each split
        for split in ['train', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                split_analysis = self._analyze_xml_split(split_path)
                analysis['splits'][split] = split_analysis
                analysis['species'].update(split_analysis['species'])
                analysis['total_images'] += split_analysis['total_images']
                analysis['xml_files'] += split_analysis['xml_files']
                
                # Update species distribution
                for species, count in split_analysis['species_distribution'].items():
                    analysis['species_distribution'][species] += count
        
        self.results['archive'] = analysis
        self.unified_species.update(analysis['species'])
        self.total_images += analysis['total_images']
        
        print(f"✅ Archive dataset: {len(analysis['species'])} species, {analysis['total_images']} images, {analysis['xml_files']} XML files")
        
    def analyze_fishesdataser2(self):
        """Analyze fishesdataser2 dataset"""
        print("🔍 Analyzing fishesdataser2...")
        dataset_path = self.datasets['fishesdataser2']
        
        if not dataset_path.exists():
            print(f"❌ fishesdataser2 not found at {dataset_path}")
            return
            
        # Check if folder is empty
        files = list(dataset_path.glob('*'))
        if not files:
            print("⚠️ fishesdataser2 folder is empty")
            self.results['fishesdataser2'] = {
                'type': 'empty',
                'total_images': 0,
                'species': set()
            }
            return
            
        # If not empty, analyze the structure
        analysis = {
            'type': 'unknown_structure',
            'files': len(files),
            'folders': len([f for f in files if f.is_dir()]),
            'images': len([f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]),
            'total_images': 0,
            'species': set()
        }
        
        self.results['fishesdataser2'] = analysis
        print(f"✅ fishesdataser2: {analysis['files']} files, {analysis['folders']} folders")
        
    def _analyze_classification_split(self, split_path):
        """Analyze a classification dataset split (folders as classes)"""
        analysis = {
            'species': set(),
            'total_images': 0,
            'species_distribution': defaultdict(int),
            'image_extensions': Counter(),
            'corrupted_images': []
        }
        
        for species_folder in split_path.iterdir():
            if species_folder.is_dir():
                species_name = species_folder.name
                analysis['species'].add(species_name)
                
                # Count images in this species folder
                image_files = list(species_folder.glob('*'))
                valid_images = 0
                
                for img_file in image_files:
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        analysis['image_extensions'][img_file.suffix.lower()] += 1
                        
                        # Check if image can be opened
                        try:
                            with Image.open(img_file) as img:
                                img.verify()
                            valid_images += 1
                        except Exception:
                            analysis['corrupted_images'].append(str(img_file))
                
                analysis['species_distribution'][species_name] = valid_images
                analysis['total_images'] += valid_images
                
        return analysis
        
    def _analyze_xml_split(self, split_path):
        """Analyze a dataset split with XML annotations"""
        analysis = {
            'species': set(),
            'total_images': 0,
            'species_distribution': defaultdict(int),
            'xml_files': 0,
            'image_extensions': Counter(),
            'corrupted_images': [],
            'annotation_info': {}
        }
        
        # Get all files
        image_files = [f for f in split_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        xml_files = list(split_path.glob('*.xml'))
        
        analysis['xml_files'] = len(xml_files)
        
        # Extract species from filenames (e.g., "Catla-Body (1).JPG" -> "Catla")
        for img_file in image_files:
            filename = img_file.name
            
            # Extract species name (everything before the first '-' or '(' )
            if '-' in filename:
                species_name = filename.split('-')[0]
            elif '(' in filename:
                species_name = filename.split('(')[0].strip()
            else:
                species_name = img_file.stem
                
            analysis['species'].add(species_name)
            analysis['image_extensions'][img_file.suffix.lower()] += 1
            
            # Check if image can be opened
            try:
                with Image.open(img_file) as img:
                    img.verify()
                analysis['species_distribution'][species_name] += 1
                analysis['total_images'] += 1
            except Exception:
                analysis['corrupted_images'].append(str(img_file))
                
        return analysis
        
    def generate_unified_analysis(self):
        """Generate unified analysis across all datasets"""
        print("\n📊 Generating unified analysis...")
        
        unified_analysis = {
            'total_datasets': len([k for k, v in self.results.items() if v.get('total_images', 0) > 0]),
            'total_images': self.total_images,
            'total_species': len(self.unified_species),
            'species_list': sorted(list(self.unified_species)),
            'dataset_comparison': {},
            'species_overlap': {},
            'recommendations': []
        }
        
        # Compare datasets
        for dataset_name, dataset_info in self.results.items():
            if dataset_info.get('total_images', 0) > 0:
                unified_analysis['dataset_comparison'][dataset_name] = {
                    'images': dataset_info['total_images'],
                    'species': len(dataset_info.get('species', set())),
                    'type': dataset_info['type']
                }
        
        # Find species overlap
        fishimg_species = self.results.get('FishImgDataset', {}).get('species', set())
        archive_species = self.results.get('archive', {}).get('species', set())
        
        unified_analysis['species_overlap'] = {
            'fishimg_only': list(fishimg_species - archive_species),
            'archive_only': list(archive_species - fishimg_species),
            'common': list(fishimg_species & archive_species)
        }
        
        # Generate recommendations
        recommendations = []
        
        if len(fishimg_species) > 0 and len(archive_species) > 0:
            if len(unified_analysis['species_overlap']['common']) > 0:
                recommendations.append("⚠️ Species overlap detected between FishImgDataset and archive")
                recommendations.append("💡 Consider merging common species or keeping datasets separate")
        
        if self.results.get('fishesdataser2', {}).get('type') == 'empty':
            recommendations.append("ℹ️ fishesdataser2 is empty - can be ignored")
        
        if len(fishimg_species) >= 20:
            recommendations.append("✅ FishImgDataset has sufficient species diversity for training")
        else:
            recommendations.append("⚠️ Consider augmenting FishImgDataset with archive data")
            
        unified_analysis['recommendations'] = recommendations
        
        return unified_analysis
        
    def create_visualizations(self, unified_analysis):
        """Create comprehensive visualizations"""
        print("📈 Creating visualizations...")
        
        # Setup the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Dataset comparison
        ax1 = plt.subplot(2, 3, 1)
        datasets = list(unified_analysis['dataset_comparison'].keys())
        images = [unified_analysis['dataset_comparison'][d]['images'] for d in datasets]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(datasets)]
        
        bars = ax1.bar(datasets, images, color=colors)
        ax1.set_title('Dataset Image Count Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Images')
        ax1.set_xlabel('Dataset')
        
        # Add value labels on bars
        for bar, value in zip(bars, images):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(images)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Species count comparison
        ax2 = plt.subplot(2, 3, 2)
        species_counts = [unified_analysis['dataset_comparison'][d]['species'] for d in datasets]
        bars2 = ax2.bar(datasets, species_counts, color=colors)
        ax2.set_title('Dataset Species Count Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Species')
        ax2.set_xlabel('Dataset')
        
        # Add value labels
        for bar, value in zip(bars2, species_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(species_counts)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 3. Species overlap (if applicable)
        if 'archive' in unified_analysis['dataset_comparison'] and 'FishImgDataset' in unified_analysis['dataset_comparison']:
            ax3 = plt.subplot(2, 3, 3)
            overlap_data = unified_analysis['species_overlap']
            categories = ['FishImgDataset\nOnly', 'Common', 'Archive\nOnly']
            values = [len(overlap_data['fishimg_only']), len(overlap_data['common']), len(overlap_data['archive_only'])]
            
            wedges, texts, autotexts = ax3.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Species Distribution Across Datasets', fontsize=14, fontweight='bold')
        
        # 4. FishImgDataset species distribution (if available)
        if 'FishImgDataset' in self.results and 'species_distribution' in self.results['FishImgDataset']:
            ax4 = plt.subplot(2, 3, 4)
            species_dist = dict(self.results['FishImgDataset']['species_distribution'])
            
            if len(species_dist) > 0:
                # Get top 15 species for better visualization
                sorted_species = sorted(species_dist.items(), key=lambda x: x[1], reverse=True)[:15]
                species_names = [s[0] for s in sorted_species]
                species_counts = [s[1] for s in sorted_species]
                
                bars4 = ax4.barh(range(len(species_names)), species_counts, color='skyblue')
                ax4.set_yticks(range(len(species_names)))
                ax4.set_yticklabels(species_names, fontsize=10)
                ax4.set_xlabel('Number of Images')
                ax4.set_title('Top 15 Species in FishImgDataset', fontsize=14, fontweight='bold')
                ax4.invert_yaxis()
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars4, species_counts)):
                    ax4.text(bar.get_width() + max(species_counts)*0.01, bar.get_y() + bar.get_height()/2,
                            str(value), va='center', fontsize=9)
        
        # 5. Archive dataset species distribution (if available)
        if 'archive' in self.results and 'species_distribution' in self.results['archive']:
            ax5 = plt.subplot(2, 3, 5)
            archive_dist = dict(self.results['archive']['species_distribution'])
            
            if len(archive_dist) > 0:
                species_names = list(archive_dist.keys())
                species_counts = list(archive_dist.values())
                
                bars5 = ax5.bar(species_names, species_counts, color='lightcoral')
                ax5.set_title('Species Distribution in Archive Dataset', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Number of Images')
                ax5.set_xlabel('Species')
                ax5.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars5, species_counts):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(species_counts)*0.01,
                            str(value), ha='center', va='bottom', fontsize=10)
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
COMPREHENSIVE DATASET ANALYSIS SUMMARY

📊 Total Statistics:
• Total Datasets: {unified_analysis['total_datasets']}
• Total Images: {unified_analysis['total_images']:,}
• Total Species: {unified_analysis['total_species']}

📁 Dataset Breakdown:
"""
        
        for dataset, info in unified_analysis['dataset_comparison'].items():
            summary_text += f"• {dataset}: {info['images']} images, {info['species']} species\n"
        
        if unified_analysis['recommendations']:
            summary_text += f"\n💡 Key Recommendations:\n"
            for rec in unified_analysis['recommendations'][:3]:  # Show top 3
                summary_text += f"• {rec}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('comprehensive_dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'comprehensive_dataset_analysis.png'")
        
        return fig
        
    def save_results(self, unified_analysis):
        """Save comprehensive analysis results"""
        print("💾 Saving analysis results...")
        
        # Create detailed text report
        with open('comprehensive_dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE FISH DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Datasets Analyzed: {unified_analysis['total_datasets']}\n")
            f.write(f"Total Images: {unified_analysis['total_images']:,}\n")
            f.write(f"Total Unique Species: {unified_analysis['total_species']}\n\n")
            
            f.write("DATASET DETAILS\n")
            f.write("-" * 20 + "\n")
            for dataset_name, dataset_info in self.results.items():
                f.write(f"\n{dataset_name.upper()}:\n")
                f.write(f"  Type: {dataset_info.get('type', 'Unknown')}\n")
                f.write(f"  Images: {dataset_info.get('total_images', 0)}\n")
                f.write(f"  Species: {len(dataset_info.get('species', set()))}\n")
                
                if 'splits' in dataset_info:
                    f.write(f"  Splits: {', '.join(dataset_info['splits'].keys())}\n")
                
                if dataset_info.get('species_distribution'):
                    f.write(f"  Species Distribution:\n")
                    for species, count in sorted(dataset_info['species_distribution'].items()):
                        f.write(f"    - {species}: {count} images\n")
            
            if unified_analysis.get('species_overlap'):
                f.write(f"\nSPECIES OVERLAP ANALYSIS\n")
                f.write("-" * 20 + "\n")
                overlap = unified_analysis['species_overlap']
                f.write(f"FishImgDataset only: {len(overlap['fishimg_only'])} species\n")
                f.write(f"Archive only: {len(overlap['archive_only'])} species\n")
                f.write(f"Common species: {len(overlap['common'])} species\n")
                
                if overlap['common']:
                    f.write(f"\nCommon species: {', '.join(overlap['common'])}\n")
            
            f.write(f"\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for rec in unified_analysis['recommendations']:
                f.write(f"• {rec}\n")
            
            f.write(f"\nALL SPECIES IDENTIFIED\n")
            f.write("-" * 20 + "\n")
            for i, species in enumerate(sorted(unified_analysis['species_list']), 1):
                f.write(f"{i:2d}. {species}\n")
        
        # Create summary JSON (simplified)
        summary_json = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_datasets': unified_analysis['total_datasets'],
            'total_images': unified_analysis['total_images'],
            'total_species': unified_analysis['total_species'],
            'species_list': unified_analysis['species_list'],
            'dataset_summary': {}
        }
        
        for dataset_name, dataset_info in self.results.items():
            summary_json['dataset_summary'][dataset_name] = {
                'type': dataset_info.get('type', 'Unknown'),
                'images': dataset_info.get('total_images', 0),
                'species_count': len(dataset_info.get('species', set())),
                'species_list': list(dataset_info.get('species', set()))
            }
        
        with open('comprehensive_dataset_summary.json', 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        print("✅ Results saved to:")
        print("   - comprehensive_dataset_analysis_report.txt")
        print("   - comprehensive_dataset_summary.json")
        
    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        print("🚀 Starting comprehensive dataset analysis...\n")
        
        # Analyze each dataset
        self.analyze_fishimg_dataset()
        self.analyze_archive_dataset()
        self.analyze_fishesdataser2()
        
        # Generate unified analysis
        unified_analysis = self.generate_unified_analysis()
        
        # Create visualizations
        self.create_visualizations(unified_analysis)
        
        # Save results
        self.save_results(unified_analysis)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📈 Found {unified_analysis['total_species']} unique species across {unified_analysis['total_datasets']} datasets")
        print(f"📷 Total images: {unified_analysis['total_images']:,}")
        
        return unified_analysis

def main():
    """Main execution function"""
    analyzer = ComprehensiveDatasetAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET ANALYSIS COMPLETED")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = main()