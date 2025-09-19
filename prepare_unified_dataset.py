"""
Unified Training Dataset Preparation
===================================

This script prepares a unified training dataset from all available sources:
1. FishImgDataset (31 species, 13,301 images) - Primary dataset
2. Archive (4 species, 127 images) - Additional species to expand diversity

Strategy:
- Use FishImgDataset as the primary dataset (already well-structured)
- Add Archive species as new classes to expand the model's capability
- Create unified train/val/test splits
- Handle class imbalance with appropriate techniques
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class UnifiedDatasetPreparer:
    """Prepare unified training dataset from all sources"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "UnifiedFishDataset"
        
        # Dataset sources
        self.fishimg_dataset = self.base_dir / "FishImgDataset"
        self.archive_dataset = self.base_dir / "archive"
        
        # Unified dataset structure
        self.splits = ['train', 'val', 'test']
        self.unified_species = {}
        self.class_mapping = {}
        self.dataset_stats = {}
        
    def analyze_existing_structure(self):
        """Analyze existing dataset structures"""
        print("🔍 Analyzing existing dataset structures...")
        
        # Analyze FishImgDataset
        fishimg_stats = {
            'total_images': 0,
            'species_count': 0,
            'species_distribution': {},
            'splits': {}
        }
        
        if self.fishimg_dataset.exists():
            for split in self.splits:
                split_path = self.fishimg_dataset / split
                if split_path.exists():
                    split_stats = {'species': {}, 'total': 0}
                    
                    for species_dir in split_path.iterdir():
                        if species_dir.is_dir():
                            species_name = species_dir.name
                            images = list(species_dir.glob('*.jpg')) + list(species_dir.glob('*.jpeg')) + list(species_dir.glob('*.png'))
                            image_count = len(images)
                            
                            split_stats['species'][species_name] = image_count
                            split_stats['total'] += image_count
                            
                            if species_name not in fishimg_stats['species_distribution']:
                                fishimg_stats['species_distribution'][species_name] = 0
                            fishimg_stats['species_distribution'][species_name] += image_count
                    
                    fishimg_stats['splits'][split] = split_stats
            
            fishimg_stats['total_images'] = sum(fishimg_stats['species_distribution'].values())
            fishimg_stats['species_count'] = len(fishimg_stats['species_distribution'])
        
        # Analyze Archive dataset
        archive_stats = {
            'total_images': 0,
            'species_count': 0,
            'species_distribution': {},
            'splits': {}
        }
        
        if self.archive_dataset.exists():
            for split in ['train', 'test']:  # Archive only has train/test
                split_path = self.archive_dataset / split
                if split_path.exists():
                    split_stats = {'species': {}, 'total': 0}
                    
                    # Extract species from filenames
                    image_files = list(split_path.glob('*.jpg')) + list(split_path.glob('*.JPG'))
                    
                    for img_file in image_files:
                        # Extract species name from filename (e.g., "Catla-Body (1).JPG" -> "Catla")
                        filename = img_file.name
                        if '-' in filename:
                            species_name = filename.split('-')[0]
                        else:
                            species_name = filename.split('(')[0].strip()
                        
                        if species_name not in split_stats['species']:
                            split_stats['species'][species_name] = 0
                        split_stats['species'][species_name] += 1
                        split_stats['total'] += 1
                        
                        if species_name not in archive_stats['species_distribution']:
                            archive_stats['species_distribution'][species_name] = 0
                        archive_stats['species_distribution'][species_name] += 1
                    
                    archive_stats['splits'][split] = split_stats
            
            archive_stats['total_images'] = sum(archive_stats['species_distribution'].values())
            archive_stats['species_count'] = len(archive_stats['species_distribution'])
        
        self.dataset_stats = {
            'FishImgDataset': fishimg_stats,
            'Archive': archive_stats
        }
        
        print(f"✅ FishImgDataset: {fishimg_stats['species_count']} species, {fishimg_stats['total_images']} images")
        print(f"✅ Archive: {archive_stats['species_count']} species, {archive_stats['total_images']} images")
        
        return self.dataset_stats
        
    def create_unified_structure(self):
        """Create unified dataset directory structure"""
        print("📁 Creating unified dataset structure...")
        
        # Remove existing unified dataset if it exists
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        # Create new structure
        self.output_dir.mkdir(exist_ok=True)
        for split in self.splits:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        print(f"✅ Created unified dataset structure at: {self.output_dir}")
        
    def copy_fishimg_dataset(self):
        """Copy FishImgDataset maintaining existing splits"""
        print("📂 Copying FishImgDataset...")
        
        species_copied = set()
        total_copied = 0
        
        for split in self.splits:
            source_split = self.fishimg_dataset / split
            target_split = self.output_dir / split
            
            if source_split.exists():
                for species_dir in source_split.iterdir():
                    if species_dir.is_dir():
                        species_name = species_dir.name
                        target_species_dir = target_split / species_name
                        target_species_dir.mkdir(exist_ok=True)
                        
                        # Copy all images
                        images_copied = 0
                        for img_file in species_dir.glob('*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                target_file = target_species_dir / img_file.name
                                shutil.copy2(img_file, target_file)
                                images_copied += 1
                                total_copied += 1
                        
                        species_copied.add(species_name)
                        print(f"  📁 {split}/{species_name}: {images_copied} images")
        
        print(f"✅ Copied {len(species_copied)} species, {total_copied} images from FishImgDataset")
        return species_copied, total_copied
        
    def integrate_archive_dataset(self):
        """Integrate Archive dataset as new species"""
        print("📂 Integrating Archive dataset...")
        
        if not self.archive_dataset.exists():
            print("⚠️ Archive dataset not found, skipping...")
            return set(), 0
        
        # Collect all archive images by species
        archive_images = defaultdict(list)
        
        for split in ['train', 'test']:
            split_path = self.archive_dataset / split
            if split_path.exists():
                for img_file in split_path.glob('*.jpg') + split_path.glob('*.JPG'):
                    # Extract species name
                    filename = img_file.name
                    if '-' in filename:
                        species_name = filename.split('-')[0]
                    else:
                        species_name = filename.split('(')[0].strip()
                    
                    archive_images[species_name].append(img_file)
        
        # Create train/val/test splits for archive species
        species_added = set()
        total_added = 0
        
        for species_name, image_files in archive_images.items():
            print(f"  🐟 Processing {species_name}: {len(image_files)} images")
            
            # Create species directories
            for split in self.splits:
                species_dir = self.output_dir / split / species_name
                species_dir.mkdir(exist_ok=True)
            
            # Split images into train/val/test
            if len(image_files) >= 10:  # Enough for proper splitting
                train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
                val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
            elif len(image_files) >= 5:  # Small dataset
                train_files, test_files = train_test_split(image_files, test_size=0.3, random_state=42)
                val_files = test_files[:len(test_files)//2] if len(test_files) > 1 else []
                test_files = test_files[len(test_files)//2:]
            else:  # Very small dataset - put all in train
                train_files = image_files
                val_files = []
                test_files = []
            
            # Copy files
            splits_data = [('train', train_files), ('val', val_files), ('test', test_files)]
            
            for split_name, files in splits_data:
                target_dir = self.output_dir / split_name / species_name
                for i, img_file in enumerate(files):
                    # Create unique filename to avoid conflicts
                    target_file = target_dir / f"{species_name}_{split_name}_{i+1}{img_file.suffix}"
                    shutil.copy2(img_file, target_file)
                    total_added += 1
                
                print(f"    📁 {split_name}: {len(files)} images")
            
            species_added.add(species_name)
        
        print(f"✅ Added {len(species_added)} new species, {total_added} images from Archive")
        return species_added, total_added
        
    def create_class_mapping(self):
        """Create mapping from species names to class indices"""
        print("🔢 Creating class mapping...")
        
        all_species = set()
        
        # Collect all species from unified dataset
        for split in self.splits:
            split_dir = self.output_dir / split
            if split_dir.exists():
                for species_dir in split_dir.iterdir():
                    if species_dir.is_dir():
                        all_species.add(species_dir.name)
        
        # Create sorted mapping
        sorted_species = sorted(list(all_species))
        self.class_mapping = {species: idx for idx, species in enumerate(sorted_species)}
        
        print(f"✅ Created mapping for {len(sorted_species)} species")
        return self.class_mapping
        
    def generate_dataset_statistics(self):
        """Generate comprehensive statistics for the unified dataset"""
        print("📊 Generating unified dataset statistics...")
        
        stats = {
            'total_species': 0,
            'total_images': 0,
            'split_distribution': {},
            'species_distribution': {},
            'class_balance_analysis': {},
            'recommendations': []
        }
        
        for split in self.splits:
            split_dir = self.output_dir / split
            split_stats = {'species': {}, 'total': 0}
            
            if split_dir.exists():
                for species_dir in split_dir.iterdir():
                    if species_dir.is_dir():
                        species_name = species_dir.name
                        image_count = len(list(species_dir.glob('*')))
                        
                        split_stats['species'][species_name] = image_count
                        split_stats['total'] += image_count
                        
                        if species_name not in stats['species_distribution']:
                            stats['species_distribution'][species_name] = 0
                        stats['species_distribution'][species_name] += image_count
            
            stats['split_distribution'][split] = split_stats
        
        stats['total_species'] = len(stats['species_distribution'])
        stats['total_images'] = sum(stats['species_distribution'].values())
        
        # Class balance analysis
        if stats['species_distribution']:
            species_counts = list(stats['species_distribution'].values())
            stats['class_balance_analysis'] = {
                'min_images': min(species_counts),
                'max_images': max(species_counts),
                'mean_images': np.mean(species_counts),
                'median_images': np.median(species_counts),
                'std_images': np.std(species_counts),
                'imbalance_ratio': max(species_counts) / min(species_counts) if min(species_counts) > 0 else 0
            }
        
        # Generate recommendations
        recommendations = []
        
        if stats['class_balance_analysis'].get('imbalance_ratio', 0) > 10:
            recommendations.append("⚠️ High class imbalance detected - consider data augmentation")
        
        if stats['total_species'] > 30:
            recommendations.append("✅ Good species diversity for training")
        
        if stats['total_images'] > 10000:
            recommendations.append("✅ Sufficient data volume for deep learning")
        
        # Check split ratios
        if 'train' in stats['split_distribution'] and 'val' in stats['split_distribution']:
            train_ratio = stats['split_distribution']['train']['total'] / stats['total_images']
            val_ratio = stats['split_distribution']['val']['total'] / stats['total_images']
            
            if train_ratio < 0.6:
                recommendations.append("⚠️ Training set might be too small")
            if val_ratio < 0.1:
                recommendations.append("⚠️ Validation set might be too small")
        
        stats['recommendations'] = recommendations
        
        print(f"✅ Unified dataset: {stats['total_species']} species, {stats['total_images']} images")
        
        return stats
        
    def create_visualizations(self, stats):
        """Create visualizations for the unified dataset"""
        print("📈 Creating unified dataset visualizations...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Split distribution
        ax1 = plt.subplot(2, 3, 1)
        split_totals = [stats['split_distribution'][split]['total'] for split in self.splits if split in stats['split_distribution']]
        split_names = [split for split in self.splits if split in stats['split_distribution']]
        
        bars1 = ax1.bar(split_names, split_totals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Unified Dataset Split Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Images')
        
        # Add percentages
        total = sum(split_totals)
        for bar, value in zip(bars1, split_totals):
            percentage = (value / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{value}\\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 2. Top 20 species distribution
        ax2 = plt.subplot(2, 3, 2)
        sorted_species = sorted(stats['species_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]
        species_names = [s[0] for s in sorted_species]
        species_counts = [s[1] for s in sorted_species]
        
        bars2 = ax2.barh(range(len(species_names)), species_counts, color='skyblue')
        ax2.set_yticks(range(len(species_names)))
        ax2.set_yticklabels(species_names, fontsize=9)
        ax2.set_xlabel('Number of Images')
        ax2.set_title('Top 20 Species Distribution', fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Class balance visualization
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(list(stats['species_distribution'].values()), bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Images per Species')
        ax3.set_ylabel('Number of Species')
        ax3.set_title('Class Balance Distribution', fontweight='bold')
        ax3.axvline(stats['class_balance_analysis']['mean_images'], color='red', linestyle='--', label='Mean')
        ax3.axvline(stats['class_balance_analysis']['median_images'], color='blue', linestyle='--', label='Median')
        ax3.legend()
        
        # 4. Cumulative species coverage
        ax4 = plt.subplot(2, 3, 4)
        sorted_counts = sorted(stats['species_distribution'].values(), reverse=True)
        cumulative = np.cumsum(sorted_counts)
        cumulative_pct = (cumulative / cumulative[-1]) * 100
        
        ax4.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, marker='o', markersize=3)
        ax4.set_xlabel('Species Rank')
        ax4.set_ylabel('Cumulative Coverage (%)')
        ax4.set_title('Cumulative Species Coverage', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Dataset comparison (before/after integration)
        ax5 = plt.subplot(2, 3, 5)
        datasets = ['FishImgDataset', 'Archive', 'Unified']
        species_counts = [
            self.dataset_stats['FishImgDataset']['species_count'],
            self.dataset_stats['Archive']['species_count'],
            stats['total_species']
        ]
        image_counts = [
            self.dataset_stats['FishImgDataset']['total_images'],
            self.dataset_stats['Archive']['total_images'],
            stats['total_images']
        ]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars5a = ax5.bar(x - width/2, species_counts, width, label='Species', color='lightcoral')
        ax5_twin = ax5.twinx()
        bars5b = ax5_twin.bar(x + width/2, image_counts, width, label='Images', color='lightsalmon')
        
        ax5.set_xlabel('Dataset')
        ax5.set_ylabel('Number of Species', color='lightcoral')
        ax5_twin.set_ylabel('Number of Images', color='lightsalmon')
        ax5.set_title('Dataset Integration Results', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(datasets)
        
        # Add value labels
        for bar, value in zip(bars5a, species_counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(species_counts)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
UNIFIED DATASET SUMMARY

📊 Total Statistics:
• Species: {stats['total_species']}
• Images: {stats['total_images']:,}
• Splits: {len(stats['split_distribution'])}

📈 Class Balance:
• Min/Max: {stats['class_balance_analysis']['min_images']}/{stats['class_balance_analysis']['max_images']}
• Mean: {stats['class_balance_analysis']['mean_images']:.1f}
• Std: {stats['class_balance_analysis']['std_images']:.1f}
• Imbalance Ratio: {stats['class_balance_analysis']['imbalance_ratio']:.1f}

💡 Key Recommendations:
"""
        
        for rec in stats['recommendations'][:3]:
            summary_text += f"• {rec}\\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('unified_dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'unified_dataset_analysis.png'")
        
        return fig
        
    def save_unified_dataset_info(self, stats):
        """Save unified dataset information"""
        print("💾 Saving unified dataset information...")
        
        # Save comprehensive info
        unified_info = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'source_datasets': list(self.dataset_stats.keys()),
            'output_directory': str(self.output_dir),
            'class_mapping': self.class_mapping,
            'statistics': stats,
            'file_structure': {
                'train': str(self.output_dir / 'train'),
                'val': str(self.output_dir / 'val'),
                'test': str(self.output_dir / 'test')
            }
        }
        
        # Save JSON
        with open('unified_dataset_info.json', 'w') as f:
            json.dump(unified_info, f, indent=2)
        
        # Save class mapping separately for easy access
        with open('class_mapping.json', 'w') as f:
            json.dump(self.class_mapping, f, indent=2)
        
        # Create detailed report
        with open('unified_dataset_report.txt', 'w', encoding='utf-8') as f:
            f.write("UNIFIED FISH DATASET PREPARATION REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Creation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("OVERVIEW\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Total Species: {stats['total_species']}\\n")
            f.write(f"Total Images: {stats['total_images']:,}\\n")
            f.write(f"Output Directory: {self.output_dir}\\n\\n")
            
            f.write("SPLIT DISTRIBUTION\\n")
            f.write("-" * 20 + "\\n")
            for split, split_data in stats['split_distribution'].items():
                f.write(f"{split.upper()}: {split_data['total']} images ({len(split_data['species'])} species)\\n")
            f.write("\\n")
            
            f.write("SPECIES DISTRIBUTION\\n")
            f.write("-" * 20 + "\\n")
            for species, count in sorted(stats['species_distribution'].items()):
                f.write(f"{species}: {count} images\\n")
            f.write("\\n")
            
            f.write("CLASS MAPPING\\n")
            f.write("-" * 20 + "\\n")
            for species, idx in sorted(self.class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{idx:2d}: {species}\\n")
            f.write("\\n")
            
            f.write("RECOMMENDATIONS\\n")
            f.write("-" * 20 + "\\n")
            for rec in stats['recommendations']:
                f.write(f"• {rec}\\n")
        
        print("✅ Unified dataset info saved:")
        print("   - unified_dataset_info.json")
        print("   - class_mapping.json")
        print("   - unified_dataset_report.txt")
        
    def prepare_complete_unified_dataset(self):
        """Complete workflow to prepare unified dataset"""
        print("🚀 Starting unified dataset preparation...\\n")
        
        # Step 1: Analyze existing structures
        self.analyze_existing_structure()
        print()
        
        # Step 2: Create unified structure
        self.create_unified_structure()
        print()
        
        # Step 3: Copy FishImgDataset
        fishimg_species, fishimg_count = self.copy_fishimg_dataset()
        print()
        
        # Step 4: Integrate Archive dataset
        archive_species, archive_count = self.integrate_archive_dataset()
        print()
        
        # Step 5: Create class mapping
        self.create_class_mapping()
        print()
        
        # Step 6: Generate statistics
        stats = self.generate_dataset_statistics()
        print()
        
        # Step 7: Create visualizations
        self.create_visualizations(stats)
        print()
        
        # Step 8: Save information
        self.save_unified_dataset_info(stats)
        print()
        
        print("🎉 Unified dataset preparation complete!")
        print(f"📁 Dataset location: {self.output_dir}")
        print(f"🐟 Total species: {stats['total_species']}")
        print(f"📷 Total images: {stats['total_images']:,}")
        print()
        print("✅ Ready for model training!")
        
        return stats

def main():
    """Main execution function"""
    preparer = UnifiedDatasetPreparer()
    results = preparer.prepare_complete_unified_dataset()
    
    print("\\n" + "="*60)
    print("UNIFIED DATASET PREPARATION COMPLETED")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = main()