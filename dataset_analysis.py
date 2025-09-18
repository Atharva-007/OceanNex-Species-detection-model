"""
Fish Species Dataset Analysis and Model Training
===============================================

A comprehensive analysis of the fish species dataset with CNN model training.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

def analyze_dataset():
    """Analyze the fish species dataset structure and distribution"""
    
    print("🐟 FISH SPECIES DATASET ANALYSIS")
    print("=" * 60)
    
    dataset_path = "FishImgDataset"
    splits = ['train', 'val', 'test']
    results = {}
    
    # Analyze each split
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split} directory not found")
            continue
        
        species_counts = {}
        total_images = 0
        
        species_dirs = [d for d in os.listdir(split_path) 
                       if os.path.isdir(os.path.join(split_path, d))]
        
        for species in sorted(species_dirs):
            species_path = os.path.join(split_path, species)
            image_files = [f for f in os.listdir(species_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(image_files)
            species_counts[species] = count
            total_images += count
        
        results[split] = species_counts
        print(f"{split.upper()} SET: {total_images:,} images across {len(species_dirs)} species")
    
    # Create detailed breakdown
    if 'train' in results:
        species_list = sorted(list(results['train'].keys()))
        print(f"\n📊 DATASET BREAKDOWN ({len(species_list)} species):")
        print("-" * 80)
        print(f"{'Species':<30} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'%':<6}")
        print("-" * 80)
        
        grand_total = sum(sum(split_data.values()) for split_data in results.values())
        
        for species in species_list:
            train_count = results.get('train', {}).get(species, 0)
            val_count = results.get('val', {}).get(species, 0)
            test_count = results.get('test', {}).get(species, 0)
            total = train_count + val_count + test_count
            percentage = (total / grand_total * 100) if grand_total > 0 else 0
            
            print(f"{species:<30} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8} {percentage:5.1f}%")
    
    # Create visualization
    create_visualizations(results)
    
    return results

def create_visualizations(results):
    """Create comprehensive visualizations of the dataset"""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Overall split distribution
    plt.subplot(2, 3, 1)
    splits_total = {split: sum(counts.values()) for split, counts in results.items()}
    bars = plt.bar(splits_total.keys(), splits_total.values(), 
                   color=['#2E86AB', '#A23B72', '#F18F01'])
    plt.title('Images per Dataset Split', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Train set distribution
    plt.subplot(2, 3, 2)
    if 'train' in results:
        train_counts = list(results['train'].values())
        plt.hist(train_counts, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(train_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(train_counts):.1f}')
        plt.axvline(np.median(train_counts), color='blue', linestyle='--', 
                   label=f'Median: {np.median(train_counts):.1f}')
        plt.title('Distribution of Training Images per Species')
        plt.xlabel('Number of Images per Species')
        plt.ylabel('Number of Species')
        plt.legend()
    
    # Top 10 species by total images
    plt.subplot(2, 3, 3)
    if 'train' in results:
        species_totals = {}
        for species in results['train'].keys():
            total = sum(results[split].get(species, 0) for split in results.keys())
            species_totals[species] = total
        
        top_species = sorted(species_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        species_names, counts = zip(*top_species)
        
        plt.barh(range(len(species_names)), counts, color='lightgreen')
        plt.yticks(range(len(species_names)), [name.replace(' ', '\\n') for name in species_names])
        plt.title('Top 10 Species by Total Images')
        plt.xlabel('Total Images')
        plt.gca().invert_yaxis()
    
    # Dataset split percentages
    plt.subplot(2, 3, 4)
    total_images = sum(splits_total.values())
    percentages = [count/total_images*100 for count in splits_total.values()]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    plt.pie(percentages, labels=splits_total.keys(), autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Dataset Split Distribution')
    
    # Class balance analysis
    plt.subplot(2, 3, 5)
    if 'train' in results:
        train_counts = np.array(list(results['train'].values()))
        
        # Statistics
        stats_text = f"""Dataset Statistics:
        
Total Species: {len(train_counts)}
Total Images: {sum(sum(split_data.values()) for split_data in results.values()):,}

Training Set:
• Mean per class: {np.mean(train_counts):.1f}
• Std deviation: {np.std(train_counts):.1f}
• Min images: {np.min(train_counts)}
• Max images: {np.max(train_counts)}
• Class imbalance ratio: {np.max(train_counts)/np.min(train_counts):.1f}:1"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        plt.title('Dataset Statistics')
        plt.axis('off')
    
    # Sample species names
    plt.subplot(2, 3, 6)
    if 'train' in results:
        species_list = list(results['train'].keys())
        sample_species = species_list[:15]  # Show first 15
        
        species_text = "Fish Species in Dataset:\\n\\n" + "\\n".join(f"• {species}" for species in sample_species)
        if len(species_list) > 15:
            species_text += f"\\n\\n... and {len(species_list) - 15} more species"
        
        plt.text(0.1, 0.9, species_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        plt.title('Species List (Sample)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fish_dataset_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n📈 Visualization saved as: fish_dataset_comprehensive_analysis.png")

def get_species_sample_info():
    """Get detailed information about sample images from each species"""
    
    dataset_path = "FishImgDataset"
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        print("Training directory not found")
        return
    
    print(f"\\n🔍 SAMPLE IMAGE ANALYSIS")
    print("-" * 50)
    
    species_dirs = sorted([d for d in os.listdir(train_path) 
                          if os.path.isdir(os.path.join(train_path, d))])
    
    sample_info = []
    
    for species in species_dirs[:5]:  # Analyze first 5 species
        species_path = os.path.join(train_path, species)
        image_files = [f for f in os.listdir(species_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            # Get info about first image
            first_image = image_files[0]
            image_path = os.path.join(species_path, first_image)
            
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
                    file_size = os.path.getsize(image_path)
                    
                    sample_info.append({
                        'Species': species,
                        'Sample Image': first_image,
                        'Dimensions': f"{width}x{height}",
                        'Mode': mode,
                        'File Size (KB)': round(file_size / 1024, 1)
                    })
                    
                    print(f"{species:<25}: {first_image:<20} | {width:>4}x{height:<4} | {mode:<4} | {file_size/1024:>6.1f} KB")
            except Exception as e:
                print(f"{species:<25}: Error reading sample image - {e}")
    
    return sample_info

def save_analysis_report(results):
    """Save comprehensive analysis report"""
    
    # Calculate summary statistics
    total_images = sum(sum(split_data.values()) for split_data in results.values())
    num_species = len(results.get('train', {}))
    
    splits_summary = {}
    for split, data in results.items():
        splits_summary[split] = {
            'total_images': sum(data.values()),
            'species_count': len(data),
            'avg_per_species': np.mean(list(data.values())),
            'std_per_species': np.std(list(data.values())),
            'min_per_species': min(data.values()) if data else 0,
            'max_per_species': max(data.values()) if data else 0
        }
    
    report = {
        'dataset_overview': {
            'total_images': total_images,
            'total_species': num_species,
            'splits': list(results.keys())
        },
        'split_details': splits_summary,
        'species_distribution': results,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to JSON
    with open('fish_dataset_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n💾 Analysis report saved to: fish_dataset_analysis_report.json")
    
    return report

def main():
    """Main analysis function"""
    print("Starting Fish Species Dataset Analysis...")
    print("This analysis will examine the structure, distribution, and characteristics of your fish dataset.\\n")
    
    try:
        # Run dataset analysis
        results = analyze_dataset()
        
        # Get sample image information
        sample_info = get_species_sample_info()
        
        # Save comprehensive report
        report = save_analysis_report(results)
        
        print("\\n" + "="*60)
        print("📋 ANALYSIS SUMMARY")
        print("="*60)
        print(f"✅ Dataset successfully analyzed")
        print(f"✅ Found {len(results.get('train', {}))} fish species")
        print(f"✅ Total images: {sum(sum(split_data.values()) for split_data in results.values()):,}")
        print(f"✅ Visualization created: fish_dataset_comprehensive_analysis.png")
        print(f"✅ Report saved: fish_dataset_analysis_report.json")
        
        print(f"\\n🎯 NEXT STEPS:")
        print(f"   • Review the generated visualization")
        print(f"   • Check for class imbalance issues")
        print(f"   • Consider data augmentation strategies")
        print(f"   • Proceed with model training")
        
        return results, report
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Please check that the FishImgDataset directory exists and contains the expected structure.")
        return None, None

if __name__ == "__main__":
    results, report = main()