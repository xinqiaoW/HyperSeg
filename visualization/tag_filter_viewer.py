"""
Tag Filtering Interface for Manual Screening of Hyperspectral Data Masks

This interface displays:
1. Pseudocolor RGB image of the hyperspectral data
2. Spectral curve of the masked region
3. Spectral curves of the dilated region (after dilation)
4. Spectral curves of the eroded region (after erosion)

Provides navigation and labeling functionality for manual mask quality assessment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import tifffile
import json
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_dilation, binary_erosion
from datetime import datetime


class TagFilterViewer:
    def __init__(self, root, index_file, output_file=None):
        self.root = root
        self.root.title("Tag Filter Viewer - Manual Mask Screening")
        self.root.geometry("1400x950")

        # Load index data
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.samples = index_data['samples']
        self.index_file = index_file

        # Output file for results
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(index_file))[0]
            output_file = os.path.join(os.path.dirname(index_file), f"{base_name}_filter_results.json")
        self.output_file = output_file

        # Load existing results if available
        self.filter_results = self.load_results()

        # Current state
        self.current_sample_idx = 0
        self.current_label_idx = 0
        self.current_label = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Morphological parameters
        self.dilation_size = 5
        self.erosion_size = 3
        self.num_spectra = 100  # Number of spectra to sample for display

        # Data holders
        self.image = None
        self.gt = None
        self.available_labels = []

        # Setup UI
        self.setup_ui()

        # Find first sample with labels
        self.find_next_valid_sample(0)

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel (top)
        self.setup_control_panel(main_frame)

        # Display area (middle)
        self.setup_display_area(main_frame)

        # Status bar (bottom)
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.prev_mask())
        self.root.bind('<Right>', lambda e: self.next_mask())
        self.root.bind('<Up>', lambda e: self.prev_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('k', lambda e: self.keep_mask())
        self.root.bind('K', lambda e: self.keep_mask())
        self.root.bind('d', lambda e: self.discard_mask())
        self.root.bind('D', lambda e: self.discard_mask())
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def setup_control_panel(self, parent):
        """Setup the control panel with navigation and action buttons"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Image navigation
        img_nav_frame = ttk.LabelFrame(control_frame, text="Image Navigation")
        img_nav_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(img_nav_frame, text="Previous Image (Up)",
                   command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(img_nav_frame, text="Next Image (Down)",
                   command=self.next_image).pack(side=tk.LEFT, padx=2)
        self.sample_label = ttk.Label(img_nav_frame, text="Image: 0/0")
        self.sample_label.pack(side=tk.LEFT, padx=10)

        # Mask navigation
        mask_nav_frame = ttk.LabelFrame(control_frame, text="Mask Navigation")
        mask_nav_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(mask_nav_frame, text="Previous Mask (Left)",
                   command=self.prev_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(mask_nav_frame, text="Next Mask (Right)",
                   command=self.next_mask).pack(side=tk.LEFT, padx=2)
        self.mask_label = ttk.Label(mask_nav_frame, text="Mask: 0/0")
        self.mask_label.pack(side=tk.LEFT, padx=10)

        # Action buttons
        action_frame = ttk.LabelFrame(control_frame, text="Actions")
        action_frame.pack(side=tk.LEFT, padx=5)

        self.keep_btn = ttk.Button(action_frame, text="Keep Mask (K)",
                                   command=self.keep_mask)
        self.keep_btn.pack(side=tk.LEFT, padx=2)

        self.discard_btn = ttk.Button(action_frame, text="Discard Mask (D)",
                                      command=self.discard_mask)
        self.discard_btn.pack(side=tk.LEFT, padx=2)

        # Decision indicator
        self.decision_label = ttk.Label(action_frame, text="Decision: --",
                                        font=('TkDefaultFont', 10, 'bold'))
        self.decision_label.pack(side=tk.LEFT, padx=10)

        # Progress info
        progress_frame = ttk.LabelFrame(control_frame, text="Progress")
        progress_frame.pack(side=tk.LEFT, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="Labeled: 0")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(settings_frame, text="Dilation:").pack(side=tk.LEFT, padx=2)
        self.dilation_spinbox = ttk.Spinbox(settings_frame, from_=1, to=15, width=3,
                                            command=self.update_settings)
        self.dilation_spinbox.set(self.dilation_size)
        self.dilation_spinbox.pack(side=tk.LEFT, padx=2)

        ttk.Label(settings_frame, text="Erosion:").pack(side=tk.LEFT, padx=2)
        self.erosion_spinbox = ttk.Spinbox(settings_frame, from_=1, to=15, width=3,
                                           command=self.update_settings)
        self.erosion_spinbox.set(self.erosion_size)
        self.erosion_spinbox.pack(side=tk.LEFT, padx=2)

        ttk.Button(settings_frame, text="Update",
                   command=self.update_displays).pack(side=tk.LEFT, padx=2)

    def setup_display_area(self, parent):
        """Setup the 2x2 display area for images and spectral curves"""
        display_frame = ttk.Frame(parent)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Configure grid
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)

        # Panel 1: Pseudocolor RGB Image (top-left)
        rgb_frame = ttk.LabelFrame(display_frame, text="Pseudocolor RGB (Bands 40, 20, 10)")
        rgb_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.rgb_label = ttk.Label(rgb_frame)
        self.rgb_label.pack(fill=tk.BOTH, expand=True)

        # Panel 2: Masked Region Spectra (top-right)
        mask_spectra_frame = ttk.LabelFrame(display_frame, text="Masked Region Spectra")
        mask_spectra_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.fig_mask, self.ax_mask = plt.subplots(figsize=(5, 4), dpi=80)
        self.canvas_mask = FigureCanvasTkAgg(self.fig_mask, master=mask_spectra_frame)
        self.canvas_mask.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Panel 3: Dilated Region Spectra (bottom-left)
        dilate_spectra_frame = ttk.LabelFrame(display_frame, text="Dilated Region Spectra (Outer Boundary)")
        dilate_spectra_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.fig_dilate, self.ax_dilate = plt.subplots(figsize=(5, 4), dpi=80)
        self.canvas_dilate = FigureCanvasTkAgg(self.fig_dilate, master=dilate_spectra_frame)
        self.canvas_dilate.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Panel 4: Eroded Region Spectra (bottom-right)
        erode_spectra_frame = ttk.LabelFrame(display_frame, text="Eroded Region Spectra (Inner Region)")
        erode_spectra_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.fig_erode, self.ax_erode = plt.subplots(figsize=(5, 4), dpi=80)
        self.canvas_erode = FigureCanvasTkAgg(self.fig_erode, master=erode_spectra_frame)
        self.canvas_erode.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_results(self):
        """Load existing filter results from file"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    return data.get('results', {})
            except:
                return {}
        return {}

    def save_results(self):
        """Save filter results to file"""
        output_data = {
            'index_file': self.index_file,
            'created_at': datetime.now().isoformat(),
            'results': self.filter_results
        }
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        self.update_progress_label()

    def get_result_key(self, sample_idx, label):
        """Generate a unique key for a sample-label pair"""
        sample = self.samples[sample_idx]
        img_name = os.path.basename(sample['img_path'])
        return f"{img_name}_{label}"

    def update_settings(self):
        """Update morphological operation settings"""
        try:
            self.dilation_size = int(self.dilation_spinbox.get())
            self.erosion_size = int(self.erosion_spinbox.get())
        except:
            pass

    def find_next_valid_sample(self, start_idx):
        """Find the next sample that has valid labels"""
        for idx in range(start_idx, len(self.samples)):
            sample = self.samples[idx]
            if not os.path.exists(sample['gt_path']):
                continue
            gt = np.load(sample['gt_path'])
            labels = [int(l) for l in np.unique(gt) if l > 0]
            if len(labels) > 0:
                self.load_sample(idx)
                return True
        return False

    def find_prev_valid_sample(self, start_idx):
        """Find the previous sample that has valid labels"""
        for idx in range(start_idx, -1, -1):
            sample = self.samples[idx]
            if not os.path.exists(sample['gt_path']):
                continue
            gt = np.load(sample['gt_path'])
            labels = [int(l) for l in np.unique(gt) if l > 0]
            if len(labels) > 0:
                self.load_sample(idx)
                return True
        return False

    def load_sample(self, idx):
        """Load a sample from the index"""
        if idx < 0 or idx >= len(self.samples):
            return False

        self.current_sample_idx = idx
        sample = self.samples[idx]

        img_path = sample['img_path']
        gt_path = sample['gt_path']

        # Check if files exist
        if not os.path.exists(img_path):
            self.status_label.config(text=f"Error: Image not found: {img_path}")
            return False
        if not os.path.exists(gt_path):
            self.status_label.config(text=f"Error: GT not found: {gt_path}")
            return False

        try:
            # Load image
            image_np = tifffile.imread(img_path).transpose(2, 0, 1)  # [C, H, W]
            self.image = torch.tensor(image_np.astype(np.float32))
            self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min() + 1e-8)

            # Load GT
            gt_np = np.load(gt_path)
            self.gt = torch.tensor(gt_np)

            # Find available labels (excluding 0 which is background)
            unique_labels = torch.unique(self.gt)
            self.available_labels = [int(l) for l in unique_labels if l > 0]
            self.available_labels.sort()

            if len(self.available_labels) == 0:
                self.status_label.config(text="Warning: No labels found in GT")
                return False

            self.current_label_idx = 0
            self.current_label = self.available_labels[0]

            # Update displays
            self.update_displays()
            self.update_labels()
            self.status_label.config(text=f"Loaded: {os.path.basename(img_path)}")
            return True

        except Exception as e:
            self.status_label.config(text=f"Error loading sample: {str(e)}")
            return False

    def update_displays(self):
        """Update all visual displays"""
        self.update_rgb_image()
        self.update_mask_spectra()
        self.update_dilated_spectra()
        self.update_eroded_spectra()
        self.update_decision_indicator()

    def update_labels(self):
        """Update navigation labels"""
        self.sample_label.config(text=f"Image: {self.current_sample_idx+1}/{len(self.samples)}")
        self.mask_label.config(text=f"Mask: {self.current_label_idx+1}/{len(self.available_labels)} (Label {self.current_label})")
        self.update_progress_label()

    def update_progress_label(self):
        """Update the progress label"""
        total_labeled = len(self.filter_results)
        kept = sum(1 for v in self.filter_results.values() if v.get('decision') == 'keep')
        discarded = sum(1 for v in self.filter_results.values() if v.get('decision') == 'discard')
        self.progress_label.config(text=f"Labeled: {total_labeled} (Keep: {kept}, Discard: {discarded})")

    def update_decision_indicator(self):
        """Update the decision indicator for current mask"""
        key = self.get_result_key(self.current_sample_idx, self.current_label)
        if key in self.filter_results:
            decision = self.filter_results[key].get('decision', '--')
            if decision == 'keep':
                self.decision_label.config(text="Decision: KEEP", foreground='green')
            elif decision == 'discard':
                self.decision_label.config(text="Decision: DISCARD", foreground='red')
            else:
                self.decision_label.config(text="Decision: --", foreground='black')
        else:
            self.decision_label.config(text="Decision: --", foreground='black')

    def update_rgb_image(self):
        """Update pseudocolor RGB image with mask overlay"""
        if self.image is None:
            return

        C, H, W = self.image.shape

        # Select bands for RGB (0-indexed: 40->39, 20->19, 10->9)
        bands = [min(39, C-1), min(19, C-1), min(9, C-1)]

        rgb = torch.stack([self.image[bands[0]], self.image[bands[1]], self.image[bands[2]]], dim=2)
        rgb = rgb.cpu().numpy()

        # Apply quantile normalization for better visualization
        low = np.percentile(rgb, 2)
        high = np.percentile(rgb, 98)
        rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)

        # Create mask overlay
        mask = (self.gt == self.current_label).cpu().numpy()

        # Create colored overlay (red for mask boundary)
        overlay = rgb.copy()

        # Highlight mask region with semi-transparent red
        mask_bool = mask.astype(bool)
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 80, 0, 255)
        overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1].astype(int) - 30, 0, 255)
        overlay[mask_bool, 2] = np.clip(overlay[mask_bool, 2].astype(int) - 30, 0, 255)

        # Draw mask boundary in bright red
        from scipy.ndimage import binary_erosion as scipy_erosion
        boundary = mask_bool & ~scipy_erosion(mask_bool, iterations=2)
        overlay[boundary, 0] = 255
        overlay[boundary, 1] = 0
        overlay[boundary, 2] = 0

        # Resize for display
        display_size = (400, 400)
        pil_img = Image.fromarray(overlay)
        pil_img = pil_img.resize(display_size, Image.NEAREST)

        photo = ImageTk.PhotoImage(pil_img)
        self.rgb_label.config(image=photo)
        self.rgb_label.image = photo

    def sample_spectra(self, mask, num_samples=None):
        """Sample spectra from a mask region"""
        if num_samples is None:
            num_samples = self.num_spectra

        mask_indices = torch.nonzero(mask, as_tuple=False)

        if len(mask_indices) == 0:
            return None

        num_samples = min(num_samples, len(mask_indices))
        sampled_indices = mask_indices[torch.randperm(len(mask_indices))[:num_samples]]

        spectra = []
        for idx in sampled_indices:
            y, x = idx[0].item(), idx[1].item()
            spectrum = self.image[:, y, x].cpu().numpy()
            spectra.append(spectrum)

        return np.array(spectra)

    def plot_spectra(self, ax, canvas, spectra, title, color='blue'):
        """Plot spectra on given axes"""
        ax.clear()

        if spectra is None or len(spectra) == 0:
            ax.text(0.5, 0.5, 'No pixels in region',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot individual spectra with transparency
            for spectrum in spectra:
                ax.plot(spectrum, linewidth=0.5, color=color, alpha=0.3)

            # Plot mean spectrum
            mean_spectrum = np.mean(spectra, axis=0)
            ax.plot(mean_spectrum, linewidth=2, color='red', label='Mean')

            ax.set_xlabel('Band Index', fontsize=9)
            ax.set_ylabel('Reflectance', fontsize=9)
            ax.set_title(f'{title} (n={len(spectra)})', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        ax.figure.tight_layout()
        canvas.draw()

    def update_mask_spectra(self):
        """Update spectral plot for the original masked region"""
        if self.image is None or self.gt is None:
            return

        mask = (self.gt == self.current_label)
        spectra = self.sample_spectra(mask)
        self.plot_spectra(self.ax_mask, self.canvas_mask, spectra,
                         "Original Mask", color='blue')

    def update_dilated_spectra(self):
        """Update spectral plot for the dilated region (outer boundary)"""
        if self.image is None or self.gt is None:
            return

        mask = (self.gt == self.current_label).cpu().numpy()

        # Dilate the mask
        struct_element = np.ones((2 * self.dilation_size + 1, 2 * self.dilation_size + 1))
        mask_dilated = binary_dilation(mask, structure=struct_element)

        # Outer boundary = dilated - original
        outer_boundary = mask_dilated & (~mask)
        outer_boundary_tensor = torch.tensor(outer_boundary)

        spectra = self.sample_spectra(outer_boundary_tensor)
        self.plot_spectra(self.ax_dilate, self.canvas_dilate, spectra,
                         f"Dilated Region (k={self.dilation_size})", color='green')

    def update_eroded_spectra(self):
        """Update spectral plot for the eroded region (inner region)"""
        if self.image is None or self.gt is None:
            return

        mask = (self.gt == self.current_label).cpu().numpy()

        # Erode the mask
        struct_element = np.ones((2 * self.erosion_size + 1, 2 * self.erosion_size + 1))
        mask_eroded = binary_erosion(mask, structure=struct_element)

        # Inner region is the eroded mask
        inner_region_tensor = torch.tensor(mask_eroded)

        spectra = self.sample_spectra(inner_region_tensor)
        self.plot_spectra(self.ax_erode, self.canvas_erode, spectra,
                         f"Eroded Region (k={self.erosion_size})", color='purple')

    # Navigation methods
    def prev_image(self):
        """Navigate to previous image with labels"""
        if self.current_sample_idx > 0:
            if not self.find_prev_valid_sample(self.current_sample_idx - 1):
                self.status_label.config(text="No previous valid sample found")
        else:
            self.status_label.config(text="Already at first sample")

    def next_image(self):
        """Navigate to next image with labels"""
        if self.current_sample_idx < len(self.samples) - 1:
            if not self.find_next_valid_sample(self.current_sample_idx + 1):
                self.status_label.config(text="No more valid samples found")
        else:
            self.status_label.config(text="Already at last sample")

    def prev_mask(self):
        """Navigate to previous mask in current image"""
        if len(self.available_labels) == 0:
            return

        if self.current_label_idx > 0:
            self.current_label_idx -= 1
        else:
            # Go to previous image's last mask
            if self.current_sample_idx > 0:
                old_idx = self.current_sample_idx
                if self.find_prev_valid_sample(self.current_sample_idx - 1):
                    self.current_label_idx = len(self.available_labels) - 1
                    self.current_label = self.available_labels[self.current_label_idx]

        self.current_label = self.available_labels[self.current_label_idx]
        self.update_displays()
        self.update_labels()

    def next_mask(self):
        """Navigate to next mask in current image"""
        if len(self.available_labels) == 0:
            return

        if self.current_label_idx < len(self.available_labels) - 1:
            self.current_label_idx += 1
            self.current_label = self.available_labels[self.current_label_idx]
        else:
            # Go to next image's first mask
            if self.current_sample_idx < len(self.samples) - 1:
                self.find_next_valid_sample(self.current_sample_idx + 1)
                return

        self.update_displays()
        self.update_labels()

    # Action methods
    def keep_mask(self):
        """Mark current mask as 'keep'"""
        self.record_decision('keep')
        self.decision_label.config(text="Decision: KEEP", foreground='green')
        self.status_label.config(text=f"Marked as KEEP: {self.get_result_key(self.current_sample_idx, self.current_label)}")
        # Auto-advance to next mask
        self.next_mask()

    def discard_mask(self):
        """Mark current mask as 'discard'"""
        self.record_decision('discard')
        self.decision_label.config(text="Decision: DISCARD", foreground='red')
        self.status_label.config(text=f"Marked as DISCARD: {self.get_result_key(self.current_sample_idx, self.current_label)}")
        # Auto-advance to next mask
        self.next_mask()

    def record_decision(self, decision):
        """Record the decision for current mask"""
        key = self.get_result_key(self.current_sample_idx, self.current_label)
        sample = self.samples[self.current_sample_idx]

        self.filter_results[key] = {
            'img_path': sample['img_path'],
            'gt_path': sample['gt_path'],
            'img_name': os.path.basename(sample['img_path']),
            'label': self.current_label,
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        self.save_results()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tag Filter Viewer for Manual Mask Screening')
    parser.add_argument('--index_file', type=str,
                        default='../data/hyperfree_index_labels_hf_nms.json',
                        help='Path to index JSON file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output JSON file for filter results')
    args = parser.parse_args()

    # Resolve relative path
    if not os.path.isabs(args.index_file):
        args.index_file = os.path.join(os.path.dirname(__file__), args.index_file)

    if not os.path.exists(args.index_file):
        print(f"Error: Index file not found: {args.index_file}")
        return

    root = tk.Tk()
    app = TagFilterViewer(root, args.index_file, args.output_file)
    root.mainloop()


if __name__ == '__main__':
    main()
