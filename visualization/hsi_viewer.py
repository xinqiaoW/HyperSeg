import sys
# sys.path.append("..")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import tifffile
import json
import os
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import data_filter


class HSIViewer:
    def __init__(self, root, index_file):
        self.root = root
        self.root.title("Hyperspectral Image Viewer")
        self.root.geometry("1200x900")
        
        # Load index data
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.samples = index_data['samples']
        
        # Current state
        self.current_sample_idx = 0
        self.current_label = 1
        self.num_spectra = 1000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Data holders
        self.image = None
        self.gt = None
        self.available_labels = []
        self.clicked_point = None  # Store clicked point coordinates (x, y)
        
        # Setup UI
        self.setup_ui()
        self.load_sample(0)
        
    def setup_ui(self):
        # Control panel (top)
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Sample navigation
        nav_frame = ttk.LabelFrame(control_frame, text="Sample Navigation")
        nav_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(nav_frame, text="Previous Image", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=2)
        self.sample_label = ttk.Label(nav_frame, text="Sample: 0/0")
        self.sample_label.pack(side=tk.LEFT, padx=10)
        
        # Label navigation
        label_frame = ttk.LabelFrame(control_frame, text="Label Navigation")
        label_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(label_frame, text="Previous Label", command=self.prev_label).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_frame, text="Next Label", command=self.next_label).pack(side=tk.LEFT, padx=2)
        self.label_text = ttk.Label(label_frame, text="Label: 1")
        self.label_text.pack(side=tk.LEFT, padx=10)
        
        # Spectra number setting
        spectra_frame = ttk.LabelFrame(control_frame, text="Spectra Settings")
        spectra_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(spectra_frame, text="Number of Spectra:").pack(side=tk.LEFT, padx=2)
        self.spectra_spinbox = ttk.Spinbox(spectra_frame, from_=1, to=100, width=5,
                                           command=self.update_spectra_num)
        self.spectra_spinbox.set(self.num_spectra)
        self.spectra_spinbox.pack(side=tk.LEFT, padx=2)
        ttk.Button(spectra_frame, text="Update Spectra", command=self.update_spectra_plot).pack(side=tk.LEFT, padx=2)
        
        # Metrics buttons
        metrics_frame = ttk.LabelFrame(control_frame, text="Metrics")
        metrics_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(metrics_frame, text="Spectral Purity", command=self.compute_spectral_purity).pack(side=tk.LEFT, padx=2)
        ttk.Button(metrics_frame, text="Edge Consistency", command=self.compute_edge_consistency).pack(side=tk.LEFT, padx=2)
        ttk.Button(metrics_frame, text="Spectral Variance", command=self.compute_spectral_variance).pack(side=tk.LEFT, padx=2)
        
        # Main display area
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Images
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # RGB image
        rgb_frame = ttk.LabelFrame(images_frame, text="False Color Image (Band 40, 20, 10) - Click to see spectrum")
        rgb_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.rgb_label = ttk.Label(rgb_frame)
        self.rgb_label.pack(fill=tk.BOTH, expand=True)
        # Bind click event
        self.rgb_label.bind("<Button-1>", self.on_image_click)
        
        # Mask image
        mask_frame = ttk.LabelFrame(images_frame, text="Current Label Mask")
        mask_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.mask_label = ttk.Label(mask_frame)
        self.mask_label.pack(fill=tk.BOTH, expand=True)
        
        # Right: Plots area
        plots_frame = ttk.Frame(display_frame)
        plots_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0))
        
        # Sampled Spectra plot
        spectra_frame = ttk.LabelFrame(plots_frame, text="Sampled Spectra")
        spectra_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        
        # Adjust figure size to approximately 280x280 pixels
        self.fig, self.ax = plt.subplots(figsize=(3.5, 3.5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, master=spectra_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Clicked Point Spectral Curve plot
        point_spectra_frame = ttk.LabelFrame(plots_frame, text="Clicked Point Spectrum")
        point_spectra_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        
        self.fig_point, self.ax_point = plt.subplots(figsize=(3.5, 3.5), dpi=80)
        self.canvas_point = FigureCanvasTkAgg(self.fig_point, master=point_spectra_frame)
        self.canvas_point.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty plot
        self.ax_point.text(0.5, 0.5, 'Click on image to view spectrum', 
                          ha='center', va='center', transform=self.ax_point.transAxes)
        self.canvas_point.draw()
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_sample(self, idx):
        """Load a sample from the index"""
        if idx < 0 or idx >= len(self.samples):
            return
        
        self.current_sample_idx = idx
        sample = self.samples[idx]
        
        img_path = sample['img_path']
        gt_path = sample['gt_path']
        
        # Check if files exist
        if not os.path.exists(img_path):
            self.status_label.config(text=f"Error: Image not found: {img_path}")
            return
        if not os.path.exists(gt_path):
            self.status_label.config(text=f"Error: GT not found: {gt_path}")
            return
        
        try:
            # Load image
            image_np = tifffile.imread(img_path).transpose(2, 0, 1)  # [C, H, W]
            self.image = torch.tensor(image_np.astype(np.float32))
            self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
            
            # Load GT
            gt_np = np.load(gt_path)
            self.gt = torch.tensor(gt_np)
            
            # Find available labels (excluding 0 which is background)
            unique_labels = torch.unique(self.gt)
            self.available_labels = [int(l) for l in unique_labels if l > 0]
            self.available_labels.sort()
            
            if len(self.available_labels) == 0:
                self.status_label.config(text="Warning: No labels found in GT")
                self.current_label = 0
            else:
                self.current_label = self.available_labels[0]
            
            # Update displays
            self.update_all_displays()
            self.sample_label.config(text=f"Sample: {idx+1}/{len(self.samples)}")
            self.label_text.config(text=f"Label: {self.current_label} (of {self.available_labels})")
            self.status_label.config(text=f"Loaded: {os.path.basename(img_path)}")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading sample: {str(e)}")
            
    def update_all_displays(self):
        """Update all visual displays"""
        self.update_rgb_image()
        self.update_mask_image()
        self.update_spectra_plot()
        # Clear clicked point when changing samples/labels
        self.clicked_point = None
        self.clear_point_spectrum()
        
    def update_rgb_image(self):
        """Update false color RGB image using bands 40, 20, 10"""
        if self.image is None:
            return
        
        C, H, W = self.image.shape
        
        # Select bands (handle if fewer bands available)
        bands = [min(39, C-1), min(19, C-1), min(9, C-1)]  # 0-indexed: 40->39, 20->19, 10->9
        
        rgb = torch.stack([self.image[bands[0]], self.image[bands[1]], self.image[bands[2]]], dim=2)
        rgb = rgb.cpu().numpy()
        
        # Normalize for display
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = (rgb * 255).astype(np.uint8)
        
        # Draw clicked point marker if exists
        if self.clicked_point is not None:
            x, y = self.clicked_point
            # Draw a crosshair at clicked point
            marker_size = 3
            y_coords = slice(max(0, y-marker_size), min(H, y+marker_size+1))
            x_coords = slice(max(0, x-marker_size), min(W, x+marker_size+1))
            # Draw red crosshair
            if 0 <= y < H:
                rgb[y, max(0, x-marker_size):min(W, x+marker_size+1), 0] = 255
                rgb[y, max(0, x-marker_size):min(W, x+marker_size+1), 1:] = 0
            if 0 <= x < W:
                rgb[max(0, y-marker_size):min(H, y+marker_size+1), x, 0] = 255
                rgb[max(0, y-marker_size):min(H, y+marker_size+1), x, 1:] = 0
        
        # Resize for display
        display_size = (386, 386)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize(display_size, Image.NEAREST)
        
        photo = ImageTk.PhotoImage(pil_img)
        self.rgb_label.config(image=photo)
        self.rgb_label.image = photo
        
    def update_mask_image(self):
        """Update mask image for current label"""
        if self.gt is None:
            return
        
        # Create binary mask for current label
        mask = (self.gt == self.current_label).cpu().numpy().astype(np.uint8)
        mask_rgb = np.stack([mask, mask, mask], axis=2) * 255
        
        # Resize for display
        display_size = (386, 386)
        pil_img = Image.fromarray(mask_rgb)
        pil_img = pil_img.resize(display_size, Image.NEAREST)
        
        photo = ImageTk.PhotoImage(pil_img)
        self.mask_label.config(image=photo)
        self.mask_label.image = photo
        
    def update_spectra_plot(self):
        """Sample and plot spectra from current mask region"""
        if self.image is None or self.gt is None:
            return
        
        # Get mask for current label
        mask = (self.gt == self.current_label)
        mask_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(mask_indices) == 0:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No pixels in current label', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # Sample random spectra
        num_samples = min(self.num_spectra, len(mask_indices))
        sampled_indices = mask_indices[torch.randperm(len(mask_indices))[:num_samples]]
        
        # Extract spectra
        self.ax.clear()
        for idx in sampled_indices:
            y, x = idx[0].item(), idx[1].item()
            spectrum = self.image[:, y, x].cpu().numpy()
            self.ax.plot(spectrum, linewidth=0.5)
        
        self.ax.set_xlabel('Band Index', fontsize=8)
        self.ax.set_ylabel('Reflectance', fontsize=8)
        self.ax.tick_params(labelsize=7)
        self.ax.set_title(f'Random {num_samples} Spectra', fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def update_spectra_num(self):
        """Update number of spectra from spinbox"""
        try:
            self.num_spectra = int(self.spectra_spinbox.get())
        except:
            pass
            
    def prev_image(self):
        """Navigate to previous sample"""
        new_idx = self.current_sample_idx - 1
        if new_idx >= 0:
            self.load_sample(new_idx)
        else:
            self.status_label.config(text="Already at first sample")
            
    def next_image(self):
        """Navigate to next sample"""
        new_idx = self.current_sample_idx + 1
        if new_idx < len(self.samples):
            self.load_sample(new_idx)
        else:
            self.status_label.config(text="Already at last sample")
            
    def prev_label(self):
        """Navigate to previous label"""
        if len(self.available_labels) == 0:
            return
        
        current_idx = self.available_labels.index(self.current_label) if self.current_label in self.available_labels else 0
        new_idx = (current_idx - 1) % len(self.available_labels)
        self.current_label = self.available_labels[new_idx]
        
        self.update_all_displays()
        self.label_text.config(text=f"Label: {self.current_label} (of {self.available_labels})")
        
    def next_label(self):
        """Navigate to next label"""
        if len(self.available_labels) == 0:
            return
        
        current_idx = self.available_labels.index(self.current_label) if self.current_label in self.available_labels else 0
        new_idx = (current_idx + 1) % len(self.available_labels)
        self.current_label = self.available_labels[new_idx]
        
        self.update_all_displays()
        self.label_text.config(text=f"Label: {self.current_label} (of {self.available_labels})")
        
    def on_image_click(self, event):
        """Handle click event on RGB image"""
        if self.image is None:
            return
        
        # Get click position relative to the label widget
        click_x = event.x
        click_y = event.y
        
        # Get image dimensions
        C, H, W = self.image.shape
        
        # Convert from display coordinates (386x386) to original image coordinates
        # The image is displayed at 386x386
        img_x = int(click_x * W / 386)
        img_y = int(click_y * H / 386)
        
        # Validate coordinates
        if 0 <= img_x < W and 0 <= img_y < H:
            self.clicked_point = (img_x, img_y)
            self.update_point_spectrum(img_x, img_y)
            self.update_rgb_image()  # Redraw RGB with marker
            self.status_label.config(text=f"Clicked point: ({img_x}, {img_y})")
        else:
            self.status_label.config(text=f"Click outside image bounds")
    
    def update_point_spectrum(self, x, y):
        """Update the spectral curve plot for clicked point"""
        if self.image is None:
            return
        
        # Extract spectrum at the clicked point
        spectrum = self.image[:, y, x].cpu().numpy()
        
        # Plot the spectrum
        self.ax_point.clear()
        self.ax_point.plot(spectrum, linewidth=2, color='blue', marker='o', markersize=2)
        self.ax_point.set_xlabel('Band Index', fontsize=8)
        self.ax_point.set_ylabel('Reflectance', fontsize=8)
        self.ax_point.set_title(f'Spectrum at ({x}, {y})', fontsize=9)
        self.ax_point.tick_params(labelsize=7)
        self.ax_point.grid(True, alpha=0.3)
        self.fig_point.tight_layout()
        self.canvas_point.draw()
    
    def clear_point_spectrum(self):
        """Clear the point spectrum plot"""
        self.ax_point.clear()
        self.ax_point.text(0.5, 0.5, 'Click on image to view spectrum', 
                          ha='center', va='center', transform=self.ax_point.transAxes)
        self.canvas_point.draw()
    
    def compute_spectral_purity(self):
        """Compute and display spectral purity metric"""
        if self.image is None or self.gt is None:
            return
        
        try:
            image_gpu = self.image.to(self.device)
            gt_gpu = self.gt.to(self.device)
            
            # Create mask for current label
            mask_gpu = (gt_gpu == self.current_label)
            
            purity, less_than = data_filter.compute_spectral_purity(image_gpu, mask_gpu)
            
            result_text = f"Spectral Purity: {purity:.4f}\nLess than 0.85: {less_than:.4f}"
            messagebox.showinfo("Spectral Purity", result_text)
            self.status_label.config(text=f"Spectral Purity: {purity:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error computing spectral purity: {str(e)}")
            
    def compute_edge_consistency(self):
        """Compute and display edge consistency metric"""
        if self.image is None or self.gt is None:
            return
        
        try:
            image_gpu = self.image.to(self.device)
            gt_gpu = self.gt.to(self.device)
            
            # Create mask for current label
            mask_gpu = (gt_gpu == self.current_label)
            
            consistency, ratio = data_filter.compute_relaxed_edge_consistency(
                image_gpu, mask_gpu, percentile_threshold=0.9
            )
            
            result_text = f"Edge Consistency: {consistency:.4f}\nRatio: {ratio:.4f}"
            messagebox.showinfo("Edge Consistency", result_text)
            self.status_label.config(text=f"Edge Consistency: {consistency:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error computing edge consistency: {str(e)}")
            
    def compute_spectral_variance(self):
        """Compute and display spectral variance metric"""
        if self.image is None or self.gt is None:
            return
        
        try:
            image_gpu = self.image.to(self.device)
            gt_gpu = self.gt.to(self.device)
            
            # Create mask for current label
            mask_gpu = (gt_gpu == self.current_label)
            
            variance = data_filter.compute_foreground_spectral_variance(image_gpu, mask_gpu)
            
            result_text = f"Foreground Spectral Variance: {variance:.6f}"
            messagebox.showinfo("Spectral Variance", result_text)
            self.status_label.config(text=f"Spectral Variance: {variance:.6f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error computing spectral variance: {str(e)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyperspectral Image Viewer')
    parser.add_argument('--index_file', type=str, 
                       # default='../data/hyperfree_index.json',
                        default='../data/hyperfree_index_labels_my_hf.json',
                       help='Path to index JSON file')
    args = parser.parse_args()
    
    if not os.path.exists(args.index_file):
        print(f"Error: Index file not found: {args.index_file}")
        return
    
    root = tk.Tk()
    app = HSIViewer(root, args.index_file)
    root.mainloop()


if __name__ == '__main__':
    main()
