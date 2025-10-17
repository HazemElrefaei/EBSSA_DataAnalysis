# Event-Based RSO Detection and SNR Analysis

This Python script performs **event-based SNR (Signal-to-Noise Ratio) analysis**, **event count statistics**, and **detection frame visualization** for labeled Resident Space Object (RSO) datasets captured by neuromorphic vision sensors (e.g., DAVIS346, DVXplorer).

It is designed for the **Code Availability section** of your scientific publication and is compatible with HDF5 datasets exported from the preprocessing pipeline that includes:
- `/events`: raw event stream as `[x, y, polarity, timestamp]`
- `/s1`: EDTS (Exponential Decaying Time Surface) frames
- `/detections`: normalized detection bounding boxes per frame `[x_center, y_center, width, height, timestamp]`

---

## 🔧 Features

- Visualizes EDTS (`S1`) frames with detection boxes overlaid.
- Computes **event-based SNR** using signal ROI and background noise estimation.
- Generates:
  - Event count vs. time plots.
  - SNR vs. time plots.
  - GIF animations of annotated `S1` frames.

---

## 📁 Folder Structure

The script expects this structure:

```
your_project/
├── your_script.py
├── /h5_data/
│   ├── file_name.h5
├── /results/
│   ├── file_name/
│       ├── Events/
│       ├── file_name_EventCounts.png
│       ├── file_name_SNR_Events.png
│       ├── file_name_Event_Animation.gif
```

---

## 🚀 How to Run

1. **Set Input & Output Paths**  
   Modify the following variables at the top of the script:
   ```python
   output_dir = r'/Add_Output_Path//'
   path_to_h5 = r'/Add_path_to_h5_file'
   ```

2. **Set Target Files**  
   Add the base name (without `.h5`) of the file(s) to analyze:
   ```python
   h5_file_list = ['file_name']
   ```

3. **Run the Script**  
   Use Python 3:
   ```bash
   python your_script.py
   ```

---

## 📊 Outputs

- **Event Count Plot**:  
  `file_name_EventCounts.png`: Number of events per 40ms interval.

- **Event-based SNR Plot**:  
  `file_name_SNR_Events.png`: Frame-wise SNR computed from the signal ROI vs. background noise.

- **Detection-annotated Frames**:  
  Saved in `Events/` folder as grayscale images with bounding boxes.

- **Animated Visualization**:  
  `file_name_Event_Animation.gif`: Animated playback of EDTS frames with overlaid detections.

---

## 📦 Requirements

# Data Analysis and Visualization Dependencies
matplotlib>=3.5.0
numpy>=1.21.0
opencv-python>=4.5.0
h5py>=3.6.0
tqdm>=4.62.0
plotly>=5.0.0
 
# Additional visualization dependencies
kaleido>=0.2.1

Install the dependencies using `pip`:

```bash
pip install numpy pandas opencv-python matplotlib plotly tqdm h5py
```

---

## ⚙️ Parameters

The following constants can be configured for your dataset:
```python
window_size = 0.04        # Event accumulation window (40 ms)
tau = 0.4                 # Time decay constant for EDTS
phi = 0.98                # Threshold for S1 binary mask
roi_size = 5              # Half-width of the signal ROI in pixels
start_time = 9            # Start time (s) for GIFs
end_time = 14             # End time (s) for GIFs
save_images = False       # Save static images
save_animation = True     # Save GIF animations
```

---

## 📌 Notes

- `detections` dataset should contain normalized coordinates (0–1) per frame.
- To reduce visual clutter, detection boxes are scaled by a factor of 3.
- You can export labeled data using the preprocessing script used in the Science Data Bank pipeline.

---

## 📄 License

This script is open-sourced for academic use under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 📬 Contact

For issues or questions related to the dataset or this code, please contact:

**Hazem Elrefaei**  
[helrifeai@gmail.com](mailto:helrifaei@gmail.com)  
Advanced Research and Innovation Center (ARIC) – Khalifa University
