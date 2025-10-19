import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import cv2
import h5py
from tqdm import tqdm
import subprocess
import plotly.io as pio
import plotly.graph_objects as go

if hasattr(pio.kaleido, 'scope') and hasattr(pio.kaleido.scope, '_context'):
    original_exec = pio.kaleido.scope._context.subprocess.Popen

    def silent_popen(*args, **kwargs):
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        return original_exec(*args, **kwargs)

    pio.kaleido.scope._context.subprocess.Popen = silent_popen


#####################
## Constants
#####################
window_size = 0.04  # 40 ms
window_size_frames = 0.3  # 300 ms
tau = 0.4 
roi_size = 5        # half-width of signal ROI in pixels
phi= 0.98 

start_time = 9
end_time = 14

save_images = True
save_animation = True


def save_plotly_plot(x, y, label1='line1',x2=None, y2=None,label2='line2', xlabel='', ylabel='', output_path='', show_markers=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                            y=y,
                            mode='lines+markers' if show_markers else 'lines',
                            name= label1,
                            marker=dict(size=4) if show_markers else None,
                            line=dict(width=2)
                            ))
    
     # Second plot (Orange), if provided
    if x2 is not None and y2 is not None:
        fig.add_trace(go.Scatter(
            x=x2,
            y=y2,
            mode='lines+markers' if show_markers else 'lines',
            name= label2,
            marker=dict(size=4) if show_markers else None,
            line=dict(width=2)
        ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family='Times New Roman', size=18)),
            tickfont=dict(size=18, family="Times New Roman")
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family='Times New Roman', size=18)),
            tickfont=dict(size=18, family="Times New Roman")
        ),
        legend=dict(font=dict(family="Times New Roman", size=14)),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='white'
    )
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    # Save as high-quality PNG
    fig.write_image(output_path, format='png', width=1200, height=400, scale=3)
    
def create_animation(frames, output_path, is_events=True, start_time=0, end_time=None, window_size=0.04, cmap='gray'):
    """
    Create and save an animation using matplotlib.animation.
    
    Args:
        frames (np.ndarray): Array of 2D (grayscale) or 3D (RGB) frames.
        output_path (str): File path to save animation (.mp4 or .gif).
        is_gray (bool): Whether the frames are grayscale.
        fps (int): Frames per second.
        start_time (float): Starting time in seconds.
        end_time (float): Ending time in seconds.
        window_size (float): Time window per frame in seconds.
        cmap (str): Colormap used for grayscale visualization.
    """
    def normalize_frame(frame):
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() > 1.0:
                frame = frame / 255.0
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame
    # Compute frame range
    start_idx = int(start_time / window_size)
    end_idx = int(end_time / window_size) if end_time is not None else len(frames)

    # Clip safely
    start_idx = max(0, start_idx)
    end_idx = min(len(frames), end_idx)
    
    fps = 1/window_size

    # Subset frames
    frames_subset = frames[start_idx:end_idx]
    

    h, w = frames_subset[0].shape[:2]
    dpi = 600
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi = dpi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)

    if is_events:
        # Normalize to 0-1 if needed
        if frames_subset[0].dtype != np.uint8:
            frames_subset = [np.clip(f, 0, 1) for f in frames_subset]
        im = ax.imshow(frames_subset[0], cmap=cmap, animated=True)
    else:
        for i, frame in enumerate(frames_subset):
            frames_subset[i] = normalize_frame(frame)
        im = ax.imshow(frames_subset[0], animated=True)

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames_subset,
        blit=True,
        interval=1000/fps
    )

    # Save as MP4 or GIF based on extension
    if output_path.endswith(".mp4"):
        ani.save(output_path, writer='ffmpeg', fps=fps)
    elif output_path.endswith(".gif"):
        ani.save(output_path, writer='pillow', fps=fps)
    else:
        raise ValueError("Output path must end with .mp4 or .gif")

    plt.close(fig)
    print(f"[INFO] Animation saved to: {output_path}")


#####################
## Paths
#####################

if os.name == 'nt':
    _ = os.system('cls')
else:
    _ = os.system('clear')
    
    
output_dir = Path('add_your_output_directory_here')  # Use Path object for cross-platform compatibility
path_to_h5 = Path('add_your_path_to_h5_files_here')

h5_file_list = [
     'file_name',    #do not add the extension (.h5)
]


#####################
## Analysis
#####################
file_index = 0
for h5_file_name in h5_file_list:
    (output_dir / h5_file_name).mkdir(parents=True, exist_ok=True)
    (output_dir / h5_file_name / "Events").mkdir(exist_ok=True)
    (output_dir / h5_file_name / "CMOS").mkdir(exist_ok=True)
    print(f"Processing file: {h5_file_name}.h5")
    
    with h5py.File(path_to_h5 / f"{h5_file_name}.h5", 'r') as f:
        events = f['events'][:]
        S1 = f['s1'][:]
        detections = f['detections'][:]
    
    sensor_width = int(events[:,0].max() + 1)
    sensor_height = int(events[:,1].max() + 1)
    

    # for i, s1 in enumerate(tqdm(S1, desc="Processing S1 Frames", unit="frame")):
    #     cv2.putText(s1, f"Frame: {i+1}", (10, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #     cv2.imshow("S1", s1)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()
    
    
    detection_time = detections[:,4].copy()
    detection_time = (detection_time - events[0,3]) * 1e-6
    events[:, 3] = (events[:, 3] - events[0, 3])*1e-6  # Normalize timestamps to start at 0
    
    ts0 = events[0, 3]
    ts1 = events[-1, 3]
    print(f"Time span: {(ts1 - ts0)} seconds")

    # Breaks for 40ms intervals
    breaks = np.arange(ts0, ts1, window_size)
    breaks_idx = np.searchsorted(events[:, 3], breaks)
    counts = np.diff(breaks_idx)

    if save_images:
        # --- Plot event count ---
        save_plotly_plot(
            x=breaks[:-1],
            y=counts,
            xlabel='Time (s)',
            ylabel='Event Count',
            output_path=str(output_dir / h5_file_name / f"{h5_file_name}_EventCounts.png")
        )
    print(f"Event count plot saved.")
    
    #################################
    ## Event-based SNR using Position
    #################################
    
    SNR_EVENTS = np.full(len(counts)+1, np.nan, dtype=object)
    detection_idx = 0
    
    for i, s1 in enumerate(tqdm(S1, desc="Processing S1 Frames", unit="frame")):
        pos_data = []
        _, s1_threshold = cv2.threshold(s1, phi, 1, cv2.THRESH_BINARY)

        if detection_idx < len(detection_time):
            while breaks[i] >= detection_time[detection_idx]:
                pos_data.append(detections[detection_idx])
                detection_idx += 1
                if detection_idx >= len(detection_time):
                    break

        s1_rec = s1.copy()
        mask = np.ones_like(s1_threshold, dtype=bool)
        roi = []
        if pos_data:
            for pos in pos_data:
                cx, cy = int(pos[0]*sensor_width), int(pos[1]*sensor_height)

                # Define square ROI
                x1, x2 = max(cx - roi_size, 0), min(cx + roi_size, sensor_width)
                y1, y2 = max(cy - roi_size, 0), min(cy + roi_size, sensor_height)
                
                # Draw red rectangle (in-place)
                s1_rec[y1:y2, x1:x1+2] = 1  # left
                s1_rec[y1:y2, x2-2:x2] = 1  # right
                s1_rec[y1:y1+2, x1:x2] = 1  # top
                s1_rec[y2-2:y2, x1:x2] = 1  # bottom
                
                mask[y1:y2, x1:x2] = False

        
                roi.append(s1_threshold[y1:y2, x1:x2])
        
        roi = np.array(roi)
        if roi.size == 0:
            continue
        signal = np.sum(roi)/roi.size
        
        
        background = s1_threshold[mask]
        noise = np.std(background)
        
        # Background = total - ROI
        if noise > 0 and signal > 0:
            snr_linear = signal / (noise * np.sqrt(roi.size))
            SNR_EVENTS[i] = 20 * np.log10(snr_linear)
        
        cv2.imshow("S1", s1_rec)
        cv2.waitKey(1)
        # # Save using matplotlib
        if save_images:
            plt.imsave(str(output_dir / h5_file_name / "Events" / f"{h5_file_name}_Event_Frame_{i}.png"), s1_rec, cmap='gray', format='png')
        
    cv2.destroyAllWindows()
        
    print(f"Event-based Mean SNR: {np.nanmean(SNR_EVENTS):.3f}")

    if save_images:
        save_plotly_plot(
            x=breaks[:-1],
            y=SNR_EVENTS,
            xlabel='Time (s)',
            ylabel='SNR (dB)',
            output_path=str(output_dir / h5_file_name / f"{h5_file_name}_SNR_Events.png"),
            show_markers=True
        )

    print(f"Event-based SNR plot saved.")
  
    #####################
    # Save Frames
    #####################
        
    file_index += 1
    print(f"Finished processing {h5_file_name}.h5 ({file_index}/{len(h5_file_list)})\n\n")
    
    if save_animation:
        create_animation(
            frames=S1,
            output_path=str(output_dir / h5_file_name / f"{h5_file_name}_Event_Animation.gif"),
            is_events=True,
            start_time=start_time,
            end_time=end_time,
            window_size=window_size
        )

