import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import qcodes as qc
import numpy as np
import cv2
from skimage import transform, filters
from scipy import ndimage, signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['text.usetex'] = True

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',      # use Computer Modern math fonts
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 25,
    'axes.linewidth': 1.2,
})

def get_data_from_qcodes_db(
        path_to_db: str, 
        run_id: int, 
        exp_name: str = "Initialization",
        plot: bool = False) -> list[np.array]:

    qc.dataset.initialise_or_create_database_at(path_to_db)
    dataset = qc.dataset.load_by_run_spec(experiment_name=exp_name, captured_run_id=run_id)
    if plot:
        qc.dataset.plot_dataset(dataset)
    df  = dataset.to_pandas_dataframe().reset_index()
    X_name, Y_name, Z_name = df.columns
    P_data, S_data = np.unique(df[X_name]), np.unique(df[Y_name])

    df_pivoted = df.pivot_table(values=Z_name, index=Y_name, columns=X_name)
    I_data = df_pivoted.to_numpy()
    return [P_data, S_data, I_data]

def extract_bias_point(
        lb_data: np.array,
        rb_data: np.array,
        current_data: np.array,
        minAngleDeg: float = -55,
        maxAngleDeg: float = -40,
        threshold: int = 50,
        minLineLength: int = 50,
        maxLineGap: int = 250,
        debug: bool = False,
        plot_results: bool = True) -> list[tuple]:

    
    lb_data = np.array(lb_data)
    rb_data = np.array(rb_data)
    current_data = np.array(current_data)

    
    if current_data.ndim == 1:
        nx = len(np.unique(lb_data))
        ny = len(np.unique(rb_data))
        current_data = current_data.reshape((ny, nx))

    # Here, we compute voltage-per-pixel scaling

    ny, nx = current_data.shape
    lb_voltages = np.linspace(lb_data.min(), lb_data.max(), nx)
    rb_voltages = np.linspace(rb_data.min(), rb_data.max(), ny)

    Gx, Gy = np.gradient(current_data)
    G = (1/np.sqrt(2)) * np.sqrt(Gx**2 + Gy**2)
    epsilon = 1e-3
    G_log = np.log(np.abs(G) + epsilon)

    if debug:
        plt.title(r"$\log\vert G \vert$")
        plt.imshow(
            G_log,
            extent = [lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            aspect='auto',
            origin='lower',
            cmap='binary')
        plt.xlabel("RB (V)")
        plt.colorbar(label = 'scaling')
        plt.ylabel("LB (V)")
        plt.show()

    G_log_normalized = G_log / G_log.max()
    G_uint = 255 * G_log_normalized
    image = G_uint.astype(np.uint8)

    """ image_eq = cv2.equalizeHist(image)

    
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    bg = cv2.morphologyEx(image_eq, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image_eq, bg, scale=255) """

    

    _, low_passed  = cv2.threshold(image, 10, 255, cv2.THRESH_TOZERO)
    _, band_passed = cv2.threshold(low_passed, 40, 255, cv2.THRESH_TOZERO_INV)

    #image_blur = cv2.GaussianBlur(band_passed, (5, 5), sigmaX=1.0)

    image_filtered = band_passed

    """ edges = cv2.Canny(
            out_tozero, 
            5,
            1000, 
            apertureSize=7,
            L2gradient=True
        ) """
    
    ridge_response = filters.sato(image_filtered, sigmas=[1, 2, 3], black_ridges=False)

    # Normalize ridge response for thresholding
    ridge_norm = (ridge_response - np.min(ridge_response)) / (np.max(ridge_response) - np.min(ridge_response))

    # Threshold to keep only strong ridges
    edges = ridge_norm > 0.3  # tune this threshold between 0.2–0.6 typically

    # Compute pixel-space midpoints

    ny, nx = edges.shape
    x_mid = nx // 2
    y_mid = ny // 2

    # Bottom-left in plot space → top-left in array space

    edges_masked = np.zeros_like(edges)
    edges_masked[:y_mid, :x_mid] = edges[:y_mid, :x_mid]

    if debug:
        plt.title("Edges (Bottom-Left Quadrant in Plot Coordinates)")
        plt.imshow(edges_masked, cmap='gray', origin='lower', aspect='auto')
        plt.xlabel("RB (V)")
        plt.ylabel("LB (V)")
        plt.show()

    # Use masked edges for line detection

    lines = transform.probabilistic_hough_line(edges_masked,
                                               threshold=15,        # Minimum votes to consider a line
                                               line_length=2,      # Minimum accepted length of detected line segment (pixels)
                                               line_gap=5           # Max gap between pixels to still form a single line
                                               )

    """ lines = cv2.HoughLinesP(edges_masked,
                        1, np.pi/180,
                        threshold=threshold,
                        minLineLength=minLineLength,
                        maxLineGap=maxLineGap
                        ) """

    filtered_lines = []
    if debug:
        plt.title("Detected Lines")
        plt.imshow(
            edges,
            aspect='auto',
            origin='lower',
            cmap='binary')
        plt.xlabel("RB (V)")
        plt.ylabel("LB (V)")
       
    if lines is None:
        if debug:
            print("No lines detected by HoughLinesP.")
        return []

    # Define bottom-left region boundaries in voltage space

    lb_mid = 0.5 * (lb_data[0] + lb_data[-1])
    rb_mid = 0.5 * (rb_data[0] + rb_data[-1])

    for p0, p1 in lines:
        # Extracted points nested in the list
        
        x1, y1 = p0
        x2, y2 = p1
        
        #x1,y1,x2,y2 = line[0]

        if x1 == x2:
            continue

        if debug:
            plt.plot([x1,x2], [y1, y2], c='r')

        slopeDeg = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if slopeDeg >= minAngleDeg and slopeDeg <= maxAngleDeg:
            filtered_lines.append([x1, y1, x2, y2])

        else:
            continue

    if debug:
        plt.show()
    
    bias_points = []
    for line in filtered_lines:

        # Extracted points nested in the list

        x1,y1,x2,y2=line

        # Convert to voltage

        voltageX = lb_voltages[x1]
        voltageY = rb_voltages[y1]

        # Restrict detected bias points to bottom-left quadrant

        if not (voltageX < lb_mid and voltageY < rb_mid):
            continue

        if plot_results:
            plt.title(r"$\log\vert I \vert$")
            plt.imshow(
                np.log(np.abs(current_data)),
                extent = [lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
                aspect='auto',
                origin='lower',
                cmap='binary')
            plt.xlabel("RB (V)")
            plt.ylabel("LB (V)")
            plt.scatter([voltageX], [voltageY], c='w', marker="*", s=30)
            plt.text(voltageX, voltageY, s=f"({round(voltageX,3)},{round(voltageY,3)})", color='w')

        bias_points.append((round(voltageX,3), round(voltageY,3)))

    if plot_results or debug: 
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Edges + Filtered Lines (Pixel Space)")

    # Show edges image directly (no extent mapping, no flipping)

    plt.imshow(edges, cmap='gray')

    # Overlay the raw line coordinates (pixel coordinates)

    for (x1, y1, x2, y2) in filtered_lines:
        plt.plot([x1, x2], [y1, y2], color='lime', lw=2)

    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.gca().invert_yaxis()  # match image coordinate system
    plt.show()

    if debug and filtered_lines:
        plt.figure(figsize=(8, 6))
        plt.title("Relevant Detected Lines (Filtered by Angle and Region)")
        
        # Show log|I| background for context
        plt.imshow(
            np.log(np.abs(current_data)),
            extent = [lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='binary'
        )
        
        for (x1, y1, x2, y2) in filtered_lines:
            # Convert pixel coords to voltage coords for plotting
            v1x = lb_voltages[x1]
            v1y = rb_voltages[y1]
            v2x = lb_voltages[x2]
            v2y = rb_voltages[y2]

            plt.plot([v1x, v2x], [v1y, v2y],
                     c='deepskyblue', lw=2, alpha=0.7)

        # Overlay extracted bias points (if any)
        if bias_points:
            voltages_x, voltages_y = zip(*bias_points)
            #plt.scatter(voltages_x, voltages_y,
            #            color='gold', marker='*', s=200,
            #            edgecolor='black', label='Bias Points')

        plt.xlabel("RB (V)")
        plt.ylabel("LB (V)")
        plt.legend()
        plt.show()

    if plot_results and bias_points:
        plt.figure(figsize=(8, 6))
        plt.title("Original Data with Extracted Bias Points")
        
        # Plot current_data directly using imshow
        plt.imshow(
            np.log(np.abs(current_data)),
            extent = [lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        
        # Overlay detected bias points as stars
        voltages_x, voltages_y = zip(*bias_points)
        #plt.scatter(voltages_x, voltages_y,
        #            color='gold', marker='*', s=200, edgecolor='black', label='Bias Points')
        
        plt.xlabel("RB (V)")
        plt.ylabel("LB (V)")
        plt.colorbar(label = 'Current (nA)')
        plt.legend()
        plt.show()

    # --- parameters you can tune ---
    perp_length_pixels = 80    # total length (in pixels) of the perpendicular sampling line
    perp_samples = 300         # number of sample points along each perpendicular trace
    deriv_peak_prominence = None  # optional, pass to find_peaks; None => use fallback selection

    # Insert this block after you compute `filtered_lines` (and after you computed lb_voltages, rb_voltages)
    perp_results = []  # will store results per filtered line: dict with top-3 voltage coords + derivative values

    if filtered_lines:
        # Prepare figure for aggregated overlay of found top-3 points
        if plot_results:
            plt.figure(figsize=(8,6))
            plt.title("Original Data with Extracted Perpendicular Peaks")
            plt.imshow(
                np.log(np.abs(current_data) + 1e-12),  # small epsilon to avoid log(0)
                extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
                origin='lower', aspect='auto', cmap='coolwarm'
            )
            plt.xlabel("RB (V)")
            plt.ylabel("LB (V)")

        ny_img, nx_img = current_data.shape
        x_index_arr = np.arange(nx_img)
        y_index_arr = np.arange(ny_img)

        for (x1, y1, x2, y2) in filtered_lines:
            # midpoint in pixel coords (may be float if you average)
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)

            # direction along the line (dx, dy)
            dx = (x2 - x1)
            dy = (y2 - y1)
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            dxu = dx / length
            dyu = dy / length

            # perpendicular unit vector (-dy, dx)
            pxu = -dyu
            pyu = dxu

            # sample parameter t across the perpendicular line centered at midpoint
            half_L = perp_length_pixels / 2.0
            t = np.linspace(-half_L, half_L, perp_samples)

            # compute sampling coordinates (floating point)
            samp_x = mx + pxu * t
            samp_y = my + pyu * t

            # Clip sampling coords a bit to remain inside image bounds
            samp_x_clipped = np.clip(samp_x, 0, nx_img - 1)
            samp_y_clipped = np.clip(samp_y, 0, ny_img - 1)

            # map_coordinates expects coordinates as [rows(y), cols(x)]
            trace = ndimage.map_coordinates(current_data, [samp_y_clipped, samp_x_clipped], order=3, mode='reflect')

            # compute derivative along t (units: value per pixel)
            dt = t[1] - t[0]  # spacing in pixels
            deriv = np.gradient(trace, dt)

            # work with absolute derivative to find the strongest conductance changes
            deriv_mag = np.abs(deriv)

            # try to find peaks in derivative magnitude
            peaks, props = signal.find_peaks(deriv_mag, prominence=deriv_peak_prominence) if deriv_peak_prominence is not None else signal.find_peaks(deriv_mag)
            # If find_peaks finds fewer than 3, fall back to selecting top-3 indices by magnitude
            if peaks.size == 0:
                # fallback: choose top-3 absolute derivative indices
                topk_idx = np.argpartition(-deriv_mag, range(min(3, len(deriv_mag))))[:3]
                # sort them by amplitude descending
                topk_idx = topk_idx[np.argsort(-deriv_mag[topk_idx])]
                chosen_indices = topk_idx.tolist()
            else:
                # get up to top-3 peaks by derivative magnitude (prominence already applied)
                peak_mags = deriv_mag[peaks]
                order = np.argsort(-peak_mags)
                chosen_indices = peaks[order][:3].tolist()

                # if fewer than 3 peaks, fill remaining with top global derivative points (non-peak)
                if len(chosen_indices) < 3:
                    remaining = 3 - len(chosen_indices)
                    # global top indices excluding already chosen
                    global_order = np.argsort(-deriv_mag)
                    for idx in global_order:
                        if idx not in chosen_indices:
                            chosen_indices.append(int(idx))
                        if len(chosen_indices) == 3:
                            break

            # Collect the chosen points, convert back to pixel coords and then to voltages
            chosen_points = []
            for idx in chosen_indices:
                px = samp_x_clipped[idx]
                py = samp_y_clipped[idx]
                # convert pixel x->voltage using interpolation of lb_voltages (x is column index)
                volt_x = float(np.interp(px, x_index_arr, lb_voltages))
                volt_y = float(np.interp(py, y_index_arr, rb_voltages))
                chosen_points.append({
                    "pixel_index": idx,
                    "pixel_coord": (px, py),
                    "voltage": (round(volt_x, 6), round(volt_y, 6)),
                    "deriv_value": float(deriv[idx]),
                    "deriv_mag": float(deriv_mag[idx])
                })

            # store results for this line
            perp_results.append({
                "line_pixels": (x1, y1, x2, y2),
                "midpoint_pixel": (mx, my),
                "perp_trace_t": t,
                "perp_trace_values": trace,
                "perp_trace_deriv": deriv,
                "top3": chosen_points
            })

            # Plot the 1D trace and mark the top-3
            if plot_results:
                plt.figure(figsize=(7, 3))
                plt.title(f"Perpendicular trace @ midpoint ({mx:.1f},{my:.1f}) — Top 3 derivative peaks")
                # x-axis: distance along perpendicular in pixels or convert to voltage distance if desired
                # here we show sample index (t) on x-axis and log|I| on y for readability
                plt.plot(t, trace, label="trace (I)")
                plt.plot(t, deriv, label="derivative dI/dt", alpha=0.7)
                for p in chosen_points:
                    idx = int(p["pixel_index"])
                    plt.axvline(t[idx], linestyle='--', alpha=0.8)
                    plt.scatter([t[idx]], [trace[idx]], s=50, marker='*', label=f"{p['voltage']}")
                    plt.text(t[idx], trace[idx], f"{p['voltage']}\nΔ={p['deriv_value']:.3e}", fontsize=8)
                plt.xlabel("perpendicular displacement (pixels)")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # overlay the top-3 points on the original 2D result plot
            if plot_results:
                for p in chosen_points:
                    vx, vy = p["voltage"]
                    plt.scatter([vx], [vy], marker='o', s=80, edgecolor='k', label='perp-peak' if False else "")
        # show overlay
        if plot_results:
            plt.show()

    # return both bias_points (as before) and perp_results if you want the extra info
    # modify the function return accordingly — for backward compatibility, you can return bias_points and perp_results
    return bias_points, perp_results

def extract_bias_point_final(
        lb_data: np.array,
        rb_data: np.array,
        current_data: np.array,
        minAngleDeg: float = -55,
        maxAngleDeg: float = -35,
        minLineLength: int = 50,
        maxLineGap: int = 250,
        debug: bool = False,
        plot_results: bool = True) -> list[tuple]:

    # We start by normalizing our inputs

    lb_data = np.array(lb_data)
    rb_data = np.array(rb_data)
    current_data = np.array(current_data)

    # Now, we reshape the data into an array

    if current_data.ndim == 1:
        nx = len(np.unique(lb_data))
        ny = len(np.unique(rb_data))
        current_data = current_data.reshape((ny, nx))

    ny, nx = current_data.shape

    # Here, we define the voltage ranges

    lb_voltages = np.linspace(lb_data.min(), lb_data.max(), nx)
    rb_voltages = np.linspace(rb_data.min(), rb_data.max(), ny)

    
    # ---------- Gradient Calculation and Ridge Detection ----------
    

    # Now, compute the gradient and the log of the gradient

    Gx, Gy = np.gradient(current_data)
    G = (1.0 / np.sqrt(2.0)) * np.sqrt(Gx**2 + Gy**2)

    g_lo, g_hi = np.percentile(G, [2, 98])
    G_clipped = np.clip(G, g_lo, g_hi)
    G_scaled = (G_clipped - g_lo) / (g_hi - g_lo)

    G_uint = (255 * G_scaled).astype(np.uint8)

    low = int(0.10 * 255)    # discard noise
    high = int(0.35 * 255)  # discard strongest boundaries

    # These next lines threshold above and below to keep a certain color band

    _, low_passed = cv2.threshold(G_uint, 10, low, cv2.THRESH_TOZERO)
    _, band_passed = cv2.threshold(low_passed, high, 255, cv2.THRESH_TOZERO_INV)

    # Here, we apply ridge detection, meaning we are detecting peaks within the image, then finding the middles of those peaks, widthwise

    epsilon = 1e-12

    ridge = filters.sato(band_passed, sigmas=[1, 2, 3], black_ridges=False)
    ridge_norm = (ridge - ridge.min()) / (np.ptp(ridge) + epsilon)
    ridge_filtered = ridge_norm > 0.20

    # Now, we limit our analysis to the bottom left-quadrant

    ridge_masked = np.zeros_like(ridge_filtered)
    ridge_masked[:ny // 2, :nx // 2] = ridge_filtered[:ny // 2, :nx // 2]

    # From these edges, we detect lines using a probabilistic hough transform 

    lines = transform.probabilistic_hough_line(
        ridge_masked,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
    )

    if not lines:
        return []

    # Now, we filter for lines within a certain angle range

    filtered_lines = []
    for p0, p1 in lines:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        angle = np.degrees(np.arctan2(dy, dx))
        if minAngleDeg <= angle <= maxAngleDeg:
            filtered_lines.append((*p0, *p1))

    if not filtered_lines:
        return []

    # Previously, we limited analysis to the bottom left quadrant, here we're defining the voltage range for that quadrant

    lb_mid_volt = 0.5 * (lb_data.min() + lb_data.max())
    rb_mid_volt = 0.5 * (rb_data.min() + rb_data.max())

    perp_candidates = []
    perp_traces_for_plot = []

    perp_length_pixels = max(40, int(min(nx, ny) * 0.5))
    perp_samples = 400
    smooth_sigma = 2.0

    x_index_arr = np.arange(nx)
    y_index_arr = np.arange(ny)

    # Now, for each filtered line, we define a line perpendicular to it, then find the peaks in current along them

    for x1, y1, x2, y2 in filtered_lines:
        
        # midpoints
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        # distances
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
            continue

        # perpendicular direction
        pxu, pyu = -dy / L, dx / L

        # length along the perpendicular lines in pixel space
        t = np.linspace(-perp_length_pixels / 2,
                        perp_length_pixels / 2,
                        perp_samples)

        # Limiting the values of the array to the bottom left quadrant
        samp_x = np.clip(mx + pxu * t, 0, nx - 1)
        samp_y = np.clip(my + pyu * t, 0, ny - 1)
        trace_id = len(perp_traces_for_plot)

        # defining current
        trace = ndimage.map_coordinates(
            current_data,
            [samp_y, samp_x],
            order=3,
            mode="reflect"
        )

        # smooth the trace
        trace_smooth = ndimage.gaussian_filter1d(trace, smooth_sigma)
        
        # Calculating conductance
        ds = np.sqrt(np.diff(samp_x)**2 + np.diff(samp_y)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        conductance = np.gradient(trace_smooth, s)

        trace_info = {
            "px": samp_x.copy(),
            "py": samp_y.copy(),
            "trace_id": trace_id,
            "trace": trace_smooth.copy(),
            "conductance": conductance.copy(),
            "s": s.copy()
        }

        # find local maxima of current
        noise_sigma = 1.4826 * np.median(np.abs(trace_smooth - np.median(trace_smooth)))
        prominence_thresh = 4.0 * noise_sigma
        peaks, _ = signal.find_peaks(trace_smooth,
                                     prominence=prominence_thresh,
                                     distance=15)

        if len(peaks) == 0:
            continue

        # defining the the maxima in voltage space from pixel space
        peak_idx = peaks
        px = samp_x[peak_idx]
        py = samp_y[peak_idx]

        vx = np.interp(px, x_index_arr, lb_voltages)
        vy = np.interp(py, y_index_arr, rb_voltages)

        # bottom-left quadrant restriction
        valid = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        peak_idx = peak_idx[valid]
        px = px[valid]
        py = py[valid]
        vx = vx[valid]
        vy = vy[valid]

        if len(peak_idx) == 0:
            continue

        # compiling data into trace info
        for k, p in enumerate(peak_idx):
            prom = signal.peak_prominences(trace_smooth, peak_idx)[0]
            score = prom[k]

            perp_candidates.append(
                (
                    score,
                    vx[k],
                    vy[k],
                    px[k],
                    py[k],
                    trace_id
                )
            )

        trace_info["peak_idx"] = peak_idx.copy()
        trace_info["vx_peak"] = vx
        trace_info["vy_peak"] = vy

        perp_traces_for_plot.append(trace_info)

    if not perp_candidates:
        return []


    # ---------- Selecting Final Bias Points ----------


    # First, we sort the points by current
    
    perp_candidates.sort(key=lambda x: -x[0])

    # Then, we pick the top 4 points of highest current

    N_FINAL = 4
    top_candidates = perp_candidates[:N_FINAL]

    selected_trace_ids = {c[5] for c in top_candidates}

    selected_peaks_by_trace = {
        c[5]: (c[1], c[2]) for c in top_candidates
    }

    for tr in perp_traces_for_plot:
        tid = tr["trace_id"]
        if tid in selected_peaks_by_trace:
            tr["peaks"] = [selected_peaks_by_trace[tid]]

    perp_bias_points = [
        (round(vx, 3), round(vy, 3))
        for (_, vx, vy, _, _, _) in top_candidates
    ]

    perp_traces_for_plot = [
        tr for tr in perp_traces_for_plot
        if tr["trace_id"] in selected_trace_ids
    ]

    # Now, we overlay perpendicular traces (strictly clipped to BL quadrant)
    
    for tr in perp_traces_for_plot:
        vx = np.interp(tr["px"], x_index_arr, lb_voltages)
        vy = np.interp(tr["py"], y_index_arr, rb_voltages)

        in_quad = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        if not np.any(in_quad):
            continue

        idx = np.where(in_quad)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        blocks = np.split(idx, splits + 1)

        peak_idx = tr.get("peak_idx", None)
        chosen_block = None
        if peak_idx is not None:
            for b in blocks:
                if np.intersect1d(peak_idx, b).size > 0:
                    chosen_block = b
                    break
        if chosen_block is None:
            chosen_block = blocks[0]

        tr["chosen_block"] = chosen_block


    # ---------- Final Plotting ----------


    if plot_results:

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10,10))

        # Show image
        im = ax.imshow(
            current_data,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        # Set axis limits
        ax.set_xlim(lb_data.min(), lb_data.max())
        ax.set_ylim(rb_data.min(), rb_data.max())

        # Round ticks
        step = 0.001
        def round_to_step(x, step): return step * np.round(x / step)

        x0, x1 = round_to_step(lb_data.min(), step), round_to_step(lb_data.max(), step)
        y0, y1 = round_to_step(rb_data.min(), step), round_to_step(rb_data.max(), step)
        
        ax.set_xticks([0.65, 0.85])
        ax.set_yticks([0.60, 0.70])
        ax.set_xticklabels(["0.65", "0.85"], fontsize=40)
        ax.set_yticklabels(["0.60", "0.70"], fontsize=40)

        ax.tick_params(
            which="major",
            direction="in",
            length=6,
            width=1.2,
            top=True,
            right=True
        )

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(9))
        ax.yaxis.set_minor_locator(AutoMinorLocator(9))

        # Style minor ticks (no labels by default)
        ax.tick_params(
            which="minor",
            direction="in",
            length=3,
            width=1.0,
            top=True,
            right=True
        )

        # Axis labels
        ax.set_xlabel(r'V$_{B2}$ (V)', fontsize=45, labelpad = -35)
        ax.set_ylabel(r'V$_{B1}$ (V)', fontsize=45)

        ax.yaxis.set_label_coords(-0.025, 0.40)

        # Create horizontal colorbar above the axes
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(
            ax,
            width="100%",
            height="50%",
            loc="upper center",
            bbox_to_anchor=(0, 1.08, 1, 0.1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("I (nA)", fontsize=35, labelpad=10)
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
        cbar.set_ticklabels(['0', '', '', '', '1'])
        cbar.ax.tick_params(labelsize=30, direction="in", length=6)

        cbar.ax.minorticks_on()

        cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # Style minor ticks
        cbar.ax.tick_params(
            which="minor",
            direction="in",
            length=4,
            width=1.0
        )

        # Block Boundary

        # Top side: horizontal line from left-mid to right-mid
        ax.plot(
            [lb_mid_volt, lb_data.min()],  # x: left → right
            [rb_mid_volt, rb_mid_volt],    # y constant at top
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Right side: vertical line from bottom-mid to top-mid
        ax.plot(
            [lb_mid_volt, lb_mid_volt],    # x constant at right
            [rb_data.min(), rb_mid_volt],  # y: bottom → top
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Hough lines
        for x1, y1, x2, y2 in filtered_lines:
            # compute voltage coordinates
            v1x = np.interp(x1, x_index_arr, lb_voltages)
            v1y = np.interp(y1, y_index_arr, rb_voltages)
            v2x = np.interp(x2, x_index_arr, lb_voltages)
            v2y = np.interp(y2, y_index_arr, rb_voltages)
            
            # Uncomment below to see the detected Hough lines
            # ax.plot([v1x, v2x], [v1y, v2y], c='deepskyblue', lw=1.2) 

        # Perpendicular traces and peaks
        for tr in perp_traces_for_plot:
            idx = tr.get("chosen_block", None)
            if idx is None or len(idx) == 0: continue
            vx = np.interp(tr["px"], np.arange(nx), lb_voltages)
            vy = np.interp(tr["py"], np.arange(ny), rb_voltages)
            ax.plot(vx[idx], vy[idx], c='yellow', lw=1.5, alpha=0.9)
            valid_peaks = np.intersect1d(tr.get("peak_idx", []), idx)
            ax.scatter(vx[valid_peaks], vy[valid_peaks], s=150, c='white',
                    marker='*', edgecolors='black', zorder=10)

        ax.set_box_aspect(0.775)

        plt.show()


        # These are 1D perpendicular trace plots

        if debug:
            for tr in perp_traces_for_plot:
                s = tr["s"]
                I = tr["trace"]                 # smoothed current
                dIds = tr["conductance"]        # dI/ds
                peak_idx = tr.get("peak_idx", [])
                chosen_block = tr.get("chosen_block", None)

                fig, axs = plt.subplots(
                    2, 1, figsize=(7, 5), sharex=True
                )

                # We plot the current traces here

                axs[0].plot(s, I, color="black", lw=1.3)
                axs[0].set_ylabel("Current")
                axs[0].set_title(
                    f"Perpendicular trace {tr['trace_id']}"
                )

                # Mark current peaks
                
                if len(peak_idx) > 0:
                    axs[0].scatter(
                        s[peak_idx],
                        I[peak_idx],
                        c="red",
                        s=40,
                        zorder=5,
                        label="Current peaks"
                    )

                # Shade chosen block (if present)
                
                if chosen_block is not None and len(chosen_block) > 0:
                    axs[0].axvspan(
                        s[chosen_block[0]],
                        s[chosen_block[-1]],
                        color="orange",
                        alpha=0.15,
                        label="Selected block"
                    )

                axs[0].legend(loc="best")

                # Here, we plot the conductance as well

                axs[1].plot(s, dIds, color="tab:blue", lw=1.2)
                axs[1].set_xlabel("Arc length s (pixels)")
                axs[1].set_ylabel("dI/ds")

                plt.tight_layout()
                plt.show()


    if debug:
        
        # We first plot the original current data
        
        plt.figure(figsize=(10, 6))
        plt.imshow(current_data,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Original Current Data") 
        plt.show()
        
        # Next, we plot the gradient of the data, i.e. the conductance

        plt.figure(figsize=(10, 6))
        plt.imshow(G,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("G (Conductance)") 
        plt.show()

        # Then, we plot G_log normalized to 255, or G_uint

        plt.figure(figsize = (10, 6))
        plt.imshow(G_uint, 
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title(r"$G Normalized$")
        plt.show()

        # Here is the plot of the band-passed G_uint, i.e. after being thresholded

        plt.figure(figsize = (10, 6))
        plt.imshow(band_passed, 
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title(r"$G Normalized and Thresholded$")
        plt.show()

        # Here is a plot of the ridges

        plt.figure(figsize=(10, 6))
        plt.imshow(ridge,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Ridges detected")
        plt.show()

        # Here is a plot of the normalized ridges

        plt.figure(figsize=(10, 6))
        plt.imshow(ridge_norm,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Normalized Ridges Detected")
        plt.show()

        # Here is a plot of the ridges filtered for strength

        plt.figure(figsize=(10, 6))
        plt.imshow(ridge_masked,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Filtered Ridges Detected")
        plt.show()


        # ---------- Hough Line Plotting ----------


        # Now, we'll plot a set of lines from the Hough Transform at each preprocessing stage


        # We start with lines detected from the original data

        plt.figure(figsize = (10, 6))
        plt.imshow(
            current_data,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        current_data,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Original Data")
        plt.show()

        # Now, we detect lines from the gradient

        plt.figure(figsize = (10, 6))
        plt.imshow(
            G,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        G,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Gradient")
        plt.show()

        # Now, we detect lines from the normalized Gradient

        plt.figure(figsize = (10, 6))
        plt.imshow(
            G_uint,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        G_uint,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from G Normalized")
        plt.show()

        # Now, we detect lines from the G_log normalized after thresholding

        plt.figure(figsize = (10, 6))
        plt.imshow(
            band_passed,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        band_passed,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from band-passed data")
        plt.show()

        # Now, we detect lines from the ridges

        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Ridges")
        plt.show()

        # Now, we detect lines from the Normalized Ridges

        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge_norm,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge_norm,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Normalized Ridges")
        plt.show()

        # Now, we detect lines from the ridges after filtering

        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge_filtered,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge_filtered,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range

        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Filtered Ridges")
        plt.show()

    return perp_bias_points

