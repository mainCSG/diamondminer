from numpy import ndarray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import scipy as sp 
from typing import Optional, Dict, List, Tuple

@dataclass
class CoulombDiamond:
    """Class for keeping track of Coulomb diamond information."""
    name: str
    left_vertex: tuple
    top_vertex: tuple
    right_vertex: tuple
    bottom_vertex: tuple
    oxide_thickness: float = None # m
    epsR: float = None # F/m
    e: float = 1.60217663e-19 # C
    eps0: float = 8.8541878128e-12 # F/m

    def width(self) -> float:
        return np.abs(self.right_vertex[0] - self.left_vertex[0])
    
    def height(self) -> float:
        return np.abs(self.top_vertex[1] - self.bottom_vertex[1])
    
    def beta(self) -> float:
        # positive slopes (drain lever arm)
        beta1 = (self.top_vertex[1] - self.left_vertex[1]) / (self.top_vertex[0] - self.left_vertex[0])
        beta2 = (self.bottom_vertex[1] - self.right_vertex[1]) / (self.bottom_vertex[0] - self.right_vertex[0])
        return 0.5 * (beta1 + beta2)
    
    def gamma(self) -> float:
        # positive slopes (source lever arm)
        gamma1 = -1 * (self.top_vertex[1] - self.right_vertex[1]) / (self.top_vertex[0] - self.right_vertex[0])
        gamma2 = -1 * (self.bottom_vertex[1] - self.left_vertex[1]) / (self.bottom_vertex[0] - self.left_vertex[0])
        return 0.5 * (gamma1 + gamma2)
    
    def alpha(self) -> float:
        return (self.beta()**-1 + self.gamma()**-1)**-1

    def lever_arm(self) -> float:
        return self.charging_voltage() / self.addition_voltage() 
    
    def addition_voltage(self) -> float:
        return self.width()
    
    def charging_voltage(self) -> float:
        return self.height() / 2

    def total_capacitance(self) -> float:
        return self.e / self.charging_voltage()
    
    def source_capacitance(self) -> float:
        return self.gate_capacitance() / self.gamma()

    def drain_capacitance(self) -> float:
        return self.gate_capacitance() * ((1- self.beta()) / self.beta() )
    
    def gate_capacitance(self) -> float:
        return self.e / self.addition_voltage()
    
    def dot_area(self) -> float:
        # Assumes parallel plate capicitaor geometry
        dot_area = (self.oxide_thickness * self.total_capacitance()) / (self.eps0 * self.epsR)
        return dot_area

    def dot_radius(self) -> float:
        if self.oxide_thickness is None:
            # Less accurate
            if self.epsR is None:
                return -1
            #https://arxiv.org/pdf/1910.05841
            return self.total_capacitance() / (8 * self.eps0 * self.epsR)
        else:
            #https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-05700-9/MediaObjects/41467_2018_5700_MOESM1_ESM.pdf
            dot_radius = np.sqrt(self.dot_area() / np.pi)
            return dot_radius

    def print_summary(self):
        print(f"Summary ({self.name}):")
        print("===================")
        print("\n")

        print("Constants")
        print("---------")
        print(f"Elementary Charge (e): {self.e:.5e} C")
        print(f"Permittivity of Free Space (\u03F50): {self.eps0:.5e} F/m")
        if self.epsR is not None:
            print(f"Relative Permittivity (\u03F5R): {self.epsR:.5f}")
        if self.oxide_thickness is not None:
            print(f"Oxide Thickness: {self.oxide_thickness * 1e9:.5f} nm")
        print("---------")
        print("\n")

        print("Geometry")
        print("---------")
        print(f"Left Vertex: {self.left_vertex}")
        print(f"Top Vertex: {self.top_vertex}")
        print(f"Right Vertex: {self.right_vertex}")
        print(f"Bottom Vertex: {self.bottom_vertex}")
        print(f"Width: {self.width()*1e3:.5f} mV")
        print(f"Height: {self.height()*1e3:.5f} mV")
        print("---------")
        print("\n")
        
        print("Dot Properties")
        print("--------------")
        print(f"Total Lever Arm (\u03B1): {self.lever_arm():.5f} eV/V")
        print(f"Drain Lever Arm (\u03B2): {self.beta():.5f} eV/V")
        print(f"Source Lever Arm (\u03B3): {self.gamma():.5f} eV/V")
        print(f"Addition Voltage: {self.addition_voltage() * 1e3:.5f} mV")
        print(f"Charging Voltage: {self.charging_voltage()* 1e3:.5f} mV")
        print(f"Gate Capacitance: {self.gate_capacitance() * 1e18:.5f} aF")
        print(f"Source Capacitance: {self.source_capacitance() * 1e18:.5f} aF")
        print(f"Drain Capacitance: {self.drain_capacitance() * 1e18:.5f} aF")
        print(f"Total Capacitance: {self.total_capacitance() * 1e18:.5f} aF")
        print(f"Dot Radius: {self.dot_radius() * 1e9:.5f} nm")
        print("--------------")
        print("\n")

    def plot(self, ax):
        vertices = [self.left_vertex, self.top_vertex, self.right_vertex, self.bottom_vertex, self.left_vertex]
        polygon = patches.Polygon(vertices, closed=True, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(polygon)
        
        # Calculate the center of the diamond for the label
        center_x = (self.left_vertex[0] + self.right_vertex[0]) / 2
        ax.text(center_x, 0, self.name, color='blue', ha='center', va='center', fontsize=10, weight='bold')

class Miner:

    def __init__(self,
                gate_data: ndarray,
                ohmic_data: ndarray,
                current_data: ndarray,
                epsR: Optional[float] = None,
                oxide_thickness: Optional[float] = None,
                binary_threshold: float = 1.1,
                blur_sigma: float = 1.0,
                blur_kernel: tuple = (3,3)) -> None:
        self.epsR = epsR
        self.oxide_thickness = oxide_thickness
        self.gate_data = gate_data
        self.ohmic_data = ohmic_data
        self.current_data = current_data

        self.current_data_height, self.current_data_width = self.current_data.shape
        self.image_ratio = self.current_data_height / self.current_data_width
        self.gate_voltage_per_pixel = np.abs((self.gate_data[-1] - self.gate_data[0]) / self.current_data_width)
        self.ohmic_voltage_per_pixel = np.abs((self.ohmic_data[-1] - self.ohmic_data[0]) / self.current_data_height)

        self.binary_threshold = binary_threshold
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
    
    def filter_raw_data(self, 
                        current_data: ndarray,
                        binary_threshold: float = 1.1,
                        blur_sigma: float = 1.0,
                        blur_kernel: tuple = (3,3)) -> ndarray:
        
        assert binary_threshold >= 1

        filtered_current_data = np.log(
            np.abs(current_data)
        )

        new_data = np.zeros_like(filtered_current_data)

        mask = filtered_current_data < binary_threshold * np.nanmax(filtered_current_data)

        new_data[mask] = 255
        new_data = new_data.astype(np.uint8)

        # Apply Gaussian blur to smooth the image and reduce noise
        filtered_data = cv2.GaussianBlur(
            new_data, 
            blur_kernel, 
            blur_sigma
        )
        
        return filtered_data

    def extract_diamonds(self, debug: bool = False) -> None:

        upper_mask = np.zeros_like(self.current_data, dtype=bool)
        upper_mask[:self.current_data_height//2, :] = True

        lower_mask = np.zeros_like(self.current_data, dtype=bool)
        lower_mask[self.current_data_height//2:, :] = True

        masks = {
            'upper': upper_mask,
            "lower": lower_mask
        }

        line_dict = {
            'upper': {'positive': [], 'negative': []},
            'lower': {'positive': [], 'negative': []}
        }

        if debug:
            plt.title("Filtered Data + Detected Lines")
            plt.imshow(
                self.filter_raw_data(
                    self.current_data,
                    binary_threshold=self.binary_threshold,
                    blur_sigma=self.blur_sigma,
                    blur_kernel=self.blur_kernel
                ),
                cmap='binary', 
                aspect='auto'
            )
            plt.show()

        for section in ["upper", "lower"]:
            image = self.current_data * masks[section]
            image_threshold = self.filter_raw_data(
                image,
                binary_threshold=self.binary_threshold,
                blur_sigma=self.blur_sigma,
                blur_kernel=self.blur_kernel
            )
            image_edges = self.extract_edges(image_threshold)
            if debug:
                plt.title(f"Section [{section}] Edges")
                plt.imshow(
                    image_edges,
                    cmap='binary', 
                    aspect='auto'
                )
                

            image_lines = self.extract_lines(image_edges)
            # Iterate over points
            for points in image_lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]

                if x1 == x2:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                
                if np.abs(slope)/self.image_ratio < 0.5:
                    continue
                y_middle = self.current_data_height//2
                y_upper = self.current_data_height
                y_lower = 0
                if y2 > y_middle:
                    px = self.x_intercept(x1, y1, x2, y2, y_upper)
                    py = y_upper
                else:
                    px = self.x_intercept(x1, y1, x2, y2, y_lower)
                    py = y_lower

                x_intercept = self.x_intercept(x1, y1, x2, y2, y_middle)
                # if x_intercept < 0 or x_intercept > self.current_data_width:
                #     continue

                if slope > 0:
                    line_dict[section]["negative"].append([px, py, x_intercept, y_middle])
                else:
                    line_dict[section]["positive"].append([px, py, x_intercept, y_middle])

                if debug:
                    if section == "upper":
                        color = 'blue'
                    else:
                        color = "red"

                    if slope > 0:
                        color = "dark" + color
                    else:
                        color = color

                    plt.plot([px, x_intercept], [py, y_middle], c=color)
            if debug:
                plt.show()

        upper_pos = sorted(line_dict['upper']['positive'], key = lambda x: x[2])
        upper_neg = sorted(line_dict['upper']['negative'], key = lambda x: x[2])
        lower_pos = sorted(line_dict['lower']['positive'], key = lambda x: x[2])
        lower_neg = sorted(line_dict['lower']['negative'], key = lambda x: x[2])

        assert len(upper_pos) == len(upper_neg), "Unbalanced lines detected in the upper half"
        assert len(lower_pos) == len(lower_neg), "Unbalanced lines detected in the lower half"

        diamond_shapes = []  
        for u_p, u_n, l_p, l_n in zip(upper_pos, upper_neg, lower_pos, lower_neg):
            # [2] is the x-intercept of the line

            left_x_int = [max(0, int((u_p[2] + l_n[2]) / 2)), self.current_data_height //2]
            right_x_int = [min(self.current_data_height, int((u_n[2] + l_p[2]) / 2)), self.current_data_height //2]
            upper_vertex = self.get_intersect(u_p, u_n)
            lower_vertex = self.get_intersect(l_n, l_p)

            diamond_shapes.append([left_x_int, upper_vertex, right_x_int, lower_vertex])

        for i in range(len(diamond_shapes) - 1):
            left_diamond = diamond_shapes[i]
            right_diamond = diamond_shapes[i+1]

            average_x = int((left_diamond[2][0] + right_diamond[0][0]) / 2)
            left_diamond[2][0] = average_x
            right_diamond[0][0] = average_x

        detected_coulomb_diamonds = []
        for number, diamond_shape in enumerate(diamond_shapes):
                    
            xs, ys = zip(*(diamond_shape+diamond_shape[:1]))
            gate_values = [self.gate_data[0] + x * self.gate_voltage_per_pixel for x in xs]
            ohmic_values = [y * self.ohmic_voltage_per_pixel - self.ohmic_data[-1] for y in ys]
            diamond_vertices_voltage = np.vstack((gate_values, ohmic_values)).T[:-1, :]

            detected_coulomb_diamonds.append(
                CoulombDiamond(
                    name=f"#{number}",
                    left_vertex=diamond_vertices_voltage[0],
                    top_vertex=diamond_vertices_voltage[3],
                    right_vertex=diamond_vertices_voltage[2],
                    bottom_vertex=diamond_vertices_voltage[1],
                    oxide_thickness=self.oxide_thickness,
                    epsR=self.epsR
                )
            )

        self.diamonds = detected_coulomb_diamonds
        return detected_coulomb_diamonds

    def estimate_temperatures(
        self, 
        diamonds: list[CoulombDiamond], 
        ohmic_value: float,
        temperature_guess: float = 1) -> list[float]:
    
        fig, axes = plt.subplots()
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.375, 0.22, 0.3, 0.25]
        ax_inset = fig.add_axes([left, bottom, width, height])
        axes.imshow(
            self.current_data, 
            cmap='binary',
            aspect='auto',
            origin='lower',
            extent=[
                self.gate_data[0],
                self.gate_data[-1], 
                self.ohmic_data[0], 
                self.ohmic_data[-1]
            ],
        )
        
        axes.set_xlabel("Gate Voltage (V)")
        axes.set_ylabel("Ohmic Voltage (V)")
        axes.axhline(ohmic_value, c='k', linestyle="--")
        axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        temperatures = []
        for i in range(len(diamonds)-1):
            # Get estimated lever arm from neighbouring diamonds
            left_diamond = diamonds[i]
            right_diamond = diamonds[i+1]
            left_Vg, left_Vsd = left_diamond.top_vertex
            right_Vg, right_Vsd = right_diamond.top_vertex
            average_lever_arm = (left_diamond.lever_arm() + right_diamond.lever_arm()) / 2

            # Filter out the coulomb oscillation between two diamonds
            gate_mask = np.where((self.gate_data >= left_Vg) * (self.gate_data <= right_Vg))[0]
            ohmic_index = (np.abs(self.ohmic_data - ohmic_value)).argmin()
            P_data_filtered = self.gate_data[gate_mask]
            ohmic_value = self.ohmic_data[ohmic_index]
            oscillation = self.current_data[ohmic_index, gate_mask]

            # Fit data to Coulomb peak theoretical formula
            guess = [oscillation.min(), oscillation.max(), np.average(P_data_filtered), temperature_guess]
            (a,b,V0,Te), coeffs_cov = sp.optimize.curve_fit(
                lambda V, a, b, V0, Te: self.coulomb_peak(V, average_lever_arm, a, b, V0, Te), 
                P_data_filtered, 
                oscillation, 
                p0=guess,
                bounds=([-np.inf, -np.inf, -np.inf, -np.inf],[np.inf,np.inf,np.inf,np.inf])
            )

            # Plot results
            Vs = np.linspace(P_data_filtered.min(), P_data_filtered.max(), 100)
            ax_inset.plot(P_data_filtered, oscillation, 'k.')
            ax_inset.plot(Vs, self.coulomb_peak(Vs, average_lever_arm, a, b, V0, Te),'k-')
            ax_inset.set_xlabel(r"Gate Voltage (V)")
            ax_inset.set_ylabel(r"Current (A)")

            temperatures.append(Te)
        temperatures = np.array(temperatures)
        T_avg = np.average(np.abs(temperatures))
        T_stddev = np.std(temperatures)
    
        axes.set_title(
            f'$T_e$ = ${round(T_avg,3)}$ K $\pm\ {round(T_stddev/np.sqrt(len(temperatures)), 3)}$ K'
        )
                
        plt.show()
        return temperatures

    def coulomb_peak(self, V, alpha, a, b, V0, Te):
        kB = 8.6173303e-5 # eV / K
        return a + b * (np.cosh(alpha * (V0 - V) / (2 * kB * Te)))**(-2)

    def get_statistics(self, diamonds: Optional[list[CoulombDiamond]] = None) -> dict:
        if diamonds is None:
            diamonds = self.diamonds

        methods = [
            'lever_arm',
            'addition_voltage',
            'charging_voltage',
            'total_capacitance',
            'gate_capacitance',
            'source_capacitance',
            'drain_capacitance',
            'dot_radius',
        ]
        methods_print = {
            'lever_arm': lambda mu, std: f"Average Lever Arm (\u03B1) : {mu:.5f} eV/V \u00b1 {std:.5f} eV/V",
            'addition_voltage' : lambda mu, std: f"Average Addition Voltage: {1e3 * mu:.5f} mV \u00b1 {1e3 *std:.5f} mV",
            'charging_voltage': lambda mu, std: f"Average Charging Voltage: {1e3 *mu:.5f} mV \u00b1 {1e3 *std:.5f} mV",
            'total_capacitance': lambda mu, std: f"Average Total Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'gate_capacitance': lambda mu, std: f"Average Gate Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'source_capacitance': lambda mu, std: f"Average Source Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'drain_capacitance': lambda mu, std: f"Average Drain Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'dot_radius': lambda mu, std: f"Average Dot Radius: {1e9 * mu:.5f} nm \u00b1 {1e9 * std:.5f} nm",
        }


        results = {}

        for method in methods:
            results[method] = sp.stats.norm.fit([getattr(diamond, method)() for diamond in diamonds])
        
        for method, (mu, std) in results.items():
            print(
                methods_print[method](mu, std/len(diamonds))
            )

        return results
    
    def extract_edges(self, image: ndarray) -> ndarray:

        # Perform Canny edge detection
        edges = cv2.Canny(
            image, 
            0, 
            0, 
            apertureSize=3
        )

        return edges

    def extract_lines(self, 
                      edges: ndarray,
                      threshold: int = 10,
                      maxLineGap: int = 250,
                      minLineLength: int = None,
                      distanceThreshold: int = None,
                      angleThreshold: float = None
                      ) -> list:

        if minLineLength is None:
            minLineLength = self.current_data_height // 10
        if distanceThreshold is None:
            distanceThreshold = self.current_data_width//10
        if angleThreshold is None:
            angleThreshold = 0.25
        lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=threshold, # Min number of votes for valid line
            minLineLength=minLineLength, # Min allowed length of line
            maxLineGap=maxLineGap # Max allowed gap between line for joining them
            )
        # Filter out duplicate lines
        lines = self.filter_duplicate_lines(lines, distance_threshold=distanceThreshold, angle_threshold=angleThreshold)

        return lines

    def filter_duplicate_lines(self, lines, distance_threshold=None, angle_threshold=None):
        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            keep_line = True
            if y2 == y1:
                continue
            for other_line in filtered_lines:
                ox1, oy1, ox2, oy2 = other_line[0]
                # Calculate the slopes of the two lines
                slope1 = np.arctan2((y2 - y1), (x2 - x1))
                slope2 = np.arctan2((oy2 - oy1), (ox2 - ox1))
                if np.sign(slope1) != np.sign(slope2):
                    # slopes are opposite, definitely not similar
                    continue
                if self.are_lines_similar(x1, y1, x2, y2, ox1, oy1, ox2, oy2, distance_threshold, angle_threshold):
                    keep_line = False
                    break
            if keep_line:
                filtered_lines.append(line)

        return filtered_lines

    def get_intersect(self, l1, l2):
        l1_x1, l1_y1, l1_x2, l1_y2 = l1
        l2_x1, l2_y1, l2_x2, l2_y2 = l2
        s = np.vstack([[l1_x1, l1_y1], [l1_x2, l1_y2], [l2_x1, l2_y1], [l2_x2, l2_y2]])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return [float('inf'), float('inf')]
        else:
            return [int(x/z), int(y/z)]

    def are_lines_similar(self, x1, y1, x2, y2, ox1, oy1, ox2, oy2, distance_threshold, angle_threshold):

        y_middle = self.current_data_height//2
        y_upper = self.current_data_height
        y_lower = 0
        
        if y2 > y_middle:
            px = self.x_intercept(x1, y1, x2, y2, y_upper)
        else:
            px = self.x_intercept(x1, y1, x2, y2, y_lower)

        if oy2 > y_middle:
            opx = self.x_intercept(ox1, oy1, ox2, oy2, y_upper)
        else:
            opx = self.x_intercept(ox1, oy1, ox2, oy2, y_lower)

        distance_difference = np.abs(px - opx)

        # Calculate the slopes of the two lines
        slope1 = np.arctan2((y2 - y1), (x2 - x1))
        slope2 = np.arctan2((oy2 - oy1), (ox2 - ox1))
        angle_difference = np.abs(slope1 - slope2)%(2*np.pi)

        # Check if both distance and angle difference are below the thresholds
        return distance_difference < distance_threshold and angle_difference < angle_threshold

    def x_intercept(self, x1, y1, x2, y2, y_i):
        # Check if the line is vertical to avoid division by zero
        if y2 == y1:
            raise ValueError("The line defined by the points is horizontal and does not intercept the x-axis.")
        if x1 == x2:
            return int(x1)
        # Calculate the slope
        m = (y2 - y1) / (x2 - x1)
        # Calculate the x-intercept
        x_i = (y_i - y1) / m + x1
        
        return int(x_i)
    
    def plot_diamonds(self):

        fig, ax = plt.subplots()
        ax.imshow(
            self.current_data, 
            cmap='binary',
            aspect='auto',
            origin='lower',
            extent=[
                self.gate_data[0],
                self.gate_data[-1], 
                self.ohmic_data[0], 
                self.ohmic_data[-1]
            ],
        )
        ax.set_title("Coulomb Diamonds")
        ax.set_xlabel("Gate Voltage (V)")
        ax.set_ylabel("Ohmic Voltage (V)")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        for diamond in self.diamonds:
            diamond.print_summary()
            diamond.plot(ax)

        plt.tight_layout()
        plt.show()

    def preprocess_data(self,
                        current_data: ndarray,
                        binary_threshold: float = 0.8,
                        blur_sigma: float = 1.0,
                        blur_kernel: tuple = (3,3)) -> ndarray:

        pass

    def detect_diamond_edges(self,
                             processed_data: ndarray,
                             threshold: float = 1.0,
                             ):

        pass

    def estimate_diamond_slopes(self,
                                ):

        pass

    def estimate_line_intercepts(self,
                                 ):
        
        pass

    def cost_function(self,
                      ):
        
        pass

    def minimize_cost(self,
                      ):
        
        pass

    def plot_diamonds(self,
                      ):
        
        pass

    def acquire_statistics(self,
                           ):

        pass

    

@dataclass
class ThresholdAnalysisResult:
    """Store results for a single threshold value"""
    threshold: float
    num_diamonds: int
    diamonds: list
    avg_addition_voltage: Optional[float] = None
    std_addition_voltage: Optional[float] = None
    avg_lever_arm: Optional[float] = None
    std_lever_arm: Optional[float] = None
    avg_charging_voltage: Optional[float] = None
    std_charging_voltage: Optional[float] = None


def analyze_threshold_sensitivity(
    miner,
    threshold_range: np.ndarray = None,
    debug: bool = False,
    plot_results: bool = True,
    continue_on_error: bool = True
) -> Dict[float, ThresholdAnalysisResult]:
    """
    Analyze how binary threshold affects Coulomb diamond detection.
    
    Parameters:
    -----------
    miner : Miner
        The Miner object with data already loaded
    threshold_range : np.ndarray, optional
        Array of threshold values to test (default: np.linspace(1.05, 1.5, 10))
    debug : bool
        Show detailed debug plots for each threshold
    plot_results : bool
        Show summary plots at the end
    continue_on_error : bool
        If True, continue testing other thresholds when one fails
        
    Returns:
    --------
    results : Dict[float, ThresholdAnalysisResult]
        Dictionary mapping threshold values to analysis results
    """
    
    if threshold_range is None:
        threshold_range = np.linspace(1.05, 1.5, 10)
    
    # Store original threshold
    original_threshold = miner.binary_threshold
    
    results = {}
    successful_count = 0
    failed_count = 0
    
    print(f"Testing {len(threshold_range)} threshold values...")
    print("=" * 60)
    
    for thresh in threshold_range:
        print(f"\nThreshold: {thresh:.3f}")
        print("-" * 40)
        
        # Update threshold
        miner.binary_threshold = thresh
        
        try:
            # Extract diamonds with current threshold
            diamonds = miner.extract_diamonds(debug=debug)
            
            # Compute statistics if diamonds were found
            result = ThresholdAnalysisResult(
                threshold=thresh,
                num_diamonds=len(diamonds),
                diamonds=diamonds
            )
            
            if len(diamonds) > 0:
                addition_voltages = [d.addition_voltage() for d in diamonds]
                lever_arms = [d.lever_arm() for d in diamonds]
                charging_voltages = [d.charging_voltage() for d in diamonds]
                
                result.avg_addition_voltage = np.mean(addition_voltages)
                result.std_addition_voltage = np.std(addition_voltages)
                result.avg_lever_arm = np.mean(lever_arms)
                result.std_lever_arm = np.std(lever_arms)
                result.avg_charging_voltage = np.mean(charging_voltages)
                result.std_charging_voltage = np.std(charging_voltages)
                
                print(f"  ✓ Diamonds detected: {len(diamonds)}")
                print(f"  Avg addition voltage: {result.avg_addition_voltage*1e3:.3f} ± {result.std_addition_voltage*1e3:.3f} mV")
                print(f"  Avg lever arm: {result.avg_lever_arm:.5f} ± {result.std_lever_arm:.5f} eV/V")
            else:
                print(f"  ✓ No diamonds detected (valid result)")
                
            results[thresh] = result
            successful_count += 1
            
        except AssertionError as e:
            error_msg = str(e)
            print(f"  ✗ Detection failed: {error_msg}")
            results[thresh] = ThresholdAnalysisResult(
                threshold=thresh,
                num_diamonds=-1,  # Use -1 to indicate failure vs 0 for no diamonds
                diamonds=[]
            )
            failed_count += 1
            
            if not continue_on_error:
                miner.binary_threshold = original_threshold
                raise
                
        except Exception as e:
            print(f"  ✗ Unexpected error: {str(e)}")
            results[thresh] = ThresholdAnalysisResult(
                threshold=thresh,
                num_diamonds=-1,
                diamonds=[]
            )
            failed_count += 1
            
            if not continue_on_error:
                miner.binary_threshold = original_threshold
                raise
    
    # Restore original threshold
    miner.binary_threshold = original_threshold
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Successful: {successful_count}/{len(threshold_range)}")
    print(f"Failed: {failed_count}/{len(threshold_range)}")
    
    if plot_results:
        plot_threshold_analysis(results, threshold_range, miner)
    
    return results


def plot_threshold_analysis(
    results: Dict[float, ThresholdAnalysisResult],
    threshold_range: np.ndarray,
    miner
):
    """Create comprehensive visualization of threshold sensitivity analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data for plotting (exclude failed runs with num_diamonds = -1)
    thresholds = []
    num_diamonds = []
    failed_thresholds = []
    avg_addition = []
    std_addition = []
    avg_lever = []
    std_lever = []
    avg_charging = []
    std_charging = []
    
    for thresh, result in sorted(results.items()):
        if result.num_diamonds == -1:
            # Mark failed attempts
            failed_thresholds.append(thresh)
            continue
            
        thresholds.append(thresh)
        num_diamonds.append(result.num_diamonds)
        avg_addition.append(result.avg_addition_voltage if result.avg_addition_voltage else np.nan)
        std_addition.append(result.std_addition_voltage if result.std_addition_voltage else np.nan)
        avg_lever.append(result.avg_lever_arm if result.avg_lever_arm else np.nan)
        std_lever.append(result.std_lever_arm if result.std_lever_arm else np.nan)
        avg_charging.append(result.avg_charging_voltage if result.avg_charging_voltage else np.nan)
        std_charging.append(result.std_charging_voltage if result.std_charging_voltage else np.nan)
    
    if len(thresholds) == 0:
        print("No successful threshold values to plot!")
        return
    
    thresholds = np.array(thresholds)
    num_diamonds = np.array(num_diamonds)
    avg_addition = np.array(avg_addition) * 1e3  # Convert to mV
    std_addition = np.array(std_addition) * 1e3
    avg_charging = np.array(avg_charging) * 1e3  # Convert to mV
    std_charging = np.array(std_charging) * 1e3
    avg_lever = np.array(avg_lever)
    std_lever = np.array(std_lever)
    
    # Plot 1: Number of diamonds detected
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(thresholds, num_diamonds, 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax1.fill_between(thresholds, 0, num_diamonds, alpha=0.3)
    
    # Mark failed thresholds
    if failed_thresholds:
        y_max = max(num_diamonds) if len(num_diamonds) > 0 else 1
        for ft in failed_thresholds:
            ax1.axvspan(ft-0.01, ft+0.01, color='red', alpha=0.3, label='Failed' if ft == failed_thresholds[0] else '')
    
    ax1.axvline(miner.binary_threshold, color='red', linestyle='--', 
                label=f'Current threshold ({miner.binary_threshold:.3f})', linewidth=2)
    ax1.set_xlabel('Binary Threshold', fontsize=12)
    ax1.set_ylabel('Number of Diamonds Detected', fontsize=12)
    ax1.set_title('Diamond Detection vs Binary Threshold', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Addition voltage
    ax2 = fig.add_subplot(gs[1, 0])
    valid_mask = ~np.isnan(avg_addition)
    if np.any(valid_mask):
        ax2.errorbar(thresholds[valid_mask], avg_addition[valid_mask], 
                     yerr=std_addition[valid_mask], fmt='o-', linewidth=2, 
                     markersize=6, capsize=4, color='tab:green')
        ax2.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Binary Threshold', fontsize=11)
        ax2.set_ylabel('Avg Addition Voltage (mV)', fontsize=11)
        ax2.set_title('Addition Voltage Stability', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Addition Voltage Stability', fontsize=12, fontweight='bold')
    
    # Plot 3: Lever arm
    ax3 = fig.add_subplot(gs[1, 1])
    valid_mask = ~np.isnan(avg_lever)
    if np.any(valid_mask):
        ax3.errorbar(thresholds[valid_mask], avg_lever[valid_mask], 
                     yerr=std_lever[valid_mask], fmt='o-', linewidth=2, 
                     markersize=6, capsize=4, color='tab:orange')
        ax3.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Binary Threshold', fontsize=11)
        ax3.set_ylabel('Avg Lever Arm (eV/V)', fontsize=11)
        ax3.set_title('Lever Arm Stability', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Lever Arm Stability', fontsize=12, fontweight='bold')
    
    # Plot 4: Charging voltage
    ax4 = fig.add_subplot(gs[1, 2])
    valid_mask = ~np.isnan(avg_charging)
    if np.any(valid_mask):
        ax4.errorbar(thresholds[valid_mask], avg_charging[valid_mask], 
                     yerr=std_charging[valid_mask], fmt='o-', linewidth=2, 
                     markersize=6, capsize=4, color='tab:purple')
        ax4.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Binary Threshold', fontsize=11)
        ax4.set_ylabel('Avg Charging Voltage (mV)', fontsize=11)
        ax4.set_title('Charging Voltage Stability', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Charging Voltage Stability', fontsize=12, fontweight='bold')
    
    # Plot 5: Coefficient of variation (CV) for addition voltage
    ax5 = fig.add_subplot(gs[2, 0])
    valid_mask = (~np.isnan(avg_addition)) & (avg_addition != 0)
    if np.any(valid_mask):
        cv_addition = (std_addition[valid_mask] / np.abs(avg_addition[valid_mask])) * 100
        ax5.plot(thresholds[valid_mask], cv_addition, 'o-', linewidth=2, 
                markersize=6, color='tab:red')
        ax5.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Binary Threshold', fontsize=11)
        ax5.set_ylabel('CV (%)', fontsize=11)
        ax5.set_title('Addition Voltage Variability', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Addition Voltage Variability', fontsize=12, fontweight='bold')
    
    # Plot 6: CV for lever arm
    ax6 = fig.add_subplot(gs[2, 1])
    valid_mask = (~np.isnan(avg_lever)) & (avg_lever != 0)
    if np.any(valid_mask):
        cv_lever = (std_lever[valid_mask] / avg_lever[valid_mask]) * 100
        ax6.plot(thresholds[valid_mask], cv_lever, 'o-', linewidth=2, 
                markersize=6, color='tab:brown')
        ax6.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax6.set_xlabel('Binary Threshold', fontsize=11)
        ax6.set_ylabel('CV (%)', fontsize=11)
        ax6.set_title('Lever Arm Variability', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Lever Arm Variability', fontsize=12, fontweight='bold')
    
    # Plot 7: Combined stability metric
    ax7 = fig.add_subplot(gs[2, 2])
    valid_for_stability = (~np.isnan(avg_addition)) & (~np.isnan(avg_lever)) & (num_diamonds > 0)
    if np.any(valid_for_stability):
        # Get CV values for valid indices
        cv_add_vals = (std_addition / np.abs(avg_addition)) * 100
        cv_lever_vals = (std_lever / avg_lever) * 100
        
        # Only use indices that have valid data for stability calculation
        valid_indices = np.where(valid_for_stability)[0]
        
        norm_diamonds = np.zeros(len(thresholds))
        norm_diamonds[valid_indices] = num_diamonds[valid_indices] / num_diamonds[valid_indices].max() if num_diamonds[valid_indices].max() > 0 else 0
        
        # Normalize CVs (lower CV = better, so we invert)
        max_cv_add = cv_add_vals[valid_indices].max() if len(valid_indices) > 0 and cv_add_vals[valid_indices].max() > 0 else 1
        max_cv_lever = cv_lever_vals[valid_indices].max() if len(valid_indices) > 0 and cv_lever_vals[valid_indices].max() > 0 else 1
        
        norm_cv_add = np.zeros(len(thresholds))
        norm_cv_lever = np.zeros(len(thresholds))
        norm_cv_add[valid_indices] = 1 - (cv_add_vals[valid_indices] / max_cv_add)
        norm_cv_lever[valid_indices] = 1 - (cv_lever_vals[valid_indices] / max_cv_lever)
        
        stability_score = (norm_diamonds + norm_cv_add + norm_cv_lever) / 3 * 100
        
        ax7.plot(thresholds[valid_for_stability], stability_score[valid_for_stability], 
                'o-', linewidth=2, markersize=6, color='tab:cyan')
        ax7.axvline(miner.binary_threshold, color='red', linestyle='--', linewidth=2)
        ax7.fill_between(thresholds[valid_for_stability], 0, 
                        stability_score[valid_for_stability], alpha=0.3)
        ax7.set_xlabel('Binary Threshold', fontsize=11)
        ax7.set_ylabel('Stability Score', fontsize=11)
        ax7.set_title('Overall Stability Metric', fontsize=12, fontweight='bold')
        ax7.grid(alpha=0.3)
        ax7.set_ylim([0, 105])
    else:
        ax7.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Overall Stability Metric', fontsize=12, fontweight='bold')
    
    plt.suptitle('Binary Threshold Sensitivity Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if np.any(~np.isnan(avg_addition)):
        optimal_idx = np.nanargmin(std_addition)
        print(f"\nMost stable addition voltage at threshold: {thresholds[optimal_idx]:.3f}")
        print(f"  Value: {avg_addition[optimal_idx]:.3f} ± {std_addition[optimal_idx]:.3f} mV")
    
    if np.any(~np.isnan(avg_lever)):
        optimal_idx = np.nanargmin(std_lever)
        print(f"\nMost stable lever arm at threshold: {thresholds[optimal_idx]:.3f}")
        print(f"  Value: {avg_lever[optimal_idx]:.5f} ± {std_lever[optimal_idx]:.5f} eV/V")
    
    if len(num_diamonds) > 0:
        max_detection_idx = np.argmax(num_diamonds)
        print(f"\nMaximum diamonds detected at threshold: {thresholds[max_detection_idx]:.3f}")
        print(f"  Number: {num_diamonds[max_detection_idx]}")
    
    current_idx = np.argmin(np.abs(thresholds - miner.binary_threshold)) if len(thresholds) > 0 else None
    if current_idx is not None:
        print(f"\nCurrent threshold: {miner.binary_threshold:.3f}")
        print(f"  Diamonds detected: {num_diamonds[current_idx]}")
    
    if failed_thresholds:
        print(f"\nFailed thresholds: {len(failed_thresholds)}")
        print(f"  Values: {[f'{t:.3f}' for t in failed_thresholds]}")


def compare_threshold_diamonds(
    miner,
    threshold1: float,
    threshold2: float,
    plot: bool = True
) -> Tuple[list, list]:
    """
    Compare diamond detection between two specific threshold values.
    
    Parameters:
    -----------
    miner : Miner
        The Miner object
    threshold1 : float
        First threshold value
    threshold2 : float
        Second threshold value
    plot : bool
        Whether to plot side-by-side comparison
        
    Returns:
    --------
    diamonds1, diamonds2 : tuple of lists
        Diamond lists for each threshold
    """
    
    original_threshold = miner.binary_threshold
    
    # Extract diamonds with first threshold
    miner.binary_threshold = threshold1
    diamonds1 = miner.extract_diamonds(debug=False)
    
    # Extract diamonds with second threshold
    miner.binary_threshold = threshold2
    diamonds2 = miner.extract_diamonds(debug=False)
    
    # Restore original
    miner.binary_threshold = original_threshold
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for ax, diamonds, thresh in zip(axes, [diamonds1, diamonds2], [threshold1, threshold2]):
            ax.imshow(
                miner.current_data,
                cmap='binary',
                aspect='auto',
                origin='lower',
                extent=[
                    miner.gate_data[0],
                    miner.gate_data[-1],
                    miner.ohmic_data[0],
                    miner.ohmic_data[-1]
                ]
            )
            ax.set_title(f'Threshold = {thresh:.3f} ({len(diamonds)} diamonds)', fontsize=14)
            ax.set_xlabel('Gate Voltage (V)', fontsize=12)
            ax.set_ylabel('Ohmic Voltage (V)', fontsize=12)
            
            for diamond in diamonds:
                diamond.plot(ax)
        
        plt.tight_layout()
        plt.show()
    
    return diamonds1, diamonds2