from numpy import ndarray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass

@dataclass
class CoulombDiamond:
    """Class for keeping track of Coulomb diamond information."""
    name: str
    left_vertex: tuple
    top_vertex: tuple
    right_vertex: tuple
    bottom_vertex: tuple
    e: float = 1.60217663e-19 # C
    eps0: float = 8.8541878128e-12 # F/m
    epsR: float = 11.7 # Silicon

    def width(self) -> float:
        return np.abs(self.right_vertex[0] - self.left_vertex[0])
    
    def height(self) -> float:
        return np.abs(self.top_vertex[1] - self.bottom_vertex[1])

    def lever_arm(self) -> float:
        return self.charging_voltage() / self.addition_voltage() 
    
    def addition_voltage(self) -> float:
        return self.width()
    
    def charging_voltage(self) -> float:
        return self.height() / 2

    def total_capacitance(self) -> float:
        return self.e / self.charging_voltage()
    
    def gate_capacitance(self) -> float:
        return self.e / self.addition_voltage()
    
    def dot_size(self) -> float:
        return self.total_capacitance() / (8 * self.eps0 * self.epsR)

    def print_summary(self):
        print(f"Summary ({self.name}):")
        print("====================")
        print(f"Left Vertex: {self.left_vertex}")
        print(f"Top Vertex: {self.top_vertex}")
        print(f"Right Vertex: {self.right_vertex}")
        print(f"Bottom Vertex: {self.bottom_vertex}")
        print(f"Elementary Charge (e): {self.e:.5e} C")
        print(f"Permittivity of Free Space (\u03F50): {self.eps0:.5e} F/m")
        print(f"Relative Permittivity (\u03F5R): {self.epsR:.5f}")
        print(f"Width: {self.width():.5f} V")
        print(f"Height: {self.height():.5f} V")
        print(f"Lever Arm (\u03B1): {self.lever_arm():.5f} eV/V")
        print(f"Addition Voltage: {self.addition_voltage():.5f} V")
        print(f"Charging Voltage: {self.charging_voltage():.5f} V")
        print(f"Total Capacitance: {self.total_capacitance() * 1e18:.5f} aF")
        print(f"Gate Capacitance: {self.gate_capacitance() * 1e18:.5f} aF")
        print(f"Dot Size: {self.dot_size() * 1e9:.5f} nm")
        print("\n")

    def plot(self, ax):
        vertices = [self.left_vertex, self.top_vertex, self.right_vertex, self.bottom_vertex, self.left_vertex]
        polygon = patches.Polygon(vertices, closed=True, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(polygon)
        
        # Calculate the center of the diamond for the label
        center_x = (self.left_vertex[0] + self.right_vertex[0]) / 2
        center_y = (self.top_vertex[1] + self.bottom_vertex[1]) / 2
        ax.text(center_x, 0, self.name, color='blue', ha='center', va='center', fontsize=10, weight='bold')

class Miner:

    def __init__(self,
                gate_data: ndarray,
                ohmic_data: ndarray,
                current_data: ndarray) -> None:
            
        self.gate_data = gate_data
        self.ohmic_data = ohmic_data
        self.current_data = current_data

        self.current_data_height, self.current_data_width = self.current_data.shape

        self.gate_voltage_per_pixel = (self.gate_data[-1] - self.gate_data[0]) / self.current_data_width
        self.ohmic_voltage_per_pixel = (self.ohmic_data[-1] - self.ohmic_data[0]) / self.current_data_height
    
    def filter_raw_data(self, 
                        current_data: ndarray) -> ndarray:

        filtered_current_data = np.log(
            np.abs(current_data)
        )

        new_data = np.zeros_like(filtered_current_data)

        mask = filtered_current_data < 1.1 * np.nanmax(filtered_current_data)

        new_data[mask] = 255
        
        return new_data.astype(np.uint8)

    def extract_diamonds(self, debug: bool = False) -> None:

        upper_mask = np.zeros_like(self.current_data, dtype=bool)
        upper_mask[self.current_data_height//2:, :] = True

        lower_mask = np.zeros_like(self.current_data, dtype=bool)
        lower_mask[:self.current_data_height//2, :] = True

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
            plt.imshow(self.filter_raw_data(self.current_data), cmap='binary', aspect='auto')


        for section in ["upper", "lower"]:
            image = self.current_data * masks[section]
            image_threshold = self.filter_raw_data(image)
            image_edges = self.extract_edges(image_threshold)
            image_lines = self.extract_lines(image_edges)
            
            # Iterate over points
            for points in image_lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]

                if x1 != x2:
                    slope = (y2 - y1) / (x2 - x1)

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

                    if slope > 0:
                        line_dict[section]["positive"].append([px, py, x_intercept, y_middle])
                    else:
                        line_dict[section]["negative"].append([px, py, x_intercept, y_middle])

            if debug:
                if section == "upper":
                    color = 'blue'
                else:
                    color = "red"
                self.plot_lines(image_lines, c=color)

        upper_pos = sorted(line_dict['upper']['positive'], key = lambda x: x[2])
        upper_neg = sorted(line_dict['upper']['negative'], key = lambda x: x[2])
        lower_pos = sorted(line_dict['lower']['positive'], key = lambda x: x[2])
        lower_neg = sorted(line_dict['lower']['negative'], key = lambda x: x[2])

        assert len(upper_pos) == len(upper_neg), "Unbalanced lines detected in the upper half"
        assert len(lower_pos) == len(lower_neg), "Unbalanced lines detected in the lower half"

        diamond_shapes = []  
        for u_p, u_n, l_p, l_n in zip(upper_pos, upper_neg, lower_pos, lower_neg):
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
                    top_vertex=diamond_vertices_voltage[1],
                    right_vertex=diamond_vertices_voltage[2],
                    bottom_vertex=diamond_vertices_voltage[3],
                )
            )

        return detected_coulomb_diamonds

    def extract_edges(self, image: ndarray) -> ndarray:

        # Apply Gaussian blur to smooth the image and reduce noise
        blurred = cv2.GaussianBlur(
            image, 
            (3,3), 
            1
        )

        # Perform Canny edge detection
        edges = cv2.Canny(
            blurred, 
            0, 
            0, 
            apertureSize=3
        )

        return edges

    def extract_lines(self, edges: ndarray) -> list:

        lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=10, # Min number of votes for valid line
            minLineLength=20, # Min allowed length of line
            maxLineGap=20 # Max allowed gap between line for joining them
            )

        # Filter out duplicate lines
        filtered_lines = self.filter_duplicate_lines(lines, distance_threshold=5, angle_threshold=0.5)

        return filtered_lines

    def filter_duplicate_lines(self, lines, distance_threshold=10, angle_threshold=.1):
        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            keep_line = True
            for other_line in filtered_lines:
                ox1, oy1, ox2, oy2 = other_line[0]
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
        # Calculate the distance between the midpoints of the two lines
        midpoint1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        midpoint2 = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)

        distance_difference = np.abs(midpoint1[0] - midpoint2[0])

        # Calculate the slopes of the two lines
        slope1 = np.arctan2((y2 - y1), (x2 - x1))
        slope2 = np.arctan2((oy2 - oy1), (ox2 - ox1))
        angle_difference = np.abs(slope1 - slope2)

        # Check if both distance and angle difference are below the thresholds
        return distance_difference < distance_threshold and angle_difference < angle_threshold 
      
    def plot_lines(self, lines: list, c: str = 'blue'):

        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]

            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)

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

                if slope > 0:
                    color = "dark" + c
                else:
                    color = c

                plt.plot([px, x_intercept], [py, y_middle], c=color)

    def x_intercept(self, x1, y1, x2, y2, y_i):
        # Check if the line is vertical to avoid division by zero
        if y2 == y1:
            raise ValueError("The line defined by the points is horizontal and does not intercept the x-axis.")
        
        # Calculate the slope
        m = (y2 - y1) / (x2 - x1)
        # Calculate the x-intercept
        x_i = (y_i - y1) / m + x1
        
        return int(x_i)