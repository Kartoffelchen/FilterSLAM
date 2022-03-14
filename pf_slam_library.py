
from math import sin, cos, atan2, sqrt, pi
from lego_robot import LegoLogfile
import numpy as np

def write_cylinders(file_desc, line_header, cylinder_list):
    file_desc.writelines(line_header)
    for c in cylinder_list:
        file_desc.writelines("%.1f %.1f " % tuple(c))
        count+=1
    file_desc.writelines("\n")

def write_error_ellipses(file_desc, line_header, covariance_matrix_list): 
    file_desc.writelines(line_header)
    for m in covariance_matrix_list:
        eigenvals, eigenvects = np.linalg.eig(m)
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        file_desc.writelines("%.3f %.1f %.1f" % \
                 (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1])))
    file_desc.writelines("\n")

def compute_derivative(scan, min_dist): #N
    jumps = [ 0 ]
    for i in range(1, len(scan) - 1):
        l = scan[i-1]
        r = scan[i+1]
        if l > min_dist and r > min_dist:
            derivative = (r - l) / 2.0
            jumps.append(derivative)
        else:
            jumps.append(0)
    jumps.append(0)
    return jumps

def find_cylinders(scan, scan_derivative, jump, min_dist): #n
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list

def get_cylinders_from_scan(scan, jump, min_dist, cylinder_offset): #n
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    result = []
    for c in cylinders:
        bearing = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        x, y = distance*cos(bearing), distance*sin(bearing)
        result.append( (np.array([distance, bearing]), np.array([x, y])) )
    return result

def get_mean(particles): #n
    mean_x, mean_y = 0.0, 0.0
    head_x, head_y = 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        mean_x += x
        mean_y += y
        head_x += cos(theta)
        head_y += sin(theta)
    n = max(1, len(particles))
    return np.array([mean_x / n, mean_y / n, atan2(head_y, head_x)])

def get_error_ellipse_and_heading_variance(particles, mean): 
    center_x, center_y, center_heading = mean
    n = len(particles)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0)

    sxx, sxy, syy = 0.0, 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        dx = x - center_x
        dy = y - center_y
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    cov_xy = np.array([[sxx, sxy], [sxy, syy]]) / (n-1)

    var_heading = 0.0
    for p in particles:
        dh = (p.pose[2] - center_heading + pi) % (2*pi) - pi
        var_heading += dh * dh
    var_heading = var_heading / (n-1)

    eigenvals, eigenvects = np.linalg.eig(cov_xy)
    ellipse_angle = atan2(eigenvects[1,0], eigenvects[0,0])

    return (ellipse_angle, sqrt(abs(eigenvals[0])),
            sqrt(abs(eigenvals[1])),
            sqrt(var_heading))

def print_particles(particles, file_desc): 
    if not particles:
        return
    
    file_desc.writelines("PA ")
    for p in particles:
        file_desc.writelines("%.0f %.0f %.3f" % tuple(p.pose))
    file_desc.writelines("\n")
