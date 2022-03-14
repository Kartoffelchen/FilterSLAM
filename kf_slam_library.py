
from math import sin, cos, pi
from lego_robot import LegoLogfile


def write_cylinders(file_desc, line_header, cylinder_list): #y
    file_desc.writelines( line_header )
    for c in cylinder_list:
        file_desc.writelines( "%.1f %.1f" % c)
    file_desc.writelines( "\n" )
    
def write_error_ellipses(file_desc, line_header, error_ellipse_list): #n
    file_desc.writelines( line_header )
    for e in error_ellipse_list:
        file_desc.writelines( "%.3f %.1f %.1f" % e)
    file_desc.writelines(  "\n" )

# Find the derivative in scan data, ignoring invalid measurements.
def compute_derivative(scan, min_dist): #y
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


def find_cylinders(scan, scan_derivative, jump, min_dist): #y
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            # Start a new cylinder, independent of on_cylinder.
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            # Save cylinder if there was one.
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        # Always add point, if it is a valid measurement.
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list

def get_observations(scan, jump, min_dist, cylinder_offset, #y
                     robot,
                     max_cylinder_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    scanner_pose = (
        robot.state[0] + cos(robot.state[2]) * robot.scanner_displacement,
        robot.state[1] + sin(robot.state[2]) * robot.scanner_displacement,
        robot.state[2])

    result = []
    for c in cylinders:
        angle = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        xs, ys = distance*cos(angle), distance*sin(angle)
        x, y = LegoLogfile.scanner_to_world(scanner_pose, (xs, ys))
        best_dist_2 = max_cylinder_distance * max_cylinder_distance
        best_index = -1
        for index in range(robot.number_of_landmarks):
            pole_x, pole_y = robot.state[3+2*index : 3+2*index+2]
            dx, dy = pole_x - x, pole_y - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_index = index
        result.append(((distance, angle), (x, y), (xs, ys), best_index))

    return result
