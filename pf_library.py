
from math import sin, cos, pi
from lego_robot import LegoLogfile

def compute_derivative(scan, min_dist): # y
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
        if scan_derivative[i] <  -jump:
            
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            # Save cylinder if there was one.
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
        result.append( (distance, bearing, x, y) )
    return result

def assign_cylinders(cylinders, robot_pose, scanner_displacement,  #n
                     reference_cylinders):
    
    scanner_pose = (robot_pose[0] + cos(robot_pose[2]) * scanner_displacement,
                    robot_pose[1] + sin(robot_pose[2]) * scanner_displacement,
                    robot_pose[2])

    
    result = []
    for c in cylinders:
        
        x, y = LegoLogfile.scanner_to_world(scanner_pose, c[2:4])
        
        best_dist_2 = 1e300
        best_ref = None
        for ref in reference_cylinders:
            dx, dy = ref[0] - x, ref[1] - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_ref = ref
        
        if best_ref:
            result.append((c[0:2], best_ref))

    return result
